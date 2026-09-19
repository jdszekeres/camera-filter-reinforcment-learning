import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import json
import os
from type import ACRModel
import ACR_mockup
from image import image_similarity
import math

IMAGE_SCALE = 0.1




class ImageFilterEnv(gym.Env):
    def __init__(self, dataset_pairs, image_scale=IMAGE_SCALE, max_steps=5):
        super().__init__()
        if not dataset_pairs:
            raise ValueError('Dataset must contain at least one input/output pair.')

        self.dataset_pairs = dataset_pairs
        self.image_scale = image_scale
        self.max_steps = 1

        self.input_image, self.target_image = self._load_pair(0)

        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(len(ACR_mockup.PARAM_BOUNDS) * 2,), dtype=np.float32)

        self.current_image = self.input_image.copy()
        self.step_count = 0

    def _load_pair(self, pair_index):
        input_path, target_path = self.dataset_pairs[pair_index]
        input_image = Image.open(input_path).convert('RGB')
        target_image = Image.open(target_path).convert('RGB')

        if input_image.height > input_image.width:
            input_image = input_image.transpose(Image.Transpose.ROTATE_270)
        if target_image.height > target_image.width:
            target_image = target_image.transpose(Image.Transpose.ROTATE_270)

        input_size = (
            max(1, int(self.image_scale * input_image.width)),
            max(1, int(self.image_scale * input_image.height)),
        )
        target_size = (
            max(1, int(self.image_scale * target_image.width)),
            max(1, int(self.image_scale * target_image.height)),
        )

        return input_image.resize(input_size, Image.Resampling.LANCZOS), target_image.resize(target_size, Image.Resampling.LANCZOS)

    def _score_action_on_pair(self, action, pair_index):
        self.input_image, self.target_image = self._load_pair(pair_index)
        acr = ACR_mockup.action_to_acr(action)

        self.input_image.save('temp_in.jpg')
        ACR_mockup.apply_acr_to_pp3(acr, 'temp.pp3', input_image_path='temp_in.jpg')

        try:
            saved = False
            for attempt in range(3):
                try:
                    ACR_mockup.render('temp_in.jpg', 'temp.pp3', 'temp_out.jpg')
                    saved = True
                    break
                except Exception:
                    pass
            if saved:
                with Image.open('temp_out.jpg') as rendered_image_file:
                    rendered_image = rendered_image_file.copy()
                    rendered_image_file.close()

                if rendered_image.size != self.target_image.size:
                    rendered_image = rendered_image.resize(self.target_image.size, Image.Resampling.LANCZOS)

                rendered_array = np.asarray(rendered_image)
                target_array = np.asarray(self.target_image)
                reward = image_similarity(rendered_array, target_array)
                self.current_image = rendered_image.copy()
            else:
                # Continue even if save failed after 3 retries
                reward = 0.0
                rendered_image = self.input_image.copy()
        finally:
            if os.path.exists('temp_in.jpg'):
                os.remove('temp_in.jpg')
            if os.path.exists('temp_out.jpg'):
                os.remove('temp_out.jpg')

        return reward, rendered_image, acr

    def evaluate_action(self, action):
        per_pair_scores = []
        for pair_index in range(len(self.dataset_pairs)):
            score, _, _ = self._score_action_on_pair(action, pair_index)
            per_pair_scores.append(score)
        return float(np.mean(per_pair_scores)), per_pair_scores

    def reset(self, *, seed=None, options=None):
        self.step_count = 0
        return np.zeros((1,), dtype=np.float32), {}

    def step(self, action):
        reward, per_pair_scores = self.evaluate_action(action)

        self.step_count += 1
        terminated = True
        truncated = self.step_count >= self.max_steps
        obs = np.zeros((1,), dtype=np.float32)
        return obs, reward, terminated, truncated, {"per_pair_scores": per_pair_scores}

    def render_action_to_file(self, action, pair_index, output_path):
        _, rendered_image, _ = self._score_action_on_pair(action, pair_index)
        rendered_image.save(output_path)
        return rendered_image

    def render(self, mode='human'):
        self.current_image.show()


class BestActionCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.best_reward = float("-inf")
        self.best_action = None

    def _on_step(self) -> bool:
        actions = self.locals.get("actions")
        rewards = self.locals.get("rewards")
        
        if actions is None or rewards is None:
            return True

        action_array = np.asarray(actions)
        reward_array = np.asarray(rewards).reshape(-1)
        if action_array.ndim == 1:
            action_array = action_array.reshape(1, -1)

        best_index = int(np.argmax(reward_array))
        best_step_reward = float(reward_array[best_index])
        if best_step_reward > self.best_reward:
            self.best_reward = best_step_reward
            self.best_action = np.array(action_array[best_index], dtype=np.float32, copy=True)
        return True


def main(LEARNING_RATE, N_STEPS, BATCH_SIZE, N_EPOCHS, TOTAL_TIMESTEPS, SEED):

    dataset_file = 'dataset/photos.json'

    if not os.path.exists(dataset_file):
        raise FileNotFoundError('Dataset manifest not found in workspace root')

    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset_rows = json.load(f)

    dataset_root = os.path.dirname(dataset_file)
    dataset_pairs = [
        (
            os.path.join(dataset_root, row['input']),
            os.path.join(dataset_root, row['output']),
        )
        for row in dataset_rows
    ]

    env = ImageFilterEnv(dataset_pairs)


    model = PPO('MlpPolicy', env, verbose=2, device='cpu', learning_rate=LEARNING_RATE, n_steps=N_STEPS, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, seed=SEED)
    best_action_callback = BestActionCallback()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True, callback=best_action_callback)

    obs, _ = env.reset()
    deterministic_action, _ = model.predict(obs, deterministic=True)
    deterministic_average_score, deterministic_pair_scores = env.evaluate_action(deterministic_action)

    if best_action_callback.best_action is not None:
        best_training_average_score, best_training_pair_scores = env.evaluate_action(best_action_callback.best_action)
    else:
        best_training_average_score = float("-inf")
        best_training_pair_scores = []

    if deterministic_average_score >= best_training_average_score:
        best_action = deterministic_action
        average_score = deterministic_average_score
        per_pair_scores = deterministic_pair_scores
        best_source = "deterministic policy output"
    else:
        best_action = best_action_callback.best_action
        average_score = best_training_average_score
        per_pair_scores = best_training_pair_scores
        best_source = "best action seen during training"

    acr = ACR_mockup.action_to_acr(best_action)

    print(f'\nUsing {best_source}:')
    print('\nBest global ACR settings (approx):')
    print(acr.model_dump_json(indent=2))
    print(f'\nAverage similarity score across dataset: {average_score:.4f}')

    for pair_index, ((input_path, target_path), score) in enumerate(zip(env.dataset_pairs, per_pair_scores), start=1):
        print(f'Pair {pair_index}/{len(env.dataset_pairs)}: {os.path.basename(input_path)} -> {os.path.basename(target_path)} | score={score:.4f}')

    final_output_path = 'rl_generated_output.jpg'
    env.render_action_to_file(best_action, 0, final_output_path)
    print(f'\nSaved rendered preview to {final_output_path}')

    # model.save('image_filter_rl')
    if best_action is None:
        print("No best action found during training.")
    else:
        f = open('rl_best_action.json', 'w', encoding='utf-8')
        f.write(json.dumps(best_action.tolist(), indent=2))
        f.close()


if __name__ == '__main__':
    LEARNING_RATE = 0.0003
    N_STEPS = 2048
    BATCH_SIZE = 64
    N_EPOCHS = 20
    TOTAL_TIMESTEPS = math.pow(2, 12)  # 4096 steps
    SEED = 42
    main(LEARNING_RATE, N_STEPS, BATCH_SIZE, N_EPOCHS, TOTAL_TIMESTEPS, SEED)
