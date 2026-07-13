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

# Parameter bounds matching type.ACRModel fields
PARAM_BOUNDS = [
    (-5.0, 5.0),      # exposure
    (-100, 100),      # contrast
    (-100, 100),      # highlights
    (-100, 100),      # shadows
    (-100, 100),      # whites
    (2000, 50000),    # temperature
    (-150, 150),      # tint
    (-100, 100),      # vibrance
    (-100, 100),      # saturation
    (-100, 100),      # texture
    (-100, 100),      # clarity
    (-100, 100),      # dehaze
    (0, 360),         # shadows_hsl.hue
    (0, 100),         # shadows_hsl.saturation
    (0, 100),         # shadows_hsl.luminance
    (0, 360),         # midtones_hsl.hue
    (0, 100),         # midtones_hsl.saturation
    (0, 100),         # midtones_hsl.luminance
    (0, 360),         # highlights_hsl.hue
    (0, 100),         # highlights_hsl.saturation
    (0, 100),         # highlights_hsl.luminance
]

IMAGE_SCALE = 0.1
FIELD_NAMES = [
    "exposure",
    "contrast",
    "highlights",
    "shadows",
    "whites",
    "temperature",
    "tint",
    "vibrance",
    "saturation",
    "texture",
    "clarity",
    "dehaze",
    "shadows_hsl.hue",
    "shadows_hsl.saturation",
    "shadows_hsl.luminance",
    "midtones_hsl.hue",
    "midtones_hsl.saturation",
    "midtones_hsl.luminance",
    "highlights_hsl.hue",
    "highlights_hsl.saturation",
    "highlights_hsl.luminance",
]

DISABLED_FIELDS = [
    "temperature",
]

def normalize_to_bounds(x_norm, lo, hi):
    # x_norm in [-1,1] -> map to [lo,hi]
    return lo + (x_norm + 1.0) * 0.5 * (hi - lo)


def action_to_acr(action):
    # action: 42 values in [-1,1] => 21 masks followed by 21 values
    action = np.asarray(action, dtype=np.float32).flatten()
    expected_size = len(PARAM_BOUNDS) * 2
    if action.size != expected_size:
        raise ValueError(f"Expected action length {expected_size}, got {action.size}")

    disabled_fields = set(DISABLED_FIELDS)

    mask_values = action[: len(PARAM_BOUNDS)]
    param_values = action[len(PARAM_BOUNDS) :]

    acr_dict = {}
    hsl_groups = {
        "shadows_hsl": {
            "mask_indices": (12, 13, 14),
            "value_indices": (12, 13, 14),
        },
        "midtones_hsl": {
            "mask_indices": (15, 16, 17),
            "value_indices": (15, 16, 17),
        },
        "highlights_hsl": {
            "mask_indices": (18, 19, 20),
            "value_indices": (18, 19, 20),
        },
    }

    for index, (field_name, mask_value, param_value, bounds) in enumerate(zip(FIELD_NAMES, mask_values, param_values, PARAM_BOUNDS)):
        if field_name.startswith("shadows_hsl.") or field_name.startswith("midtones_hsl.") or field_name.startswith("highlights_hsl."):
            continue

        if field_name in disabled_fields:
            continue

        if mask_value <= 0:
            continue

        lo, hi = bounds
        normalized_value = normalize_to_bounds(float(param_value), lo, hi)

        if field_name == "exposure":
            acr_dict[field_name] = normalized_value
        else:
            acr_dict[field_name] = int(round(normalized_value))

    for hsl_name, group_info in hsl_groups.items():
        if hsl_name in disabled_fields:
            continue

        group_masks = [mask_values[i] > 0 for i in group_info["mask_indices"]]
        if any(group_masks) and not all(group_masks):
            continue
        if not all(group_masks):
            continue

        hsl_values = {}
        for field_index, field_suffix in zip(group_info["value_indices"], ("hue", "saturation", "luminance")):
            lo, hi = PARAM_BOUNDS[field_index]
            normalized_value = normalize_to_bounds(float(param_values[field_index]), lo, hi)
            hsl_values[field_suffix] = int(round(normalized_value))

        acr_dict[hsl_name] = hsl_values

    return ACRModel(**acr_dict)




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
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(len(PARAM_BOUNDS) * 2,), dtype=np.float32)

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
        acr = action_to_acr(action)

        ACR_mockup.apply_acr_to_pp3(acr, 'temp.pp3')
        self.input_image.save('temp_in.jpg')
        ACR_mockup.render('temp_in.jpg', 'temp.pp3', 'temp_out.jpg')

        rendered_image = Image.open('temp_out.jpg')
        if rendered_image.size != self.target_image.size:
            rendered_image = rendered_image.resize(self.target_image.size, Image.Resampling.LANCZOS)
        rendered_array = np.asarray(rendered_image)
        target_array = np.asarray(self.target_image)
        reward = image_similarity(rendered_array, target_array)
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
        self.current_image = Image.open('temp_out.jpg')

        self.step_count += 1
        terminated = True
        truncated = self.step_count >= self.max_steps
        obs = np.zeros((1,), dtype=np.float32)
        return obs, reward, terminated, truncated, {"per_pair_scores": per_pair_scores}

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

    acr = action_to_acr(best_action)

    print(f'\nUsing {best_source}:')
    print('\nBest global ACR settings (approx):')
    print(acr.model_dump_json(indent=2))
    print(f'\nAverage similarity score across dataset: {average_score:.4f}')

    for pair_index, ((input_path, target_path), score) in enumerate(zip(env.dataset_pairs, per_pair_scores), start=1):
        print(f'Pair {pair_index}/{len(env.dataset_pairs)}: {os.path.basename(input_path)} -> {os.path.basename(target_path)} | score={score:.4f}')

    model.save('image_filter_rl')


if __name__ == '__main__':
    LEARNING_RATE = 0.0003
    N_STEPS = 2048
    BATCH_SIZE = 64
    N_EPOCHS = 10
    TOTAL_TIMESTEPS = 3000
    SEED = 42
    main(LEARNING_RATE, N_STEPS, BATCH_SIZE, N_EPOCHS, TOTAL_TIMESTEPS, SEED)


