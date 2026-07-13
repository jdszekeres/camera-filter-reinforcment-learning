# Camera Filter Preset Generator
## A reinforcement learning model to create Adobe Camera Raw presets 

### Usage

#### 1. Collecting data
Take photos both with and without the filter (make sure they are the same image resolution) and add them to a `dataset/input` and `dataset/output` folder. In the `dataset` folder, create a `photos.json` file with an array of photo entries in the following format 

```json
{
    "input": "input\\filename.jpg",
    "output": "output\\filename.jpg"
}
```

**Note that `\\` should be used on Windows machines and `/` should be used on Mac and Linux machines as the file seperator**


### 2. Training the model
The model can be run by simply running main.py. To change hyperparameters, look for the `if __name__ == '__main__'` section at the end of the file

### 3. Using the data
Upon completion of training, the RL model will output the most optimal parameters it has found. To use these settings, simply input them into the correspoding fields in Adobe camera raw.