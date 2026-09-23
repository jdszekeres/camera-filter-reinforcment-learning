# Camera Filter Preset Generator
## A reinforcement learning model to create Adobe Camera Raw presets 



### Usage

### Try it!
<video src="demo.mp4" controls autoplay onloadstart="this.playbackRate = 4;"></video>

I've trained a model putting my sunglasses in front of the camera. To try it:

- Download `cli.exe` from [releases](https://github.com/jdszekeres/camera-filter-reinforcment-learning/releases)
- Follow the instruction in `3. Using the data`, replacing `python cli.py` with `cli.exe`


### 0. Installtion

Install the latest version of RawTherapee (At least version 5.12) and ensure it is on your PATH (C:\\Program Files\\RawTherapee\\\<version\> on Windows or /Applications/RawTherapee.app/Contents/MacOS/rawtherapee-cli on Mac)

Install project requirements 
```python
pip install -r requirements.txt
```

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
The model weights are saved to rl_best_action.json, use `cli.py` to get useful data:

```
python cli.py apply

Applys the filter to an image

--i: The input image
--o=output.jpg: The output image

python cli.py list

Lists the settings found by the model

python cli.py histogram

Shows an ascii histogram distribution of color brightness in an image

--i: The input image

python cli.py score

Scores how an input image with a filter applied compares to another image

--i: The input image to apply the filter
--o: The image to compare to
```



