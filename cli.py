import json
import os
import sys
import numpy as np
from ACR_mockup import ACRModel, action_to_acr, apply_acr_to_pp3, render
import shutil
from uuid import uuid4
import click
from PIL import Image, ImageOps

from image import image_similarity


rl_action_path = "rl_best_action.json"
if hasattr(sys, '_MEIPASS'):
    rl_action_path = os.path.join(sys._MEIPASS, "rl_best_action.json")  # pyright: ignore[reportAttributeAccessIssue]

@click.group()
def cli():
    pass

@cli.command("apply")
@click.option('--i', type=click.Path(exists=True))
@click.option('--o', type=click.Path(), default='output.jpg', help='Output image path')
def apply_command(i, o):
    apply(i, o)

def apply(i, o):
    # Load the ACR model from the JSON file
    with open(rl_action_path, "r") as f:
        acr_model = action_to_acr(json.load(f))


    uuid = uuid4().hex

    # Preprocess input image to apply EXIF orientation
    img = Image.open(i)
    img = ImageOps.exif_transpose(img)
    corrected_input = f"{uuid}_corrected.jpg"
    img.save(corrected_input, "JPEG")

    # Apply the ACR model to a pp3 file
    apply_acr_to_pp3(acr_model, f"{uuid}.pp3", input_image_path=corrected_input)

    # Render the image using the pp3 file
    render(corrected_input, f"{uuid}.pp3", o)

    os.remove(f"{uuid}.pp3")
    os.remove(corrected_input)

@cli.command("list")
def list_actions_command():
    list_actions()

def list_actions():
    # Load the ACR model from the JSON file
    with open(rl_action_path, "r") as f:
        weights = json.load(f)

    acr_model = action_to_acr(weights)

    for field, value in acr_model.model_dump().items():
        if value is not None:
            if field == "shadows_hsl" or field == "midtones_hsl" or field == "highlights_hsl":
                for subfield, subvalue in value.model_dump().items():
                    if subvalue is not None:
                        print(f"{field}.{subfield}: {subvalue}")

            else:
                if field == "temperature_shift" and value > 0:
                    print(f"{field}: +{value}")
                else:
                    print(f"{field}: {value}")

@cli.command("histogram")
@click.option('--i', type=click.Path(exists=True))
def histogram_command(i):
    screen = histogram(i)
    # Print histogram
    height = shutil.get_terminal_size().lines - 5  # Leave some space for labels
    columns = shutil.get_terminal_size().columns
    for y in range(height):
        line = ""
        for x in range(columns):
            if screen[y, x] == 1:
                line += "\033[41m \033[0m"  # Red
            elif screen[y, x] == 2:
                line += "\033[42m \033[0m"  # Green
            elif screen[y, x] == 3:
                line += "\033[44m \033[0m"  # Blue
            else:
                line += " "
        print(line)

def histogram(i):
    # Get tui dimensions
    terminal_size = shutil.get_terminal_size()
    image = Image.open(i)
    array = np.array(image)
    # Calculate histogram
    red_hist, _ = np.histogram(array[:, :, 0], bins=terminal_size.columns, range=(0, 255))
    green_hist, _ = np.histogram(array[:, :, 1], bins=terminal_size.columns, range=(0, 255))
    blue_hist, _ = np.histogram(array[:, :, 2], bins=terminal_size.columns, range=(0, 255))

    height = terminal_size.lines - 5  # Leave some space for labels

    screen = np.zeros((height, terminal_size.columns), dtype=np.uint8)
    for x in range(terminal_size.columns):
        r = int(red_hist[x] / red_hist.max() * height)
        g = int(green_hist[x] / green_hist.max() * height) + r
        b = int(blue_hist[x] / blue_hist.max() * height) + g

        for y in range(height):
            if y < r:
                screen[height - 1 - y, x] = 1  # Red
            elif y < g:
                screen[height - 1 - y, x] = 2  # Green
            elif y < b:
                screen[height - 1 - y, x] = 3  # Blue

    return screen

@cli.command("score")
@click.option('--i', type=click.Path(exists=True))
@click.option('--o', type=click.Path(exists=True))
def score_command(i, o):
    score_value = score(i, o)
    print(f"Score: {score_value}")

def score(i, o):
    if not os.path.exists(i) or not os.path.exists(o):
        print("Input or output image path does not exist.")
        return
    uuid = uuid4().hex

    # Preprocess input image to apply EXIF orientation
    img = Image.open(i)
    img = ImageOps.exif_transpose(img)
    corrected_input = f"{uuid}_corrected.jpg"
    img.save(corrected_input, "JPEG")

    # Load the ACR model from the JSON file
    with open(rl_action_path, "r") as f:
        acr_model = action_to_acr(json.load(f))

    # Apply the ACR model to a pp3 file
    apply_acr_to_pp3(acr_model, f"{uuid}.pp3", input_image_path=corrected_input)

    # Render the image using the pp3 file
    render(corrected_input, f"{uuid}.pp3", f"{uuid}.jpg")

    os.remove(f"{uuid}.pp3")
    os.remove(corrected_input)

    image1 = Image.open(o)
    image2 = Image.open(f"{uuid}.jpg")

    score = image_similarity(np.array(image1), np.array(image2))
    os.remove(f"{uuid}.jpg")


    return score

if __name__ == "__main__":
    cli()