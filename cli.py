import json
import os
import numpy as np
from ACR_mockup import ACRModel, action_to_acr, apply_acr_to_pp3, render
import shutil
from uuid import uuid4
import click
from PIL import Image

from image import image_similarity

@click.group()
def cli():
    pass

@cli.command("apply")
@click.option('--i', type=click.Path(exists=True))
@click.option('--o', type=click.Path(), default='output.jpg', help='Output image path')
def apply(i, o):
    # Load the ACR model from the JSON file
    with open("rl_best_action.json", "r") as f:
        acr_model = action_to_acr(json.load(f))
    

    uuid = uuid4().hex

    # Apply the ACR model to a pp3 file
    apply_acr_to_pp3(acr_model, f"{uuid}.pp3", input_image_path=i)

    # Render the image using the pp3 file
    render(i, f"{uuid}.pp3", o)

    os.remove(f"{uuid}.pp3")

@cli.command("list")
def list_actions():
    # Load the ACR model from the JSON file
    with open("rl_best_action.json", "r") as f:
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

    # Print histogram
    for y in range(height):
        line = ""
        for x in range(terminal_size.columns):
            if screen[y, x] == 1:
                line += "\033[41m \033[0m"  # Red
            elif screen[y, x] == 2:
                line += "\033[42m \033[0m"  # Green
            elif screen[y, x] == 3:
                line += "\033[44m \033[0m"  # Blue
            else:
                line += " "
        print(line)

@cli.command("score")
@click.option('--i', type=click.Path(exists=True))
@click.option('--o', type=click.Path(exists=True))
def score(i, o):
    if not os.path.exists(i) or not os.path.exists(o):
        print("Input or output image path does not exist.")
        return
    uuid = uuid4().hex
    # Load the ACR model from the JSON file
    with open("rl_best_action.json", "r") as f:
        acr_model = action_to_acr(json.load(f))

    # Apply the ACR model to a pp3 file
    apply_acr_to_pp3(acr_model, f"{uuid}.pp3", input_image_path=i)

    # Render the image using the pp3 file
    render(i, f"{uuid}.pp3", f"{uuid}.jpg")

    os.remove(f"{uuid}.pp3")

    image1 = Image.open(o)
    image2 = Image.open(f"{uuid}.jpg")

    score = image_similarity(np.array(image1), np.array(image2))
    os.remove(f"{uuid}.jpg")


    print(f"Image similarity: {score}")

if __name__ == "__main__":
    cli()