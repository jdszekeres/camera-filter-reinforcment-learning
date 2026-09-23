import math
import os
import subprocess
from cli import apply
import numpy as np
from PIL import Image, ImageOps
from uuid import uuid4



INPUT = "C:\\Users\\jdsze\\Pictures\\LJ\\web\\IMG_3219_1.jpeg"
OUTPUT = f"{uuid4()}.jpg"
apply(INPUT, OUTPUT)

INPUT_IMAGE = ImageOps.exif_transpose(Image.open(INPUT))
OUTPUT_IMAGE = ImageOps.exif_transpose(Image.open(OUTPUT))
def frame(out, radius):
    img = INPUT_IMAGE.copy().convert("RGB")
    img.rotate(90, expand=True)
    width, height = img.size
    for x in range(width):
        for y in range(height):
            dx = x - width // 2
            dy = y - height // 2
            distance = np.sqrt(dx ** 2 + dy ** 2)
            if distance < radius:
                pixel = OUTPUT_IMAGE.getpixel((x, y))
            else:
                pixel = INPUT_IMAGE.getpixel((x, y))
            if pixel is not None:
                img.putpixel((x, y), pixel)
    img.save(out, "JPEG")


FPS = 30
img_files = []
max_radius = round(math.sqrt(INPUT_IMAGE.size[0] ** 2 + INPUT_IMAGE.size[1] ** 2) // 2)
inc = 10
os.makedirs("temp_frames", exist_ok=True)
for i in range(1, max_radius // inc + 1):
    out = f"output_{i}.jpg"
    rad = i * inc
    frame(out, rad)
    img_files.append(out)



subprocess.call([
    "ffmpeg", "-y",
    "-framerate", str(FPS),
    "-i", "output_%d.jpg",
    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "output.mp4",
])
os.remove(OUTPUT)
for img_file in img_files:
    os.remove(img_file)

os.rmdir("temp_frames")
