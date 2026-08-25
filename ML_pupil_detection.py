from pathlib import Path
from tkinter import Tk, filedialog

import cv2
import numpy as np

from video_frame_transform import FrameTransform

# Global variables
output_folder = "training_data_right_eye"
saved_img_counter = 235
frame_counter = 0

# Rotation applied to each frame before cropping. Must match the rect below,
# since the rect is defined in the rotated frame's coordinates.
# One of: "none", "clockwise", "counterclockwise", "180"
ROTATION = "clockwise"

# Aspect ratio (1080:648 = 5:3) matches the model's 320x192 input, so the
# resize below does not distort the pupil.
RECT = {"x": 0, "y": 50, "w": 1080, "h": 648}


def crop_frame(frame, rect):

    x = rect['x']
    y = rect['y']
    w = rect['w']
    h = rect['h']

    cropped_frame = frame[y:y+h, x:x+w]

    return cropped_frame

def gray_and_resize(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_resize = cv2.resize(gray, (320, 192), interpolation=cv2.INTER_AREA)

    return gray_resize

def save_training_data(frame):
    global saved_img_counter

    saved_img_counter += 1
    path = f"{output_folder}/image_{saved_img_counter}.jpg"
    if not cv2.imwrite(path, frame):
        raise RuntimeError(f"Could not write training image: {path}")



def access_file():

    root = Tk()
    root.withdraw() # Hide main window

    # Allowed file types
    file_types = [("Video Files", "*.mp4")]

    print("Select video file")

    # Open file selector
    root.update()
    video_path = filedialog.askopenfilename(
        title="Select a Video File", filetypes=file_types
    )

    root.destroy()
    return video_path

def main():
    global frame_counter

    path = access_file()
    video = cv2.VideoCapture(path)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width, height)

    transform = FrameTransform(rotation=ROTATION)
    rotated_width, rotated_height = transform.output_size((width, height))
    if RECT["x"] + RECT["w"] > rotated_width or RECT["y"] + RECT["h"] > rotated_height:
        raise ValueError(
            f"rect {RECT} does not fit inside the "
            f"{rotated_width}x{rotated_height} rotated frame"
        )

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    while True:
        # read frame
        ongoing, frame = video.read()
        # break if video has ended
        if ongoing == False:
            break

        frame = transform.apply_frame(frame)
        frame = crop_frame(frame, RECT)
        frame = gray_and_resize(frame)

        if frame_counter % 5 == 0:
            save_training_data(frame)

        cv2.imshow('frame', frame)
        frame_counter += 1

        # 30 ms between frames, press Q to exit
        key = cv2.waitKey(30)
        if key == 113:
            break

    cv2.destroyAllWindows()

main()