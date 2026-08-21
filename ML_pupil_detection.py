import cv2
from tkinter import Tk, filedialog
import numpy as np

# Global variables
output_folder = "training_data"
saved_img_counter = 235
frame_counter = 0

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
    cv2.imwrite(f"{output_folder}/image_{saved_img_counter}.jpg", frame)



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
    #rect = {'x':100, 'y':100, 'w':980, 'h':600} #for 1080x1920
    rect = {'x':0, 'y':150, 'w':500, 'h':300}

    while True:
        # read frame
        ongoing, frame = video.read()
        # break if video has ended
        if ongoing == False:
            break

        frame = crop_frame(frame, rect)
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