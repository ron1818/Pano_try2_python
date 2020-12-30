import cv2
import numpy as np

fpath = "../data/video.mp4"

def clip_video(fpath):
    """ clip video to a smaller clip """
    cap = cv2.VideoCapture(fpath)
    ret, frame = cap.read()
    height, width, channel = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('../data/clipped.avi',fourcc, 20, (width, height))

    counter = 0
    start = 50
    end = 700
    while cap.isOpened():
        if(counter>end):
            break
        ret, frame = cap.read()
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if(start<counter<=end): # save video
            out.write(frame)
        counter += 1


if __name__ == "__main__":
    clip_video(fpath)