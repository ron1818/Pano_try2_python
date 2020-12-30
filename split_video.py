import cv2
import numpy as np

fpath = "../data/video.mp4"

def split_video(fpath, step=40):
    """ split video to image sequences """
    cap = cv2.VideoCapture(fpath)

    counter = 1
    start=50
    end=700
    while cap.isOpened():
        ret, frame = cap.read()
        if(ret and counter%step == 0 and start<counter<=end):  # the next round of new frame
            new_frame = frame
            cv2.imwrite('../data/split_{}.jpg'.format(counter), new_frame)
            cv2.imshow("new", new_frame)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                continue
            elif cv2.waitKey(1) & 0xFF == ord('q'):
                break
        elif(counter>end):
            break
        counter += 1


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    split_video(fpath)
