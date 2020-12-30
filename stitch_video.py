from os import stat
import cv2
import numpy as np
import matplotlib.pyplot as plt

fpath = "../data/clipped.avi"

def save_frames(fpath):
    cap = cv2.VideoCapture(fpath)

    frames = list()

    counter = 1
    step = 25
    # stitched = np.array([])
    while cap.isOpened():
        ret, frame = cap.read()
        # cv2.putText(gray, "frame #: {}".format(counter), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
        counter += 1
        if(ret and counter%step == 0):  # the next round of new frame
            new_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
            frames.append(new_frame)
        if counter>700:
            break
    cap.release()
    cv2.destroyAllWindows()

    return frames

def stitch_frames(fpath):
    frames = save_frames(fpath)
    print(len(frames))
    # plt.imshow(cv2.hconcat(frames))
    # plt.show()

    # stitcher = cv2.Stitcher_create(mode=cv2.Stitcher_PANORAMA)
    # status, stitched = stitcher.stitch(frames)
    # if(status == 0):
    #     cv2.imshow("stitched", stitched)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    stitch_incremental(frames)

def stitch_incremental(imgs):
    """ stitch imgs incrementally based on time stamps """
    stitcher = cv2.Stitcher_create(mode=cv2.STITCHER_PANORAMA)
    for t in range(2, len(imgs)):
        status, stitched  = stitcher.stitch(imgs[:t])
        # cv2.imshow("tile_{}".format(t), cv2.hconcat(imgs[:t]))
        # plt.imshow(cv2.hconcat(imgs[:t]))
        # plt.show()
        if(status == 0):
            cv2.imshow("stitched_{}".format(t), stitched)
        else:
            print(status)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



def stitch_video(fpath):
    """ stitch video 10 frames apart """
    stitcher = cv2.Stitcher_create(mode=cv2.Stitcher_PANORAMA)
    # stitcher.setFeaturesMatcher(cv2.detail_AffineBestOf2NearestMatcher)
    cap = cv2.VideoCapture(fpath)
    ret, frame = cap.read()
    old_frame = frame
    height, width, channel = frame.shape
    old_mask = np.zeros((height, width), np.uint8)*255
    old_mask[:, -(width//2):] = 255
    # put frame into a image
    # tmp1 = cv2.bitwise_and(old_frame, old_frame, mask=old_mask)
    # cv2.imshow("old", tmp1)
    # cv2.waitKey(0)


    counter = 1
    step = 50
    # stitched = np.array([])
    while cap.isOpened():
        ret, frame = cap.read()
        # cv2.putText(gray, "frame #: {}".format(counter), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
        counter += 1
        if(ret and counter%step == 0):  # the next round of new frame
            new_frame = frame
            new_mask = np.ones((height, width), np.uint8)*255
            new_mask[:, -(width//2):] = 0
            cv2.imshow("new", new_frame)
            # cv2.imshow("new", cv2.bitwise_and(new_frame, new_frame, mask=new_mask))
            status, stitched = stitcher.stitch([old_frame, new_frame])
            if(status == 0):
                # cv2.imshow("stitched", stitched)
                old_frame = stitched
                old_mask = np.zeros_like(old_frame[:,:,0], np.uint8)
                # wipe out left
                old_mask[:, -(width//2):] = 255
                cv2.imshow("old", cv2.bitwise_and(old_frame, old_frame, mask=old_mask))
                cv2.imshow("old_cropped", old_frame[:,-(width//2):,:])
            else:  # failed to stitch
                print("failed to stitch")
                break

            if cv2.waitKey(1) & 0xFF == ord('c'):
                continue
            elif cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.imwrite('../data/output.jpg', old_frame)


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    stitch_frames(fpath)
