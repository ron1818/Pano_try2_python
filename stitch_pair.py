import cv2
import time
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# read images
# image_names = glob("../data/splits/*.jpg")
image_names = glob("../data/stitching/boat*.jpg")
# image_names = glob("../data/test1/DSC_*.jpg")
imgs = []
for i in image_names:
    _i = cv2.imread(i)
    _i = cv2.resize(_i, (0,0), fx=0.5, fy=0.5)
    imgs.append(_i)


def stitch_pair_cascade(imgs):
    stitcher = cv2.Stitcher_create(mode=cv2.STITCHER_PANORAMA)
    old_img = imgs[0]
    for i in range(1, len(imgs)):
        new_img = imgs[i]
        status, stitched = stitcher.stitch([old_img, new_img])
        # cv2.imshow("tile", cv2.hconcat([old_img, new_img]))
        if(status == 0):
            cv2.imshow("stitched", stitched)
            print(stitched.shape)
            old_img = stitched
        else:
            print('failed to stitch')
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("../data/test_stitched.jpg", stitched)


def stitch_pair_sequence(imgs):
    stitcher = cv2.Stitcher_create(mode=cv2.STITCHER_PANORAMA)
    for i, j in zip(range(len(imgs)-1), range(1, len(imgs))):
            status, stitched = stitcher.stitch([imgs[i], imgs[j]])
            cv2.imshow("tile_{}_{}".format(i,j), cv2.hconcat([imgs[i], imgs[j]]))
            if(status == 0):
                cv2.imshow("stitched_{}_{}".format(i, j), stitched)
            else:
                print(status)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def stitch_incremental(imgs):
    """ stitch imgs incrementally based on time stamps """
    stitcher = cv2.Stitcher_create(mode=cv2.STITCHER_PANORAMA)
    for t in range(2, len(imgs)):
        status, stitched  = stitcher.stitch(imgs[:t])
        # cv2.imshow("tile_{}".format(t), cv2.hconcat(imgs[:t]))
        plt.imshow(cv2.hconcat(imgs[:t]))
        plt.show()
        if(status == 0):
            cv2.imshow("stitched_{}".format(t), stitched)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def stitch_window(imgs):
    """ stitch img t-1 and img t paste """
    stitcher = cv2.Stitcher_create(mode=cv2.STITCHER_PANORAMA)
    # first stitch
    status, stitched  = stitcher.stitch(imgs[:2])
    if(status == 0):
        cv2.imshow("stitched_{}".format(2), stitched)
    # subsequent
    for t in range(2, len(imgs)):
        print(t)
        # first 2 stitched image size
        height1, width1, channel = stitched.shape
        # new frames' size
        height2, width2, channel = imgs[t].shape
        # crop stitch into 2 patches
        # mask1 = np.zeros(stitched.shape[:2], np.uint8)
        # mask1[:, (width1-width2):] = 1
        # mask2 = np.ones(imgs[t].shape[:2], np.uint8)
        window2 = stitched[:, (width1-width2):, :]
        window1 = stitched[:, :(width1-width2), :]
        try:
            status, stitched  = stitcher.stitch([window2, imgs[t]])
        except:
            print("error")
        
        if(status != 0):
            print(status)
            continue
        # paste stitched and window1 together
        window1 = cv2.resize(window1, (window1.shape[1], stitched.shape[0]))
        stitched = cv2.resize(stitched, (stitched.shape[1], window1.shape[0]))
        stitched = cv2.hconcat([window1, stitched])

        cv2.imshow("tile_{}".format(t), cv2.hconcat(imgs[:t]))
        # plt.imshow(cv2.hconcat([window2, imgs[t]]))
        # plt.show()
        if(status == 0):
            cv2.imshow("stitched_{}".format(t), stitched)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



def stitch_all(imgs):
    stitcher = cv2.Stitcher_create(mode=cv2.STITCHER_PANORAMA)
    status, stitched  = stitcher.stitch(imgs)
    if(status == 0):
        cv2.imshow("stitched", stitched)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def stitch_debug(imgs):
    window2 = cv2.imread("window2.jpg")
    stitcher = cv2.Stitcher_create(mode=cv2.STITCHER_PANORAMA)
    status, stitched  = stitcher.stitch([window2, imgs[4]])
    cv2.imshow("s", stitched)
    cv2.waitKey()
    cv2.destroyAllWindows()
    print(status)




if __name__ == "__main__":
    # stitch_pair_sequence(imgs)
    # stitch_incremental(imgs)
    # stitch_all(imgs)
    stitch_window(imgs)
    # stitch_debug(imgs)