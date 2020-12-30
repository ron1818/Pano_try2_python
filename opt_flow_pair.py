from glob import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

# read images
image_names = glob("../data/splits/*.jpg")
# image_names = glob("../data/stitching/boat*.jpg")
# image_names = glob("../data/test1/DSC_*.jpg")
imgs = []
for i in image_names:
    _i = cv2.imread(i)
    _i = cv2.resize(_i, (0,0), fx=0.5, fy=0.5)
    imgs.append(_i)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (45,45),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def opt_flow_pair(img1, img2):
    # calculate first img's gftt
    p0 = cv2.goodFeaturesToTrack(img1, mask = None, **feature_params)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None, **lk_params)
    if(p1 is None):
        return 0

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # color = np.random.randint(0,255,(100,3))
    # mask = np.zeros_like(img1)
    # # draw the tracks
    # for i,(new,old) in enumerate(zip(good_new,good_old)):
    #     a,b = new.ravel()
    #     c,d = old.ravel()
    #     mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    #     frame = cv2.circle(img2,(a,b),5,color[i].tolist(),-1)
    # img = cv2.add(frame,mask)

    cv2.imshow('frame',img)
    
    # sift
    sift = cv2.SIFT_create()
    kp1 = [cv2.KeyPoint(x=f[0], y=f[1], _size=20) for f in good_old]
    kp2 = [cv2.KeyPoint(x=f[0], y=f[1], _size=20) for f in good_new]
    # kp1= sift.detect(img1, None)
    # kp2= sift.detect(img2, None)
    kp1, des1 = sift.compute(img1, kp1)
    kp2, des2 = sift.compute(img2, kp2)
    des1 = np.float32(des1)
    des2 = np.float32(des2)

    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(des1,des2, k=2)

    # FLANN parameters: Fast Library for Approximate Nearest Neighbor
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # K=2 => get 2 Nearest Neighbors which is then filtered out after applying a ratio test
    # This filters out around 90% of false matches
    #(Learning OpenCV 3 Computer Vision with Python By Joe Minichino, Joseph Howse)
    matches = flann.knnMatch(des1, des2, k=2)                 # Interest points from image2 to image1

    good = []
    for m in matches:
        if (m[0].distance < 0.5*m[1].distance):
            good.append(m)
    matches = np.asarray(good)


    if (len(matches[:,0]) >= 4):
        src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

        dst = cv2.warpPerspective(img1,H,((img1.shape[1] + img2.shape[1]), img2.shape[0])) #wraped image
        dst[0:img2.shape[0], 0:img2.shape[1]] = img2 #stitched image
        cv2.imwrite('output.jpg',dst)
        plt.imshow(dst)
        plt.show()
    else:
        raise AssertionError('Can’t find enough keypoints.')

    # get mean movement vector
    delta = good_new - good_old
    delta_mean = np.mean(delta, axis=0)
    delta_std = np.std(delta, axis=0)
    direction = np.arctan2(delta_mean[1], delta_mean[0])
    amplitude = np.hypot(*delta_mean)
    print(amplitude, direction)

    # match and warp

if __name__ == "__main__":
    img1 = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(imgs[1], cv2.COLOR_BGR2GRAY)
    opt_flow_pair(img1, img2)
