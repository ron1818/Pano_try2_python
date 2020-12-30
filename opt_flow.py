import numpy as np
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('../data/clipped.avi')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

counter = 0
step = 3
while(1):
    ret,frame = cap.read()
    counter+=1
    if(counter % step != 0):
        continue

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    if(p1 is None):
        break

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # get mean movement vector
    delta = good_new - good_old
    delta_mean = np.mean(delta, axis=0)
    delta_std = np.std(delta, axis=0)
    direction = np.arctan2(delta_mean[1], delta_mean[0])
    amplitude = np.hypot(*delta_mean)
    print(amplitude, direction)

    # sift
    sift = cv2.SIFT_create()
    kp1 = [cv2.KeyPoint(x=f[0], y=f[1], _size=20) for f in good_old]
    kp2 = [cv2.KeyPoint(x=f[0], y=f[1], _size=20) for f in good_new]
    # kp1= sift.detect(img1, None)
    # kp2= sift.detect(img2, None)
    kp1, des1 = sift.compute(old_gray, kp1)
    kp2, des2 = sift.compute(frame_gray, kp2)
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

        # dst = cv2.warpPerspective(old_gray,H,((old_gray.shape[1] + frame_gray.shape[1]), frame_gray.shape[0])) #wraped image
        dst = cv2.warpPerspective(old_gray,H,(0,0))
        # dst[0:frame_gray.shape[0], 0:frame_gray.shape[1]] = frame_gray #stitched image
        fig, ax = plt.subplots(3,1)
        ax[0].imshow(old_gray)
        ax[1].imshow(frame_gray)
        ax[2].imshow(dst)
        plt.show()
    else:
        raise AssertionError('Can’t find enough keypoints.')


    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)

    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()