import cv2 as cv
import numpy as np

class Stitch2:

    finder = cv.ORB.create()
    matcher = cv.detail.BestOfNearestMarcher_create(False, 0.3)
    warp_type = "cylindrical"
    p = list()
    last_p = list()
    estimator = cv.detail_HomographyBasedEstimator()
    adjustor = cv.detail_BundleAdjusterRay()
    seam_finder = cv.detail_DpSeamFinder("COLOR_GRAD")
    all_cameras = list()
    focals = list()
    rmats = list()
    work_scale = 0.6
    seam_scale = 0.25
    conf_thresh = 0.3

    seam_work_aspect = 1
    full_img_sizes = []
    features = []
    images = []
    is_work_scale_set = False
    is_seam_scale_set = False
    is_compose_scale_set = False
    do_wave_correct = True

    images = list()
    features = list()
    
    def __init__(self, img1):
        # first image push to list
        self._get_refine_mask()
        self._getWarper()
        self._getCompensator()
        self._getMask(img1)
        self.__appendImg(img1)

    def __appendImg(self, full_img):
        self.full_img_sizes.append((full_img.shape[1], full_img.shape[0]))
        self._getMask(full_img)

        if self.work_megapix < 0:
            img = full_img
            self.work_scale = 1
            self.is_work_scale_set = True
        else:
            if self.is_work_scale_set is False:
                self.work_scale = min(1.0, np.sqrt(self.work_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                self.is_work_scale_set = True
            img = cv.resize(src=full_img, dsize=None, fx=self.work_scale, fy=self.work_scale, interpolation=cv.INTER_LINEAR_EXACT)
        if self.is_seam_scale_set is False:
            self.seam_scale = min(1.0, np.sqrt(self.seam_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
            self.seam_work_aspect = self.seam_scale / self.work_scale
            self.is_seam_scale_set = True
        img_feat = cv.detail.computeImageFeatures2(self.finder, img)
        self.features.append(img_feat)
        img = cv.resize(src=full_img, dsize=None, fx=self.seam_scale, fy=self.seam_scale, interpolation=cv.INTER_LINEAR_EXACT)
        self.images.append(img)


    def getNew(self, img2):
        # get new images and calculate all
        self.__appendImg(img2)
        img_feat = cv.detail.computeImageFeatures2(self.finder, img2)
        self.features.append(img_feat)

    def matchImg(self):
        # get matcher for last two features
        _p = self.matcher.apply2(self.features[-2:])
        self.last_p = _p
        self.p.append(_p) # append the maching info to previous info
        self.matcher.collectGarbage()

    def _get_refine_mask(self):
        # set mask
        # apply to last two features? 
        self.refine_mask = np.zeros((3, 3), np.uint8)
        self.refine_mask[0, 0] = 1
        self.refine_mask[0, 1] = 1
        self.refine_mask[0, 2] = 1
        self.refine_mask[1, 1] = 1
        self.refine_mask[1, 2] = 1
        


    def estimateImg(self):
        # cameras is the last two camera
        b, cameras = self.estimator.apply(self.features[-2:], self.last_p, None)
        if not b:
            print("Homography estimation failed.")
            exit()
        for cam in cameras:
            cam.R = cam.R.astype(np.float32)

        self.adjustor.setConfThresh(0.3)
        self.adjuster.setRefinementMask(self.refine_mask)
        b, cameras = self.adjuster.apply(self.features[-2:], self.last_p, cameras)

        if not b:
            print("Camera parameters adjusting failed.")
            return
        # append last two/one? camera to all_cameras
        self.all_cameras.append(cameras[-1])

        # append last two/one? camera's focals
        for cam in cameras:
            self.focals.append(cam.focal)
            self.rmats.append(np.copy(cam.R))

        

        self.focals.sort()
        if len(self.focals) % 2 == 1: # get median
            self.warped_image_scale = self.focals[len(self.focals) // 2]
        else:
            self.warped_image_scale = (self.focals[len(self.focals) // 2] + self.focals[len(self.focals) // 2 - 1]) / 2

    corners = []
    masks_warped = []
    images_warped = []
    images_warped_f = []
    sizes = []
    masks = []

    def waveCorrect(self):
        if self.do_wave_correct:
            self.rmats = cv.detail.waveCorrect(self.rmats, cv.detail.WAVE_CORRECT_HORIZ)
            for idx, cam in enumerate(self.all_cameras):
                cam.R = self.rmats[idx]

    def _getMask(self, img):
        um = cv.UMat(255 * np.ones((img.shape[0], img.shape[1]), np.uint8))
        self.masks.append(um)

    def _getWarper(self):
        self.warper = cv.PyRotationWarper(self.warp_type,
                                     self.warped_image_scale * self.seam_work_aspect) 
                                      # warper could be nullptr?

    def getNewMaskWarped(self):
        idx = -1
        K = self.all_cameras[idx].K().astype(np.float32)
        swa = self.seam_work_aspect
        K[0, 0] *= swa
        K[0, 2] *= swa
        K[1, 1] *= swa
        K[1, 2] *= swa
        corner, image_wp = self.warper.warp(self.images[idx], K, 
                                        self.all_cameras[idx].R,
                                         cv.INTER_LINEAR, cv.BORDER_REFLECT)
        self.corners.append(corner)
        self.sizes.append((image_wp.shape[1], image_wp.shape[0]))
        self.images_warped.append(image_wp)
        p, mask_wp = self.warper.warp(self.masks[idx], K, 
                                    self.all_cameras[idx].R, 
                                    cv.INTER_NEAREST, cv.BORDER_CONSTANT)
        self.masks_warped.append(mask_wp.get())

        imgf = image_wp.astype(np.float32)
        self.images_warped_f.append(imgf)

    def _getCompensator(self):
        expos_comp_block_size = 32
        expos_comp_nr_feeds = 1
        self.compensator = cv.detail_BlocksChannelsCompensator(
            expos_comp_block_size, expos_comp_block_size,
            expos_comp_nr_feeds
        )

    def applyCompensator(self):
        self.compensator.feed(
            corners=self.corners, images=self.images_warped, masks=self.masks_warped)

    def applySeamFinder(self):
        self.seam_finder.find(self.images_warped_f, self.corners, self.masks_warped)

    def applyBlender(self):
        for idx, full_img in enumerate(self.images):
            if not self.is_compose_scale_set:
                if self.compose_megapix > 0:
                    compose_scale = min(1.0, 
                                        np.sqrt(self.compose_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                self.is_compose_scale_set = True
                compose_work_aspect = self.compose_scale / self.work_scale
                self.warped_image_scale *= compose_work_aspect
                # a new warper
                warper = cv.PyRotationWarper(self.warp_type, self.warped_image_scale)
                for i in range(0, len(img_names)):
                    cameras[i].focal *= compose_work_aspect
                    cameras[i].ppx *= compose_work_aspect
                    cameras[i].ppy *= compose_work_aspect
                    sz = (full_img_sizes[i][0] * compose_scale, full_img_sizes[i][1] * compose_scale)
                    K = cameras[i].K().astype(np.float32)
                    roi = warper.warpRoi(sz, K, cameras[i].R)
                    corners.append(roi[0:2])
                    sizes.append(roi[2:4])




    def getAllMasksWarped(self):
        warper = cv.PyRotationWarper(self.warp_type,
                                     self.warped_image_scale * self.seam_work_aspect)  # warper could be nullptr?

        num_images = self.images.count                             
        for idx in range(0, num_images):
            K = self.all_cameras[idx].K().astype(np.float32)
            swa = self.seam_work_aspect
            K[0, 0] *= swa
            K[0, 2] *= swa
            K[1, 1] *= swa
            K[1, 2] *= swa
            corner, image_wp = warper.warp(self.images[idx], K, 
                                           self.all_cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
            self.corners.append(corner)
            self.sizes.append((image_wp.shape[1], image_wp.shape[0]))
            self.images_warped.append(image_wp)
            p, mask_wp = warper.warp(self.masks[idx], K, 
                                     self.all_cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
            self.masks_warped.append(mask_wp.get())

            for img in self.images_warped:
                imgf = img.astype(np.float32)
                self.images_warped_f.append(imgf)