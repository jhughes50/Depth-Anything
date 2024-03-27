import cv2 
import copy
import numpy as np


class StructureFromMotion:

    def __init__(self):
        self.img1_ = np.array([])
        self.img2_ = np.array([]) 

        self.kp1_ = np.array([])
        self.kp2_ = np.array([])
        
        self.desc1_ = None
        self.desc2_ = None

        self.p1_ = np.array([])
        self.p2_ = np.array([])

        self.p1_depth_ = None
        self.p2_depth_ = None

        self.sift_ = cv2.SIFT_create()
        self.flann_ = cv2.FlannBasedMatcher(dict(algorithm = 0, trees = 5),
                                            dict(checks = 50))


    @property
    def first(self):
        return self.img1_
    
    @first.setter
    def first(self, first):
        self.img1_ = first
        self.kp1_, self.desc1_ = self.sift_.detectAndCompute(first, None)

    @property
    def second(self):
        return self.img2_

    @second.setter
    def second(self, second):
        self.img2_ = second

    def next(self, img, K):
        self.img2_ = img
        self.kp2_, self.desc2_ = self.sift_.detectAndCompute(self.img2_, None)
    
        matches = self.flann_.knnMatch(self.desc1_, self.desc2_, k=2)

        good_match = list()

        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good_match.append(m)

        if len(good_match) > 10:
            self.p1_ = np.float32([ self.kp1_[m.queryIdx].pt for m in good_match ]).reshape(-1,1,2)
            self.p2_ = np.float32([ self.kp2_[m.trainIdx].pt for m in good_match ]).reshape(-1,1,2)
        
        E, mask = cv2.findEssentialMat(self.p1_, self.p2_, K, cv2.RANSAC, 0.999, 1.0);
        matches_mask = mask.ravel().tolist()

        points, R, t, mask = cv2.recoverPose(E, self.p1_, self.p2_)

        p1 = self.p1_[np.asarray(matches_mask)==1,:,:]
        p2 = self.p2_[np.asarray(matches_mask)==1,:,:]
        p1_un = np.squeeze(p1)
        p2_un = np.squeeze(p2)

        M_r = np.hstack((R, t))
        M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))

        P_l = K @ M_l
        P_r = K @ M_r

        point_4d_hom = cv2.triangulatePoints(P_l, P_r, p1_un.T, p2_un.T)
        point_3d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
        point_3d = point_3d[:3, :].T

        self.p1_depth_ = np.zeros((p1_un.shape[0],3), dtype=np.float32)
        self.p2_depth_ = np.zeros((p1_un.shape[0],3), dtype=np.float32)

        i = 0
        for im, p3d in zip(p1_un, point_3d): 
            d = np.linalg.norm(p3d)
            self.p1_depth_[i] = np.array([int(im[0]),int(im[1]),d])
            i += 1

        i = 0
        for im, p3d in zip(p2_un, point_3d):
            d = np.linalg.norm(p3d)
            self.p2_depth_[i] = np.array([int(im[0]),int(im[1]),d])
            i += 1

        self.kp1_ = copy.copy(self.kp2_)
        self.desc1_ = copy.copy(self.desc2_)

        return self.p1_depth_, self.p2_depth_ 
