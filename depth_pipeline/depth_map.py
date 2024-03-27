"""
Jason Hughes
March 2024
Module to get depth map from depth anything
As a part of the SfM pipeline
"""

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from depth_pipeline.structure_from_motion import StructureFromMotion
import torch
from torchvision.transforms import Compose
import torch.nn.functional as F
import numpy as np
import cv2


class DepthRecovery(StructureFromMotion):

    def __init__(self):
        super().__init__()
        
        self.model_ = None
        self.transform_ = None

        self.device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("[DepthMap] Using device: %s" %self.device_)

    def initDepthAnything(self):

        self.model_ = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format('vitb')).to(self.device_).eval()
        self.transform_ = Compose([Resize(width=518,
                                    height=518,
                                    resize_target=False,
                                    keep_aspect_ratio=True,
                                    ensure_multiple_of=14,
                                    resize_method='lower_bound',
                                    image_interpolation_method=cv2.INTER_CUBIC,),
                             NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                             PrepareForNet(),])


    @torch.no_grad()
    def getRelativeDepth(self):

        img1 = cv2.cvtColor(self.img1_, cv2.COLOR_BGR2RGB) / 255
        img2 = cv2.cvtColor(self.img2_, cv2.COLOR_BGR2RGB) / 255

        h, w = img1.shape[:2]
        
        img1 = self.transform_({'image': img1})['image']
        img2 = self.transform_({'image': img2})['image']

        img1 = torch.from_numpy(img1).unsqueeze(0).to(self.device_)
        img2 = torch.from_numpy(img2).unsqueeze(0).to(self.device_)

        depth1 = self.model_(img1)
        depth2 = self.model_(img2)

        depth1 = F.interpolate(depth1[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth2 = F.interpolate(depth2[None], (h, w), mode='bilinear', align_corners=False)[0, 0]

        #depth1 = self.inverseDisparity(depth1.cpu().numpy())
        #depth2 = self.inverseDisparity(depth2.cpu().numpy())

        return depth1, depth2


    def inverseDisparity(self, depth):
        range1 = np.minimum (depth.max() / (depth.min() + 0.001), 100.0)
        max1 = depth.max()
        min1 = max1 / range1

        d1 = 1 / np.maximum(depth, min1)
        d1 = (d1 - d1.min()) / (d1.max() - d1.min())
        #d1 = np.power(d1, 1.0/2.2) # optional gamma correction
        #d1 = d1 * 65535.0

        return d1#.astype("uint16")

    def sparseToDense(self):
        """
        Get the nearest depth point from SfM 
        """
        pass

    def sparseToDenseMidas(self, depth):
        """
        Use the method from MiDaS to extract
        metric depth
        """
        pass


