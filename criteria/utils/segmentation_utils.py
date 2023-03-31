import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms


class FaceSegmentation:
    """
    This class is a wrapper for generating segmentation output from
    face images. It uses a pretrained BiSeNet model from this repository
    https://github.com/zllrunning/face-parsing.PyTorch
    """

    part_to_mask_idx = {
        "background": 0,
        "skin": 1,
        "l_brow": 2,
        "r_brow": 3,
        "l_eye": 4,
        "r_eye": 5,
        "eyeglass": 6,
        "l_ear": 7,
        "r_ear": 8,
        "earrings": 9,
        "nose": 10,
        "mouth": 11,
        "u_lip": 12,
        "l_lip": 13,
        "neck": 14,
        "necklace": 15,
        "cloth": 16,
        "hair": 17,
        "hat": 18
    }

    preprocess_transformation = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    def __init__(self, face_bisenet, device):
        """
        Parameters
        ----------
        face_bisenet: pretrained model for face segmentation
        device: torch device used for performing the computation
        """
        self.face_bisenet = face_bisenet
        self.device = device
        self.face_bisenet = self.face_bisenet.to(device)

    def zero_grad(self):
        self.face_bisenet.zero_grad()

    @torch.no_grad()
    def predict(self, pil_images, one_hot=False):
        """
        Parameters
        ----------
        pil_images: {list | pil Image}, list of pil images
            or a single to predict the face segmentation
        one_hot, bool, default=False
            Whether to return the result in one_hot mode or idx mode
        """
        if isinstance(pil_images, list):
            x = [self.preprocess_transformation(pil_image) for pil_image in pil_images]
        else:
            x = [self.preprocess_transformation(pil_images)]

        x = torch.stack(x, dim=0)
        x = x.to(self.device)
        # TODO: Add one_hot functionality
        # TODO: Change prediction to mini batches to avoid out of memory error
        y = self.face_bisenet(x)[0]
        y = y.detach()
        y = torch.argmax(y, dim=1, keepdim=True)
        return y


class StuffSegmentation:
    """
    This class is a wrapper for generating segmentation output from
    Natural images. It uses a pretrained Deeplabv3 model from this repository
    https://github.com/kazuto1011/deeplab-pytorch
    The model is pretrained on COCO-Stuff
    """
    config_path = "data/cocostuff/"

    preprocess_transformation = transforms.Compose([
        transforms.Resize((513, 513)),
        transforms.ToTensor(),
        transforms.Normalize((0.481, 0.458, 0.408), (1 / 255.0, 1 / 255.0, 1 / 255.0)),
    ])

    @staticmethod
    def load_coco_stuff_config(config_path):
        labels = open(os.path.join(config_path, "labels.txt")).readlines()
        part_to_mask_idx = {}
        for label in labels:
            idx, part = label.rstrip().split("\t")
            part_to_mask_idx[part] = int(idx)

        part_hierarchy = yaml.safe_load(open(os.path.join(config_path, "cocostuff_hierarchy.yaml")))
        return part_to_mask_idx, part_hierarchy

    def __init__(self, deeplabv2_resnet101, config_path, device):
        """
        Parameters
        ----------
        deeplabv2_resnet101: pretrained model for face segmentation
        config_path: path to the folder containing 'labels.txt', and
            'cocostuff_hierarchy.yaml' files.
        device: torch device used for performing the computation
        """
        self.deeplabv2_resnet101 = deeplabv2_resnet101
        self.config_path = config_path
        self.device = device
        self.deeplabv2_resnet101 = self.deeplabv2_resnet101.to(device)
        self.part_to_mask_idx, self.part_hierarchy = self.load_coco_stuff_config(self.config_path)

    def zero_grad(self):
        self.deeplabv2_resnet101.zero_grad()

    @torch.no_grad()
    def predict(self, pil_images, one_hot=False):
        """
        Parameters
        ----------
        pil_images: {list | pil Image}, list of pil images
            or a single to predict the face segmentation
        one_hot, bool, default=False
            Whether to return the result in one_hot mode or idx mode
        """
        if isinstance(pil_images, list):
            x = [self.preprocess_transformation(pil_image) for pil_image in pil_images]
        else:
            x = [self.preprocess_transformation(pil_images)]

        x = torch.stack(x, dim=0)
        x = x.to(self.device)
        # TODO: Add one_hot functionality
        # TODO: Change prediction to mini batches to avoid out of memory error
        H, W = pil_images[0].size

        # Image -> Probability map
        y = self.deeplabv2_resnet101(x)
        y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
        y = y.detach()
        y = torch.argmax(y, dim=1, keepdim=True)
        return y


class GANLinearSegmentation:
    """
    This class is a wrapper for segmentation using the paper
        'Linear Semantics in Generative Adversarial Networks'.
        Arxiv: 'https://arxiv.org/abs/2104.00487'
        Github: 'https://github.com/AtlantixJJ/LinearGAN'
    Linear Segmentation based on GAN features
    """

    face_part_to_mask_idx = {
        "background": 0,
        "skin": 1,
        "nose": 2,
        "eyeglass": 3,
        "eyes": 4,
        "eyebrows": 5,
        "ears": 6,
        "mouth": 7,
        "u_lip": 8,
        "l_lip": 9,
        "hair": 10,
        "hat": 11,
        "earrings": 12,
        "neck": 13,
        "cloth": 14,
    }

    church_part_to_mask_idx = {
        "church": 1,
        "sky": 2,
        "tree": 3,
        "road": 4,
        "grass": 5,
        "sidewalk": 6,
        "person": 7,
        "earth": 8,
        "plant": 9,
        "car": 10,
        "stairs": 11,

    }

    data_to_part_to_mask_idx = {
        "face": face_part_to_mask_idx,
        "church": church_part_to_mask_idx
    }

    def __init__(self, linear_segmentation_model, device, data_source='face'):
        """
        Parameters
        ----------
        linear_segmentation_model: pretrained segmentation model on top of GAN features.
        device: torch device used for performing the computation
        data_source: {'face', 'church'}
            This determines the part_to_mask_idx to interpret the segmentation output.
        """
        self.linear_segmentation_model = linear_segmentation_model
        self.device = device
        self.linear_segmentation_model = self.linear_segmentation_model.to(device)
        assert data_source in ['face', 'church']
        self.data_source = data_source
        self.part_to_mask_idx = self.data_to_part_to_mask_idx[self.data_source]

    @torch.no_grad()
    def predict(self, batch_data, one_hot=False):
        """
        Parameters
        ----------
        batch_data: dict, sample generated by our gan sample generator wrapper.
            return_all_layers should be set to True.
        one_hot, bool, default=False
            Whether to return the result in one_hot mode or idx mode
        """
        # They use a different indexing and do not consider constant input of GAN
        features = [batch_data[f'layer_{l + 1}'] for l in self.linear_segmentation_model.layers]
        # The last element contains the aggregated result
        y = self.linear_segmentation_model(features)[-1].detach()
        y = torch.argmax(y, dim=1, keepdim=True)
        # TODO: Implement one_hot=True

        return y

    def zero_grad(self):
        self.linear_segmentation_model.zero_grad()
