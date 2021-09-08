import copy
import os
import pickle
import sys
import tempfile
import time
from argparse import Namespace
from pathlib import Path

import clip
import cog
import dlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
import torchvision.transforms as transforms
from PIL import Image

sys.path.insert(0, "/content")
sys.path.insert(0, "/content/encoder4editing")

from encoder4editing.models.psp import pSp
from encoder4editing.utils.alignment import align_face
from encoder4editing.utils.common import tensor2im

os.chdir("global_directions")
sys.path.insert(0, ".")

from dnnlib import tflib
from manipulate import Manipulator
from MapTS import GetBoundary, GetDt, GetFs

class Predictor(cog.Predictor):
    def setup(self):

        print("starting setup")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(
            "ViT-B/32", device=self.device, jit=False
        )

        self.graph = tf.get_default_graph()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self.sess = tf.Session(
            graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options)
        )

        self.experiment_args = {"model_path": "e4e_ffhq_encode.pt"}
        self.experiment_args["transform"] = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.resize_dims = (256, 256)

        model_path = self.experiment_args["model_path"]

        ckpt = torch.load(model_path, map_location="cpu")
        opts = ckpt["opts"]
        # pprint.pprint(opts)  # Display full options used
        # update the training options
        opts["checkpoint_path"] = model_path
        opts = Namespace(**opts)

        self.net = pSp(opts)
        self.net.eval()
        self.net.cuda()

        self.shape_predictor = dlib.shape_predictor(
            "/content/shape_predictor_68_face_landmarks.dat"
        )

        with self.graph.as_default(), self.sess.as_default():
            #tflib.init_tf()

            self.M = Manipulator(dataset_name="ffhq", sess=self.sess)
            self.fs3 = np.load("npy/ffhq/fs3.npy")
            np.set_printoptions(suppress=True)

        print("setup complete")

    @cog.input("input", type=Path, help="Input image")
    @cog.input("neutral", type=str, help="Neutral image description")
    @cog.input("target", type=str, help="Target image description")
    @cog.input(
        "manipulation_strength",
        type=float,
        min=-10,
        max=10,
        default=4.1,
        help="The higher the manipulation strength, the closer the generated image becomes to the target description. Negative values moves the generated image further from the target description",
    )
    @cog.input(
        "disentanglement_threshold",
        type=float,
        min=0.08,
        max=0.3,
        default=0.15,
        help="The higher the disentanglement threshold, the more specific the changes are to the target attribute. Lower values mean that broader changes are made to the input image",
    )
    def predict(
        self,
        input,
        neutral,
        target,
        manipulation_strength,
        disentanglement_threshold,
    ):

        # @title Align image
        #original_image = Image.open(str(input))
        #original_image = original_image.convert("RGB")
        input_image = self.run_alignment(str(input))
        #input_image = original_image
        input_image = input_image.resize(self.resize_dims)

        img_transforms = self.experiment_args["transform"]
        transformed_image = img_transforms(input_image)

        with torch.no_grad():
            images, latents = self.run_on_batch(transformed_image.unsqueeze(0))
            result_image, latent = images[0], latents[0]

            print("latents", latents)

        print(transformed_image.shape, result_image.shape)

        w_plus = latents.cpu().detach().numpy()
        with self.graph.as_default(), self.sess.as_default():
            dlatents_loaded = self.M.W2S(w_plus)

        #print("w_plus, dlatents_loaded", w_plus, dlatents_loaded)

        img_index = 0
        w_plus=latents.cpu().detach().numpy()
        with self.graph.as_default(), self.sess.as_default():
            dlatents_loaded=self.M.W2S(w_plus)

        img_indexs=[img_index]
        dlatent_tmp=[tmp[img_indexs] for tmp in dlatents_loaded]
        with self.graph.as_default(), self.sess.as_default():
            self.M.num_images = len(img_indexs)
            self.M.alpha = [0]
            self.M.manipulate_layers = [0]

        with self.graph.as_default(), self.sess.as_default():
            codes, out = self.M.EditOneC(0, dlatent_tmp)

        original = Image.fromarray(out[0, 0]).resize((512, 512))
        with self.graph.as_default(), self.sess.as_default():
            self.M.manipulate_layers = None

        classnames = [target, neutral]
        dt = GetDt(classnames, self.model)

        with self.graph.as_default(), self.sess.as_default():
            self.M.alpha = [manipulation_strength]
            boundary_tmp2, c = GetBoundary(
                self.fs3, dt, self.M, threshold=disentanglement_threshold
            )
            codes = self.M.MSCode(dlatent_tmp, boundary_tmp2)
            out = self.M.GenerateImg(codes)
        generated = Image.fromarray(out[0, 0])  # .resize((512,512))

        out_path = Path(tempfile.mkdtemp()) / "out.jpg"
        generated.save(str(out_path))

        return out_path

    def run_alignment(self, image_path):
        aligned_image = align_face(filepath=image_path, predictor=self.shape_predictor)
        print("Aligned image has shape: {}".format(aligned_image.size))
        return aligned_image

    def run_on_batch(self, inputs):
        images, latents = self.net(
            inputs.to("cuda").float(), randomize_noise=False, return_latents=True
        )
        return images, latents


def concat_images(*images):
    width = 0
    for im in images:
        width += im.width
    height = max([im.height for im in images])
    concat = Image.new("RGB", (width, height))
    offset = 0
    for im in images:
        concat.paste(im, (offset, 0))
        offset += im.width
    return concat
