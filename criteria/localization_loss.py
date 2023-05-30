import torch
from torch import nn
from criteria.utils.segmentation_utils import *
from models.facial_recognition.model_irse import Backbone
import torchvision
from PIL import Image


def _load_gan_linear_segmentation(model_path):
    """
    You can download the pretrained model from this repository
    https://github.com/AtlantixJJ/LinearGAN
    Semantic segmentation using a linear transformation on GAN features.
    """
    from linear_segmentation.semantic_extractor import EXTRACTOR_POOL

    data = torch.load(model_path)
    model_type = data["arch"]["type"]
    model = EXTRACTOR_POOL[model_type](**data["arch"])
    model.load_state_dict(data["param"])
    model = model.eval()
    return model


def _load_face_bisenet_model(model_path):
    """
    You can download the pretrained model from this repository
    https://github.com/zllrunning/face-parsing.PyTorch
    """
    from models.face_bisenet.model import BiSeNet

    model = BiSeNet(n_classes=19)
    model.load_state_dict(torch.load(model_path))
    model = model.eval()
    return model


def _load_cocostuff_deeplabv2(model_path):
    from models.deeplab.deeplabv2 import DeepLabV2
    from models.deeplab.msc import MSC

    """
    You can download the pretrained model from this repository
    https://github.com/kazuto1011/deeplab-pytorch
    """

    def DeepLabV2_ResNet101_MSC(n_classes):
        return MSC(
            base=DeepLabV2(
                n_classes=n_classes,
                n_blocks=[3, 4, 23, 3],
                atrous_rates=[6, 12, 18, 24],
            ),
            scales=[0.5, 0.75],
        )

    model = DeepLabV2_ResNet101_MSC(n_classes=182)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model = model.eval()
    return model


def combine_mask(mask1, mask2, method="left_only"):
    assert method in ["average", "union", "intersection", "left_only"]
    if method == "average":
        return 0.5 * mask1 + 0.5 * mask2
    elif method == "intersection":
        return mask1 * mask2
    elif method == "union":
        return mask1 + mask2 - mask1 * mask2
    else:
        return mask1


def raw_image_to_pil_image(raw_img):
    img = (raw_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    pil_imgs = []
    for i in range(len(img)):
        pil_imgs.append(Image.fromarray(img[i].cpu().numpy(), "RGB"))
    return pil_imgs


class LocalizationLoss(nn.Module):
    def __init__(self, opts):
        super(LocalizationLoss, self).__init__()
        self.opts = opts
        print("Loading Segmentation Models")
        segmentation_model_string = opts.segmentation_model
        self.device = torch.device("cuda")
        assert segmentation_model_string in [
            "face_segmentation",
            "stuff_segmentation",
        ]
        if segmentation_model_string == "stuff_segmentation":
            segmentation_model = _load_cocostuff_deeplabv2(
                "pretrained/cocostuff_deeplab/deeplabv2_resnet101_msc-cocostuff164k-100000.pth"
            )
            self.segmentation_model = StuffSegmentation(segmentation_model)
        elif segmentation_model_string == "face_segmentation":
            segmentation_model = _load_face_bisenet_model(
                "pretrained/face_bisenet/model.pth"
            )
            self.segmentation_model = FaceSegmentation(segmentation_model, self.device)
        self.semantic_parts = opts.semantic_parts

    def get_semantic_parts(self, text):
        # returns the semantic parts according to the given text
        parts = {
            "mouth": ["mouth", "u_lip", "l_lip"],
            "skin": ["skin"],
            "eyes": ["l_eye", "r_eye"],
            "nose": ["nose"],
            "ears": ["l_ear", "r_ear", "earrings"],
            "eye_brows": ["l_brow", "r_brow"],
            "hat": ["hair", "hat"],
            "hair": ["hair"],
            "neck": ["cloth", "neck", "necklace"],
        }
        semantic_parts = []
        for semantic_part in self.semantic_parts:
            if semantic_part in parts.keys():
                semantic_parts += parts[semantic_part]

        return semantic_parts

    ### Batch data should now be coming from the generator, instead of the direct image outoput of the gan
    def forward(self, batch_data, new_batch_data, text, i):
        # print(batch_data.keys())

        last_layer_res = None
        localization_loss = 0
        localization_layers = list(range(1, 10))
        localization_layer_weights = np.array(
            [0.0] * (len(localization_layers) - 1) + [1.0]
        )
        layer_to_resolution = {
            0: 4,
            1: 4,
            2: 8,
            3: 16,
            4: 32,
            5: 64,
            6: 128,
            7: 256,
            8: 512,
            9: 1024,
        }
        loss_functions = ["L1", "L2", "cos"]
        loss_function = loss_functions[1]
        mode = "background"

        if isinstance(self.segmentation_model, GANLinearSegmentation):
            old_segmentation_output = self.segmentation_model.predict(
                batch_data, one_hot=False
            )
        else:
            old_segmentation_output = self.segmentation_model.predict(
                raw_image_to_pil_image(batch_data["image"]), one_hot=False
            )
        segmentation_output_res = old_segmentation_output.shape[2]

        if isinstance(self.segmentation_model, GANLinearSegmentation):
            new_segmentation_output = self.segmentation_model.predict(
                new_batch_data, one_hot=False
            )
        else:
            new_segmentation_output = self.segmentation_model.predict(
                raw_image_to_pil_image(new_batch_data["image"]), one_hot=False
            )

        semantic_parts = self.get_semantic_parts(text)

        part_ids = [
            self.segmentation_model.part_to_mask_idx[part_name]
            for part_name in semantic_parts
        ]

        old_mask = 0.0
        for part_idx in part_ids:
            old_mask += 1.0 * (old_segmentation_output == part_idx)

        new_mask = 0.0
        for part_idx in part_ids:
            new_mask += 1.0 * (new_segmentation_output == part_idx)

        mask_aggregation = "left_only"

        combined_mask = combine_mask(old_mask, new_mask, mask_aggregation)

        mask = combined_mask.clone()

        # To maximize the Localization Score in localization layers
        for layer, layer_weight in zip(
            reversed(localization_layers), reversed(localization_layer_weights)
        ):
            layer_res = layer_to_resolution[layer]
            if last_layer_res != layer_res:
                if layer_res != segmentation_output_res:
                    mask = torch.nn.functional.interpolate(
                        mask,
                        size=(layer_res, layer_res),
                        mode="bilinear",
                        align_corners=True,
                    )
                else:
                    mask = combined_mask.clone()
            last_layer_res = layer_res
            if layer_weight == 0:
                continue

            x1 = batch_data[f"layer_{layer}"].detach()
            x2 = new_batch_data[f"layer_{layer}"]
            if loss_function == "L1":
                diff = torch.mean(torch.abs(x1 - x2), dim=1)
            elif loss_function == "L2":
                diff = torch.mean(torch.square(x1 - x2), dim=1)
            elif loss_function == "cos":
                diff = 1 - torch.nn.functional.cosine_similarity(
                    x1, x2, dim=1, eps=1e-8
                )
            else:
                diff = torch.mean(torch.square(x1 - x2), dim=1)
            indicator = mask[:, 0]
            if mode == "background":
                indicator = 1 - indicator

            # print('Diff shape: {}'.format(diff.shape))
            # print('Indicator shape: {}'.format(indicator.shape))
            # print('1st term: {}'.format(torch.sum(diff * indicator, dim=[1, 2])))
            # print('2nd term: {}'.format(torch.sum(diff, dim=[1, 2]) + 1e-6))

            localization_loss = (
                layer_weight * torch.sum(diff * indicator, dim=[1, 2])[0]
            )

            # 0.0 means perfect localization and 1.0 means poor localization
            localization_loss = torch.mean(localization_loss)

            # print("Localization loss: {}.".format(localization_loss))

        pil_to_tensor = torchvision.transforms.ToTensor()

        # print('Image shape: {}.'.format(batch_data["image"].shape))

        old_image = pil_to_tensor(raw_image_to_pil_image(batch_data["image"])[0])
        old_mask_image = (
            torch.nn.functional.interpolate(
                old_mask, size=(1024, 1024), mode="bilinear", align_corners=True
            )[0]
            .repeat(3, 1, 1)
            .detach()
            .cpu()
        )
        new_image = pil_to_tensor(raw_image_to_pil_image(new_batch_data["image"])[0])
        new_mask_image = (
            torch.nn.functional.interpolate(
                new_mask, size=(1024, 1024), mode="bilinear", align_corners=True
            )[0]
            .repeat(3, 1, 1)
            .detach()
            .cpu()
        )
        if self.opts.export_segmentation_image:
            images = [old_image, old_mask_image, new_image, new_mask_image]
            image_grid = torchvision.utils.make_grid(images, nrow=2)
            torchvision.utils.save_image(
                image_grid,
                os.path.join(
                    self.opts.results_dir,
                    f"seg_{str(i).zfill(5)}_loc_loss={localization_loss.item():.4f}.jpg",
                ),
            )

        return localization_loss
