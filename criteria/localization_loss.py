import torch
from torch import nn
from utils.segmentation_utils import *
from models.facial_recognition.model_irse import Backbone



def _load_gan_linear_segmentation(model_path):
    """
        You can download the pretrained model from this repository
        https://github.com/AtlantixJJ/LinearGAN
        Semantic segmentation using a linear transformation on GAN features.
    """
    from linear_segmentation.semantic_extractor import EXTRACTOR_POOL
    data = torch.load(model_path)
    model_type = data['arch']['type']
    model = EXTRACTOR_POOL[model_type](**data['arch'])
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
                n_classes=n_classes, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]
            ),
            scales=[0.5, 0.75],
        )

    model = DeepLabV2_ResNet101_MSC(n_classes=182)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model = model.eval()
    return model
def get_semantic_parts(text):
    #returns the semantic parts according to the given text


    #FaceSegmentation ffhq
    [
                        ["mouth", "u_lip", "l_lip"],
                        ["skin"],
                        ["l_eye", "r_eye"],
                        ["nose"],
                        ["l_ear", "r_ear", "earrings"],
                        ["background"],
                        ["l_brow", "r_brow"],
                        ["hair", "hat"],
                        ["hair"],
                        ["cloth", "neck", "necklace"],
                        ["eyeglass"]

    ]
    #FaceSegmentation MetFaces
    [
                        ["mouth", "u_lip", "l_lip"],
                        ["skin"],
                        ["l_eye", "r_eye"],
                        ["nose"],
                        ["l_ear", "r_ear", "earrings"],
                        ["background"],
                        ["l_brow", "r_brow"],
                        ["hair", "hat"],
                        ["hair"],
                        ["cloth", "neck", "necklace"],

                    ]

    #stuff segmenation church
    [
                        ["building-other", "house"],
                        ["sky-other", "clouds"],
                        ["tree", "grass", "bush", "plant-other"],
                        ["dirt", "mud", "sand", "gravel", "ground-other", "road", "pavement"],

                    ]



    #stuff segmenation horse 
    [
    ["horse"],
    ["person"],
    ["sky-other", "clouds"],
    ["tree", "grass", "bush", "plant-other"],
    ["dirt", "mud", "sand", "gravel", "ground-other", "road", "pavement"],]
    
    
    #stuff segmenation car
    [
        ["car", "truck", "bus", "motorcycle"],
        ["road", "pavement", "dirt"],
        ["sky-other", "clouds"],
        ["tree", "grass", "bush", "plant-other"],
    ]
    pass
def combine_mask(mask1, mask2, method='average'):
    assert method in ['average', 'union', 'intersection']
    if method == 'average':
        return 0.5 * mask1 + 0.5 * mask2
    elif method == 'intersection':
        return mask1 * mask2
    else:
        return mask1 + mask2 - mask1 * mask2


class LocalizationLoss(nn.Module):
    def __init__(self, opts):
        super(LocalizationLoss, self).__init__()
        self.opts = opts
        print('Loading Segmentation Models')
        segmentation_model_string = opts.segmentation_model
        assert opts.segmentation_model_string in ["linear_segmentation","face_segmentation","stuff_segmentation"]
        if segmentation_model_string == "linear_segmentation":
            #TODO: Add model path
            segmentation_model = _load_gan_linear_segmentation("")
            self.segmenation_model = GANLinearSegmentation(segmentation_model,data_source="face")
        elif segmentation_model_string == "stuff_segmentation":
            segmentation_model = _load_cocostuff_deeplabv2("")
            self.segmenation_model = StuffSegmentation(segmentation_model)
        elif segmentation_model_string == "face_segmentation":
            segmentation_model = _load_face_bisenet_model("")
            self.segmenation_model = FaceSegmentation(segmentation_model)


        self.generator = None
        #Add generator model to compute the localization on the different layers of the network






    ### Batch data should now be coming from the generator, instead of the direct image outoput of the gan
    def forward(self, batch_data, new_batch_data, text):



        localization_loss = 0
        localization_layers = list(range(1, 14))
        localization_layer_weights = np.array([1.0] * len(localization_layers))
        loss_functions = ["L1","L2","cos"]
        loss_function = loss_functions[1]
        mode = ""

        if isinstance(self.segmentation_model, GANLinearSegmentation):
            old_segmentation_output = self.segmentation_model.predict(batch_data, one_hot=False)
        else:
            old_segmentation_output = self.segmentation_model.predict(batch_data['image'], one_hot=False)
        segmentation_output_res = old_segmentation_output.shape[2]

        if isinstance(self.segmentation_model, GANLinearSegmentation):
            new_segmentation_output = self.segmentation_model.predict(new_batch_data, one_hot=False)
        else:
                new_segmentation_output = self.segmentation_model.predict(new_batch_data['image'], one_hot=False)

        
        semantic_parts = get_semantic_parts(text)

        part_ids = [self.segmentation_model.part_to_mask_idx[part_name] for part_name in semantic_parts]

        old_mask = 0.0
        for part_idx in part_ids:
            old_mask += 1.0 * (old_segmentation_output == part_idx)
        
        new_mask = 0.0
        for part_idx in part_ids:
                new_mask += 1.0 * (new_segmentation_output == part_idx)
        
        mask_aggregation = "average"
        
        combined_mask = combine_mask(old_mask, new_mask, mask_aggregation)


        # To maximize the Localization Score in localization layers
        for layer, layer_weight in zip(reversed(localization_layers),
                                        reversed(localization_layer_weights)):
            layer_res = gan_sample_generator.layer_to_resolution[layer]
            if last_layer_res != layer_res:
                if layer_res != segmentation_output_res:
                    mask = torch.nn.functional.interpolate(mask, size=(layer_res, layer_res),
                                                            mode='bilinear',
                                                            align_corners=True)
                else:
                    mask = combined_mask.clone()
            last_layer_res = layer_res
            if layer_weight == 0:
                continue

            x1 = batch_data[f'layer_{layer}'].detach()
            x2 = new_batch_data[f'layer_{layer}']
            if loss_function == 'L1':
                diff = torch.mean(torch.abs(x1 - x2), dim=1)
            elif loss_function == 'L2':
                diff = torch.mean(torch.square(x1 - x2), dim=1)
            elif loss_function == 'cos':
                diff = 1 - torch.nn.functional.cosine_similarity(x1, x2, dim=1, eps=1e-8)
            else:
                diff = torch.mean(torch.square(x1 - x2), dim=1)
            indicator = mask[:, 0]
            if mode == 'background':
                indicator = 1 - indicator

            localization_loss -= layer_weight * torch.sum(diff * indicator, dim=[1, 2]) / (
                    torch.sum(diff, dim=[1, 2]) + 1e-6)

            # -1.0 means perfect localization and 0 means poor localization
            localization_loss = torch.mean(localization_loss)
        return localization_loss
        return loss / count, sim_improvement / count
