import argparse
import math
import os
import shutil

import torch
import torchvision
from torch import optim
from tqdm.notebook import tqdm

from criteria.clip_loss import CLIPLoss
from criteria.id_loss import IDLoss
from criteria.localization_loss import LocalizationLoss
from models.stylegan2.model import STYLESPACE_DIMENSIONS
from models.stylegan2.model import Generator
import clip
from utils import ensure_checkpoint_exists

STYLESPACE_INDICES_WITHOUT_TORGB = [
    i
    for i in range(len(STYLESPACE_DIMENSIONS))
    if i not in list(range(1, len(STYLESPACE_DIMENSIONS), 3))
]


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def main(args):
    ensure_checkpoint_exists(args.ckpt)
    text_inputs = torch.cat([clip.tokenize(args.description)]).cuda()
    shutil.rmtree(args.results_dir, ignore_errors=True)
    os.makedirs(args.results_dir, exist_ok=True)

    g_ema = Generator(args.stylegan_size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()
    mean_latent = g_ema.mean_latent(4096)

    if args.latent_path:
        latent_code_init = torch.load(args.latent_path).cuda()
    else:
        latent_code_init_not_trunc = torch.randn(1, 512).cuda()
        with torch.no_grad():
            _, latent_code_init, _ = g_ema(
                [latent_code_init_not_trunc],
                return_latents=True,
                truncation=args.truncation,
                truncation_latent=mean_latent,
            )

    with torch.no_grad():
        result_orig, _ = g_ema(
            [latent_code_init], input_is_latent=True, randomize_noise=False
        )

    if args.work_in_stylespace:
        with torch.no_grad():
            _, _, latent_code_init = g_ema(
                [latent_code_init], input_is_latent=True, return_latents=True
            )
        latent = [s.detach().clone() for s in latent_code_init]
        for c, s in enumerate(latent):
            if c in STYLESPACE_INDICES_WITHOUT_TORGB:
                s.requires_grad = True
    else:
        latent = latent_code_init.detach().clone()
        latent.requires_grad = True

    clip_loss = CLIPLoss(args)
    id_loss = IDLoss(args)
    localization_loss = LocalizationLoss(args)

    if args.work_in_stylespace:
        optimizer = optim.Adam(latent, lr=args.lr)
    else:
        optimizer = optim.Adam([latent], lr=args.lr)

    pbar = tqdm(range(args.step))

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr

        result_gen, _ = g_ema(
            [latent],
            input_is_latent=True,
            randomize_noise=False,
            input_is_stylespace=args.work_in_stylespace,
            return_all_layers=True,
        )

        c_loss = clip_loss(result_gen["image"], text_inputs)
        loc_loss = localization_loss(result_orig, result_gen, text_inputs, i)

        if args.id_lambda > 0:
            i_loss = id_loss(result_gen["image"], result_orig["image"])[0]
        else:
            i_loss = 0

        if args.work_in_stylespace:
            l2_loss = sum(
                [
                    ((latent_code_init[c] - latent[c]) ** 2).sum()
                    for c in range(len(latent_code_init))
                ]
            )
        else:
            l2_loss = ((latent_code_init - latent) ** 2).sum()
        loss = (
            args.clip_lambda * c_loss
            + args.l2_lambda * l2_loss
            + args.id_lambda * i_loss
            + args.loc_lambda * loc_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        latent_dir = (
            [(latent_code_init[c] - latent[c]) for c in range(len(latent_code_init))]
            if (args.work_in_stylespace)
            else latent - latent_code_init
        )

        pbar.set_description(
            (f"loss: {loss.item():.4f}, loc_loss: {loc_loss.item():.1f};")
        )
        if (
            args.save_intermediate_image_every > 0
            and i % args.save_intermediate_image_every == 0
        ):
            with torch.no_grad():
                result_gen, _ = g_ema(
                    [latent],
                    input_is_latent=True,
                    randomize_noise=False,
                    input_is_stylespace=args.work_in_stylespace,
                )

            torchvision.utils.save_image(
                result_gen["image"],
                os.path.join(args.results_dir, f"{str(i).zfill(5)}.jpg"),
                normalize=True,
                range=(-1, 1),
            )
    # note: the losses correspond to the loss just before the last optimization step
    output = {
        "orig_image": result_orig["image"],
        "orig_latent": latent_code_init,
        "gen_image": result_gen["image"],
        "latent_dir": latent_dir,
        "losses": {
            "loss": loss,
            "c_loss": c_loss,
            "i_loss": i_loss,
            "l2_loss": l2_loss,
            "loc_loss": loc_loss,
        },
        "lambdas": {
            "clip_lambda": args.clip_lambda,
            "id_lambda": args.id_lambda,
            "l2_lambda": args.l2_lambda,
            "loc_lambda": args.loc_lambda,
        },
        "generator": g_ema,
    }

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--description",
        type=str,
        default="a person with purple hair",
        help="the text that guides the editing/generation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="../pretrained_models/stylegan2-ffhq-config-f.pt",
        help="pretrained StyleGAN2 weights",
    )
    parser.add_argument(
        "--stylegan_size", type=int, default=1024, help="StyleGAN resolution"
    )
    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument(
        "--step", type=int, default=300, help="number of optimization steps"
    )
    parser.add_argument(
        "--clip_lambda",
        type=float,
        default=0.008,
        help="weight of clip loss (used for editing only)",
    )
    parser.add_argument(
        "--l2_lambda",
        type=float,
        default=0.008,
        help="weight of the latent distance (used for editing only)",
    )
    parser.add_argument(
        "--id_lambda",
        type=float,
        default=0.000,
        help="weight of id loss (used for editing only)",
    )
    parser.add_argument(
        "--loc_lambda",
        type=float,
        default=0.000,
        help="weight of localization loss (used for editing only)",
    )
    parser.add_argument(
        "--latent_path",
        type=str,
        default=None,
        help="starts the optimization from the given latent code if provided. Otherwose, starts from"
        "the mean latent in a free generation, and from a random one in editing. "
        "Expects a .pt format",
    )
    parser.add_argument(
        "--truncation",
        type=float,
        default=0.7,
        help="used only for the initial latent vector, and only when a latent code path is"
        "not provided",
    )
    parser.add_argument("--work_in_stylespace", default=False, action="store_true")
    parser.add_argument(
        "--save_intermediate_image_every",
        type=int,
        default=20,
        help="if > 0 then saves intermidate results during the optimization",
    )
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument(
        "--ir_se50_weights",
        default="../pretrained_models/model_ir_se50.pth",
        type=str,
        help="Path to facial recognition network used in ID loss",
    )
    parser.add_argument(
        "--segmentation_model",
        default="face_segmentation",
        type=str,
        help="Which segmentation model to use, either linear_segmentation, face_segmentation or stuff_segmentation",
    )
    parser.add_argument(
        "--semantic_parts",
        default=["hair"],
        type=str,
        help="which semantic part to use for the localization loss",
    )
    parser.add_argument(
        "--export_segmentation_image",
        default="False",
        type=bool,
        help="Should we save the segmentation output image or not",
    )

    args = parser.parse_args()

    result_image = main(args)

    torchvision.utils.save_image(
        result_image.detach().cpu(),
        os.path.join(args.results_dir, "final_result.jpg"),
        normalize=True,
        scale_each=True,
        range=(-1, 1),
    )
