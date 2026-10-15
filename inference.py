import argparse
import os
import subprocess
import sys

import numpy as np
import time
import shutil

import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from DDF_dataset import DDFDataset
from pytorch_lightning import seed_everything

ROOT = os.path.dirname(os.path.abspath(__file__))
EXTRACT_SCRIPT = os.path.join(ROOT, "attribute", "injection", "extract_attribute_generic.py")


def parse_args():
    parser = argparse.ArgumentParser(
        description="FaceReclaim two-step inference: "
                    "(Step 1) generate multi-scale FAD masks when they are missing; "
                    "(Step 2) restore the original target face with the diffusion model.")
    parser.add_argument("--config_path", type=str, default="configs/FaceReclaim_s1.yaml",
                        help="path to model config")
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints/last.ckpt",
                        help="path to model checkpoint")
    parser.add_argument("--save_folder", type=str, default="recover", help="folder to save outputs")
    parser.add_argument("--source_folder", type=str, default="/FaceReclaim/source",
                        help="dataset source folder (identity source faces)")
    parser.add_argument("--target_folder", type=str, default="/FaceReclaim/target",
                        help="dataset target folder (original faces to be restored)")
    parser.add_argument("--result_folder", type=str, default="/FaceReclaim/result",
                        help="dataset result folder (face-swapped result faces)")
    parser.add_argument("--mask_root", type=str, default="/FaceReclaim/mask_FAD",
                        help="root folder holding mask_64/mask_32/mask_16 (created by Step 1 if missing)")
    parser.add_argument("--gnet_weights", type=str,
                        default=os.path.join(ROOT, "attribute", "injection", "weights", "G_net_epoch_1000.pth"),
                        help="FAD attribute-encoder weights used by Step 1")
    parser.add_argument("--face_weights", type=str,
                        default=os.path.join(ROOT, "attribute", "injection", "weights", "resnet18_110.pth"),
                        help="resnet_face18 identity weights used by Step 1")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id used by Step 1 (mask generation)")
    parser.add_argument("--no_mask_gen", action="store_true",
                        help="never auto-generate masks; fail if mask_64/32/16 are missing")
    return parser.parse_args()


def masks_ready(mask_root):
    for scale in ("64", "32", "16"):
        d = os.path.join(mask_root, "mask_" + scale)
        if not os.path.isdir(d) or len(os.listdir(d)) == 0:
            return False
    return True


def generate_masks(args):
    """Step 1/2: generate the multi-scale FAD masks from the result faces."""
    if not os.path.isfile(EXTRACT_SCRIPT):
        sys.exit("ERROR: FAD mask-extraction script not found: %s" % EXTRACT_SCRIPT)
    if not os.path.isfile(args.gnet_weights):
        sys.exit("ERROR: FAD G_net weights not found: %s (pass --gnet_weights)" % args.gnet_weights)
    if not os.path.isfile(args.face_weights):
        sys.exit("ERROR: resnet face weights not found: %s (pass --face_weights)" % args.face_weights)
    os.makedirs(args.mask_root, exist_ok=True)
    print(">>> [Step 1/2] Generating FAD masks from result faces in %s -> %s"
          % (args.result_folder, args.mask_root), flush=True)
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    result_dir_abs = os.path.abspath(args.result_folder)
    mask_dir_abs = os.path.abspath(args.mask_root)
    cmd = [sys.executable, EXTRACT_SCRIPT,
           "--result_dir", result_dir_abs,
           "--mask_dir", mask_dir_abs,
           "--gnet_weights", args.gnet_weights,
           "--face_weights", args.face_weights,
           "--gpu", str(args.gpu)]
    subprocess.run(cmd, cwd=os.path.dirname(EXTRACT_SCRIPT), env=env, check=True)
    if not masks_ready(args.mask_root):
        sys.exit("ERROR: mask generation finished but mask_64/32/16 were not produced under %s" % args.mask_root)


def main():
    args = parse_args()

    for name, p in [("source", args.source_folder), ("target", args.target_folder),
                    ("result", args.result_folder)]:
        if not os.path.isdir(p):
            sys.exit("ERROR: %s folder not found: %s" % (name, p))

    # ---------------- Step 1: masks ----------------
    if masks_ready(args.mask_root):
        print(">>> [Step 1/2] Masks already exist under %s (skip generation)" % args.mask_root, flush=True)
    else:
        if args.no_mask_gen:
            sys.exit("ERROR: masks missing under %s (run without --no_mask_gen to auto-generate them)"
                     % args.mask_root)
        generate_masks(args)

    # ---------------- Step 2: restore ----------------
    print(">>> [Step 2/2] Restoring original faces -> %s" % args.save_folder, flush=True)
    os.makedirs(args.save_folder, exist_ok=True)

    config = OmegaConf.load(args.config_path)
    model_config = config['model']
    model = instantiate_from_config(model_config)
    model.init_from_ckpt_old(args.ckpt_path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.cuda(device)
    model.eval()

    dataset = DDFDataset(source_folder=args.source_folder,
                         target_folder=args.target_folder,
                         result_folder=args.result_folder,
                         mask_64=os.path.join(args.mask_root, 'mask_64') + '/',
                         mask_32=os.path.join(args.mask_root, 'mask_32') + '/',
                         mask_16=os.path.join(args.mask_root, 'mask_16') + '/')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    total_batches = len(dataloader)
    for batch_idx, batch in tqdm(enumerate(dataloader), total=total_batches, desc='Inference'):
        source_name = batch["source_name"][0]
        recover_name = source_name.split('_')[0] + '_recover.jpg'
        output_path = os.path.join(args.save_folder, recover_name)
        if os.path.exists(output_path):
            continue
        batch["edited"] = batch["edited"].cuda(device)
        batch["edit"]["c_concat"] = batch["edit"]["c_concat"].cuda(device)
        batch["edit"]["c_crossattn"] = batch["edit"]["c_crossattn"].cuda(device)
        batch["id"]["source_id"] = batch["id"]["source_id"].cuda(device)
        batch["id"]["target_id"] = batch["id"]["target_id"].cuda(device)
        batch["mask"]["mask_64"] = batch["mask"]["mask_64"].cuda(device)
        batch["mask"]["mask_32"] = batch["mask"]["mask_32"].cuda(device)
        batch["mask"]["mask_16"] = batch["mask"]["mask_16"].cuda(device)
        with torch.no_grad():
            with model.ema_scope("Plotting"):
                source_name, images = model.log_images(batch, sample=True, ddim_steps=50, use_ddim=True)
                recover_name = source_name[0].split('_')[0] + '_recover.jpg'
                output_path = os.path.join(args.save_folder, recover_name)
                x = images['samples']
                x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
                x = 255.0 * rearrange(x, "1 c h w -> h w c")
                recover = Image.fromarray(x.type(torch.uint8).cpu().numpy())
                recover.save(output_path)

    print(len(os.listdir(args.save_folder)))


if __name__ == '__main__':
    main()