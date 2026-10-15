import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf


ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from DDF_dataset import DDFDataset
from ldm.util import instantiate_from_config
from Dataset_ID_nosource import faceDataset_ID


OLD_PREFIX = "/mnt/home/chenxiaofu/facetrack/data/deepfakes"
NEW_PREFIX = "/mnt/data/zhonghuanlong/Dataset/DeepFake"
EXTRACT_SCRIPT = Path(
    "/mnt/home/zhonghuanlong/Research/Face/facereclaim/attribute/injection/extract_attribute_generic.py"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Restore target faces from fake images read by the FLIP dataloader."
    )
    parser.add_argument("--fake_root", required=True,
                        help="FLIP dataset root containing the fake images")
    parser.add_argument("--save_root", default=None,
                        help="output root; defaults to fake_root with the old prefix replaced")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="dataset ratio passed to faceDataset_ID")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--config_path", default=str(ROOT / "configs" / "FaceReclaim_s1.yaml"))
    parser.add_argument("--ckpt_path", default=str(ROOT / "checkpoints" / "last.ckpt"))
    parser.add_argument("--gnet_weights", default=str(
        ROOT / "attribute" / "injection" / "weights" / "G_net_epoch_1000.pth"))
    parser.add_argument("--face_weights", default=str(
        ROOT / "attribute" / "injection" / "weights" / "resnet18_110.pth"))
    parser.add_argument("--ddim_steps", type=int, default=50)
    return parser.parse_args()


def output_root(fake_root, save_root):
    if save_root:
        return Path(save_root)
    fake_root = str(Path(fake_root))
    if fake_root.startswith(OLD_PREFIX):
        return Path(NEW_PREFIX + fake_root[len(OLD_PREFIX):])
    raise ValueError(
        f"fake_root does not start with {OLD_PREFIX}; pass --save_root explicitly"
    )


def tensor_to_image(tensor):
    image = ((tensor.detach().cpu().float().clamp(-1, 1) + 1) * 127.5)
    image = image.permute(1, 2, 0).numpy().astype(np.uint8)
    return Image.fromarray(image).convert("RGB").resize((256, 256), Image.BICUBIC)


def dataset_file_names(dataset, dataset_root):
    # import pdb; pdb.set_trace()
    dataset_root = Path(dataset_root).resolve()
    names = []
    for value in dataset.image_list:
        value = value[0] if isinstance(value, (tuple, list)) else value
        value_path = Path(str(value))
        try:
            relative_path = value_path.resolve().relative_to(dataset_root)
        except ValueError:
            relative_path = Path(value_path.name)
        names.append(relative_path)
    if names:
        return names
    return []


def materialize_flip_dataset(loader, original_names, work_dir):
    result_dir = work_dir / "result"
    source_dir = work_dir / "source"
    target_placeholder_dir = work_dir / "target_placeholder"
    result_dir.mkdir()
    source_dir.mkdir()
    target_placeholder_dir.mkdir()

    index = 0
    saved_names = []
    for batch in tqdm(loader, desc="Reading FLIP fake images"):
        images = batch["image"]
        for image in images:
            stem = f"{index + 1:05d}"
            original_path = (
                original_names[index]
                if index < len(original_names)
                else Path(f"{stem}.jpg")
            )
            fake = tensor_to_image(image)
            fake.save(result_dir / f"{stem}_result.jpg", quality=95)
            fake.save(source_dir / f"{stem}_source.jpg", quality=95)
            # DDFDataset needs an edited image for shape/latent bookkeeping.
            # This is only a placeholder; the real target is never read.
            fake.save(target_placeholder_dir / f"{stem}_target.jpg", quality=95)
            saved_names.append(original_path)
            index += 1

    return source_dir, target_placeholder_dir, result_dir, saved_names


def generate_masks(result_dir, mask_dir, args):
    mask_dir.mkdir()
    subprocess.run(
        [
            sys.executable,
            str(EXTRACT_SCRIPT),
            "--result_dir", str(result_dir),
            "--mask_dir", str(mask_dir),
            "--gnet_weights", args.gnet_weights,
            "--face_weights", args.face_weights,
            "--gpu", str(torch.cuda.current_device() if torch.cuda.is_available() else 0),
        ],
        cwd=str(EXTRACT_SCRIPT.parent),
        check=True,
    )


def restore(work_dir, output_dir, output_names, args):
    config = OmegaConf.load(args.config_path)
    model = instantiate_from_config(config["model"])
    model.init_from_ckpt_old(args.ckpt_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    dataset = DDFDataset(
        source_folder=str(work_dir / "source"),
        target_folder=str(work_dir / "target_placeholder"),
        result_folder=str(work_dir / "result"),
        mask_64=str(work_dir / "mask" / "mask_64") + "/",
        mask_32=str(work_dir / "mask" / "mask_32") + "/",
        mask_16=str(work_dir / "mask" / "mask_16") + "/",
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for index, batch in enumerate(
            tqdm(loader, total=len(output_names), desc="Restoring target images")
        ):
            output_path = output_dir / output_names[index]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists():
                continue

            batch["edited"] = batch["edited"].to(device)
            for key in ("c_concat", "c_crossattn"):
                batch["edit"][key] = batch["edit"][key].to(device)
            for key in ("source_id", "target_id"):
                batch["id"][key] = batch["id"][key].to(device)
            for key in ("mask_64", "mask_32", "mask_16"):
                batch["mask"][key] = batch["mask"][key].to(device)

            with model.ema_scope("Plotting"):
                _, images = model.log_images(
                    batch, sample=True, ddim_steps=args.ddim_steps, use_ddim=True
                )

            image = images["samples"][0].clamp(-1, 1)
            image = ((image + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
            Image.fromarray(image).save(output_path)


def main():
    args = parse_args()
    fake_root = Path(args.fake_root)
    if not fake_root.is_dir():
        raise SystemExit(f"Fake dataset not found: {fake_root}")

    save_root = output_root(fake_root, args.save_root)
    dataset = faceDataset_ID(
        root_path=str(fake_root),
        ratio=args.ratio,
        form_gallery=False,
        return_parsing=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    original_names = dataset_file_names(dataset, fake_root)

    with tempfile.TemporaryDirectory(prefix="facereclaim_flip_") as temp:
        work_dir = Path(temp)
        print(f"[FaceReclaim] Temporary work_dir: {work_dir}", flush=True)
        print("[FaceReclaim] Reading fake images...", flush=True)
        _, _, result_dir, output_names = materialize_flip_dataset(
            loader, original_names, work_dir
        )
        print("[FaceReclaim] Generating FAD masks...", flush=True)
        generate_masks(result_dir, work_dir / "mask", args)
        print("[FaceReclaim] Restoring target images...", flush=True)
        restore(work_dir, save_root, output_names, args)

    print("[FaceReclaim] Temporary work_dir has been removed.", flush=True)
    print(f"Saved {len(output_names)} restored images to {save_root}")


if __name__ == "__main__":
    main()
