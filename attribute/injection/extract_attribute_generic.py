"""Generate FAD masks; retains legacy CLI and heatmap encoding."""
import argparse
import inspect
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import Config
from models import resnet_face18
from models.fs_networks import Generator_Adain_Upsample


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result_dir', required=True)
    p.add_argument('--mask_dir', required=True)
    p.add_argument('--gnet_weights', default='weights/G_net_epoch_1000.pth')
    p.add_argument('--face_weights', required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--attribute_eval', action='store_true',
                   help='use fixed BatchNorm statistics; changes legacy mask behavior')
    p.add_argument('--full_fad_forward', action='store_true',
                   help='compute legacy identity and full generator for mask equivalence checks')
    a = p.parse_args()
    if a.batch_size < 1 or a.num_workers < 0:
        p.error('batch_size must be positive and num_workers nonnegative')
    if a.batch_size > 1 and not a.attribute_eval:
        p.error('batch_size > 1 requires --attribute_eval to avoid batch-dependent training statistics')
    return a


def get_attmap1(features):
    # Match the original sqrt -> channel sum -> L2 normalization -> JET RGB encoding.
    outputs = features.sqrt().sum(1)
    outputs = torch.nn.functional.normalize(outputs.flatten(1), p=2, dim=1).reshape_as(outputs)
    maps = outputs.float().cpu().numpy()
    result = []
    for am in maps:
        am = 255 * (am - np.min(am)) / (np.max(am) - np.min(am) + 1e-12)
        am = np.uint8(np.floor(am))
        result.append(cv2.cvtColor(cv2.applyColorMap(am, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB))
    return result


class AttDataset(Dataset):
    def __init__(self, img_path):
        root = Path(img_path)
        self.files = sorted(root.glob('*_result.jpg'))
        known = {p.stem for p in self.files}
        self.files += sorted(p for p in root.glob('*_result.png') if p.stem not in known)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        cv2.setNumThreads(0)
        path = self.files[index]
        color = cv2.imread(str(path))
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if color is None or gray is None:
            raise RuntimeError('Cannot read mask input: ' + str(path))
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color = torch.from_numpy(color.copy()).permute(2, 0, 1).float() * 2 / 255 - 1
        gray = torch.from_numpy(cv2.resize(gray, (128, 128))).float().unsqueeze(0) * 2 / 255 - 1
        return color, gray, path.stem.removesuffix('_result') if hasattr(str, 'removesuffix') else path.stem[:-7]


def load_weights(model, path):
    # Older PyTorch forwards unknown keywords to pickle.Unpickler.
    options = {'weights_only': False} if 'weights_only' in inspect.signature(torch.load).parameters else {}
    weights = torch.load(path, map_location='cpu', **options)
    weights = weights.get('state_dict', weights)
    model.load_state_dict({k.removeprefix('module.') if hasattr(str, 'removeprefix') else
                           (k[7:] if k.startswith('module.') else k): v for k, v in weights.items()})


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('FAD mask extraction requires CUDA')
    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda', args.gpu)
    cv2.setNumThreads(0)
    for size in (64, 32, 16):
        (Path(args.mask_dir)/f'mask_{size}').mkdir(parents=True, exist_ok=True)
    net = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9)
    load_weights(net, args.gnet_weights)
    identity = resnet_face18(Config().use_se)
    load_weights(identity, args.face_weights)
    net.to(device).requires_grad_(False)
    # Default keeps original batch=1 training-mode attribute statistics for reproducibility.
    net.train(not args.attribute_eval)
    identity.to(device).eval().requires_grad_(False)
    loader = DataLoader(AttDataset(args.result_dir), batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=False, pin_memory=True)
    from PIL import Image
    with torch.no_grad():
        for color, gray, stems in tqdm(loader, desc='Extracting FAD masks'):
            color = color.to(device, non_blocking=True)
            if args.full_fad_forward:
                embedding, _ = identity(gray.to(device, non_blocking=True))
                _, features = net(color, embedding)
            else:
                features = net.extract_mask_features(color)
            for size, feature in zip((64, 32, 16), features[:3]):
                arrays = get_attmap1(feature)
                for stem, array in zip(stems, arrays):
                    Image.fromarray(array).save(Path(args.mask_dir)/f'mask_{size}'/f'{stem}_mask_{size}.png')


if __name__ == '__main__':
    main()
