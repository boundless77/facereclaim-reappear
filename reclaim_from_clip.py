"""Batch FaceReclaim inference. See OPTIMIZED_INFERENCE.md for usage."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import os
from pathlib import Path
import random
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OLD_PREFIX = "/mnt/home/chenxiaofu/facetrack/data/deepfakes"
NEW_PREFIX = "/mnt/data/fuanlan/Dataset/DeepFake"
EXTRACT_SCRIPT = ROOT / "attribute/injection/extract_attribute_generic.py"
TEXT = "remove identity from the result face"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fake_root', help='one method directory')
    p.add_argument('--all_datasets', action='store_true', help='run the 3 datasets x 5 methods')
    p.add_argument('--input_base', default=OLD_PREFIX + '/image')
    p.add_argument('--output_base', default=NEW_PREFIX + '/image')
    p.add_argument('--save_root', help='output directory for --fake_root')
    p.add_argument('--ratio', type=float, default=1.0)
    p.add_argument('--max_source_ids', type=int, default=5, help='0 disables source sampling')
    p.add_argument('--seed', type=int, default=42, help='image selection/shuffle seed only; diffusion noise is not seeded')
    p.add_argument('--batch_size', type=int, default=4, help='actual diffusion inference batch size')
    p.add_argument('--num_workers', type=int, default=4, help='CPU image/mask loading workers')
    p.add_argument('--save_workers', type=int, default=2, help='bounded background image writers')
    p.add_argument('--mask_batch_size', type=int, default=1)
    p.add_argument('--mask_workers', type=int, default=2)
    p.add_argument('--attribute_eval', action='store_true', help='use FAD eval mode; required for mask batches > 1')
    p.add_argument('--full_fad_forward', action='store_true', help='use full legacy FAD computation for comparison')
    p.add_argument('--input_size', type=int, choices=(112, 256), default=112,
                   help='112 preserves the old resize pipeline; 256 changes preprocessing')
    p.add_argument('--precision', choices=('fp32', 'fp16', 'bf16'), default='fp32')
    p.add_argument('--no_control_cache', action='store_true', help='recompute attribute control at every DDIM step for comparison')
    p.add_argument('--gpu', type=int, default=0, help='logical GPU index after CUDA_VISIBLE_DEVICES')
    p.add_argument('--gpus', type=int, default=1, help='number of visible GPUs to use; default 1')
    p.add_argument('--num_shards', type=int, default=1, help='internal: number of whole-job workers')
    p.add_argument('--shard_index', type=int, default=0, help='internal: whole-job worker index')
    p.add_argument('--job_queue', help=argparse.SUPPRESS)
    p.add_argument('--ddim_steps', type=int, default=50)
    p.add_argument('--tmp_dir', help='temporary files parent; use local SSD')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--config_path', default=str(ROOT / 'configs/FaceReclaim_s1.yaml'))
    p.add_argument('--weights_root', default=str(ROOT), help='contains checkpoints/ and attribute/injection/weights/')
    for name in ('ckpt_path', 'vae_weights', 'arcface_weights', 'gnet_weights', 'face_weights', 'clip_path'):
        p.add_argument('--' + name)
    a = p.parse_args()
    if a.gpus < 1:
        p.error('--gpus must be positive')
    if a.gpus > 1 and not a.all_datasets:
        p.error('--gpus > 1 requires --all_datasets; a single method runs on one GPU')
    if a.gpus > 1 and (a.gpu != 0 or a.num_shards != 1 or a.shard_index != 0):
        p.error('--gpus > 1 manages GPU indices and shards; omit --gpu and shard options')
    if a.num_shards < 1 or not 0 <= a.shard_index < a.num_shards:
        p.error('require num_shards >= 1 and 0 <= shard_index < num_shards')
    if bool(a.fake_root) == a.all_datasets:
        p.error('choose exactly one of --fake_root or --all_datasets')
    if a.all_datasets and a.save_root:
        p.error('--save_root is for a single --fake_root; use --output_base for all datasets')
    for key in ('batch_size', 'save_workers', 'mask_batch_size', 'ddim_steps'):
        if getattr(a, key) < 1:
            p.error(key + ' must be positive')
    if a.num_workers < 0 or a.mask_workers < 0 or a.max_source_ids < 0:
        p.error('worker counts and max_source_ids must be nonnegative')
    if a.mask_batch_size > 1 and not a.attribute_eval:
        p.error('mask batches > 1 require --attribute_eval; legacy FAD uses training BatchNorm')
    w = Path(a.weights_root).expanduser().resolve()
    defaults = dict(ckpt_path=w/'last.ckpt', vae_weights=w/'checkpoints/vae.ckpt',
                    arcface_weights=w/'checkpoints/arcface_checkpoint.tar',
                    gnet_weights=w/'attribute/injection/weights/G_net_epoch_1000.pth',
                    face_weights=w/'attribute/injection/weights/resnet18_110.pth',
                    clip_path=w/'ldm/modules/encoders/openai/clip-vit-large-patch14')
    for key, value in defaults.items():
        setattr(a, key, str(Path(getattr(a, key) or value).expanduser().resolve()))
    return a


def output_root(fake_root, save_root):
    if save_root:
        return Path(save_root)
    value = str(fake_root)
    if value.startswith(OLD_PREFIX + '/'):
        return Path(NEW_PREFIX + value[len(OLD_PREFIX):])
    raise ValueError('Pass --save_root for an input outside the original server prefix')


def build_jobs(args):
    if not args.all_datasets:
        return [(Path(args.fake_root).expanduser().resolve(),
                 output_root(args.fake_root, args.save_root).expanduser().resolve())]
    return [((Path(args.input_base)/dataset/method).expanduser().resolve(),
             (Path(args.output_base)/dataset/method).expanduser().resolve())
            for dataset in ('celeba64_std_100c', 'vgg2_100p_256', 'cdfv2_8')
            for method in ('blendface', 'diffswap_tgt0.001', 'e4s24', 'infoswap', 'simswap')]


def selected_paths(root, args):
    # Reuse the original sampling rules, without identity-label parsing or random replacement on read errors.
    from Dataset_ID_nosource import get_all_image_files, sample_source_ids, select_images_per_pair
    paths = sample_source_ids(get_all_image_files(str(root)), args.max_source_ids, args.seed)
    paths = select_images_per_pair(paths, args.ratio)
    random.Random(args.seed).shuffle(paths)
    return [Path(p) for p in paths]


def selected_paths_with_backfill(root, args):
    """Choose the last N readable images in each pair for --ratio -N."""
    import cv2
    from Dataset_ID_nosource import get_all_image_files, sample_source_ids
    from tqdm import tqdm

    paths = sample_source_ids(get_all_image_files(str(root)), args.max_source_ids, args.seed)
    by_pair = {}
    for path in paths:
        by_pair.setdefault(Path(path).parent, []).append(Path(path))

    selected, unreadable = [], []
    wanted = abs(int(args.ratio))
    for pair_dir in tqdm(sorted(by_pair), desc='Checking input images'):
        count = 0
        for path in sorted(by_pair[pair_dir], reverse=True):
            if count >= wanted:
                break
            try:
                raw = cv2.imread(str(path))
                if raw is None:
                    raise ValueError('OpenCV could not decode the image')
            except (OSError, ValueError, cv2.error) as exc:
                unreadable.append({'path': str(path), 'error': str(exc)})
                continue
            selected.append(path)
            count += 1
    random.Random(args.seed).shuffle(selected)
    return selected, unreadable


def filter_readable_pairs(pairs):
    import cv2
    from tqdm import tqdm
    readable, unreadable = [], []
    for path, name in tqdm(pairs, desc='Checking input images'):
        try:
            raw = cv2.imread(str(path))
            if raw is None:
                raise ValueError('OpenCV could not decode the image')
        except (OSError, ValueError, cv2.error) as exc:
            unreadable.append({'path': str(path), 'error': str(exc)})
            continue
        readable.append((path, name))
    return readable, unreadable


def shard_jobs(jobs, num_shards=1, shard_index=0):
    # Partition whole dataset/method jobs, preserving each job's image order.
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError('Invalid shard configuration')
    if num_shards == 1:
        return jobs
    return jobs[shard_index::num_shards]


class FakeImages:
    def __init__(self, paths, size):
        self.paths, self.size = paths, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        import cv2
        from PIL import Image
        from torchvision import transforms as T
        cv2.setNumThreads(0)
        path = self.paths[index]
        raw = cv2.imread(str(path))
        if raw is None:
            raise RuntimeError('Cannot read image: ' + str(path))
        # Match the old torchvision Resize -> normalize -> uint8 -> PIL resize -> JPEG pipeline.
        image = Image.fromarray(raw[:, :, ::-1])
        tensor = T.Compose([T.Resize((self.size, self.size)), T.ToTensor(),
                            T.Normalize((.5, .5, .5), (.5, .5, .5))])(image)
        import numpy as np
        array = ((tensor.clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).numpy().astype(np.uint8)
        return index, np.asarray(Image.fromarray(array).resize((256, 256), Image.BICUBIC)).copy()


class RestoreImages:
    def __init__(self, work_dir, count):
        self.work_dir, self.count = Path(work_dir), count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        import numpy as np
        import torch
        from PIL import Image
        stem = f'{index + 1:05d}'
        with Image.open(self.work_dir/'result'/f'{stem}_result.jpg') as im:
            image = 2 * torch.from_numpy(np.array(im.convert('RGB')).copy()).permute(2, 0, 1).float() / 255 - 1
        masks = {}
        for size in (64, 32, 16):
            with Image.open(self.work_dir/'mask'/f'mask_{size}'/f'{stem}_mask_{size}.png') as im:
                mask = torch.from_numpy(np.array(im.convert('L')).copy()).float().unsqueeze(0) / 255
            if tuple(mask.shape) != (1, size, size):
                raise ValueError(f'Invalid mask dimensions for {stem}: {tuple(mask.shape)}')
            masks[f'mask_{size}'] = mask
        return dict(index=index, image=image, mask=masks)


def make_loader(dataset, batch, workers):
    import torch
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), drop_last=False)


def bounded_submit(pool, pending, fn, *args):
    # Backpressure keeps CPU RAM bounded and propagates background write errors.
    if len(pending) >= pool._max_workers * 2:
        pending.popleft().result()
    pending.append(pool.submit(fn, *args))


def save_image(array, path, jpeg_quality=None):
    from PIL import Image
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement prevents an interrupted save being treated as a completed output.
    handle, name = tempfile.mkstemp(prefix='.' + path.stem + '-', suffix=path.suffix, dir=str(path.parent))
    os.close(handle)
    try:
        options = {} if jpeg_quality is None else {'quality': jpeg_quality}
        Image.fromarray(array).save(name, **options)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def save_temporary_image(array, work, index):
    stem = f'{index + 1:05d}'
    result = work/'result'/f'{stem}_result.jpg'
    save_image(array, result, 95)
    for folder, suffix in (('source', 'source'), ('target_placeholder', 'target')):
        target = work/folder/f'{stem}_{suffix}.jpg'
        try:
            os.link(result, target)
        except OSError:
            shutil.copyfile(result, target)


def materialize(paths, work, args):
    from tqdm import tqdm
    (work/'result').mkdir()
    (work/'source').mkdir()
    (work/'target_placeholder').mkdir()
    pending = deque()
    with ThreadPoolExecutor(max_workers=args.save_workers) as pool:
        for indices, arrays in tqdm(make_loader(FakeImages(paths, args.input_size), args.batch_size,
                                                args.num_workers), desc='Preparing fake images'):
            for index, array in zip(indices.tolist(), arrays.numpy()):
                bounded_submit(pool, pending, save_temporary_image, array, work, index)
        for future in pending:
            future.result()


def generate_masks(work, args):
    command = [sys.executable, str(EXTRACT_SCRIPT), '--result_dir', str(work/'result'),
               '--mask_dir', str(work/'mask'), '--gnet_weights', args.gnet_weights,
               '--face_weights', args.face_weights, '--gpu', str(args.gpu),
               '--batch_size', str(args.mask_batch_size), '--num_workers', str(args.mask_workers)]
    if args.attribute_eval:
        command.append('--attribute_eval')
    if args.full_fad_forward:
        command.append('--full_fad_forward')
    subprocess.run(command, cwd=str(EXTRACT_SCRIPT.parent), check=True)


def load_model(args, device):
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    config = OmegaConf.load(args.config_path)
    params = config.model.params
    params.first_stage_config.params.ckpt_path = args.vae_weights
    params.id_stage_config.setdefault('params', {})['checkpoint_path'] = args.arcface_weights
    params.text_stage_config.setdefault('params', {})['version'] = args.clip_path
    model = instantiate_from_config(config.model)
    model.init_from_ckpt_old(args.ckpt_path)
    return model.to(device).eval().requires_grad_(False)


def restore(model, paths, names, work, output, args, device, encoded_text):
    import torch
    from tqdm import tqdm
    from ldm.models.diffusion.ddim import DDIMSampler
    dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16}.get(args.precision)
    amp = (lambda: torch.autocast(device_type='cuda', dtype=dtype)) if dtype else nullcontext
    sampler = DDIMSampler(model)
    pending = deque()
    with ThreadPoolExecutor(max_workers=args.save_workers) as pool, torch.no_grad(), model.ema_scope('Inference'):
        for batch in tqdm(make_loader(RestoreImages(work, len(paths)), args.batch_size,
                                     args.num_workers), desc='Restoring batches'):
            image = batch['image'].to(device, non_blocking=True)
            masks = {key: value.to(device, non_blocking=True) for key, value in batch['mask'].items()}
            count = image.shape[0]
            with amp():
                latent = model.encode_first_stage(image).mode().detach()
                identity = model.id_stage_model(image)
                condition = model.cond_stage_model(identity, encoded_text.expand(count, -1, -1), latent)
                attributes = masks
                if not args.no_control_cache:
                    attributes = {'_cached_control': tuple(model.attribute_stage_model(
                        x=latent, attributes=list(masks.values())))}
                samples, _ = sampler.sample(args.ddim_steps, count,
                    (model.channels, model.image_size, model.image_size),
                    conditioning={'c_concat': [latent], 'c_crossattn': [condition]},
                    attributes=attributes, eta=0., verbose=False)
                images = model.decode_first_stage(samples)
            arrays = ((images.float().clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()
            if len(arrays) != count:
                raise RuntimeError('Model returned an incomplete batch')
            for index, array in zip(batch['index'].tolist(), arrays):
                bounded_submit(pool, pending, save_image, array, output/names[index])
            del image, masks, latent, identity, condition, attributes, samples, images
        for future in pending:
            future.result()


def queued_jobs(jobs, queue_dir, gpu):
    """Atomically claim whole jobs; a claimed job belongs to exactly one GPU."""
    for index, job in enumerate(jobs):
        try:
            (Path(queue_dir)/str(index)).mkdir()
        except FileExistsError:
            continue
        print(f'[FaceReclaim GPU {gpu}] claimed task {index + 1}/{len(jobs)}: {job[0]}', flush=True)
        yield job


def worker_command(arguments, count, rank, queue_dir=None):
    # Final argparse values override parent options and prevent recursion.
    command = [sys.executable, str(ROOT/'reclaim_from_clip.py'), *arguments,
               '--gpus', '1', '--gpu', str(rank), '--num_shards', '1',
               '--shard_index', '0']
    if queue_dir is not None:
        command += ['--job_queue', str(queue_dir)]
    return command


class TerminalProgress:
    """Parent owns all cursor movement; workers only supply progress text."""
    def __init__(self, count, stream=None):
        self.stream = stream or sys.stdout
        self.rows = [f'[GPU {rank}] waiting...' for rank in range(count)]
        self.lock = threading.Lock()
        self.stream.write('\n' * count)
        self.stream.flush()

    def write(self, gpu, line, progress=False):
        with self.lock:
            # Cursor rests below the reserved rows. Clear them before messages.
            self.stream.write(f'\x1b[{len(self.rows)}A')
            for _ in self.rows:
                self.stream.write('\r\x1b[2K\n')
            self.stream.write(f'\x1b[{len(self.rows)}A')
            if progress:
                self.rows[gpu] = f'[GPU {gpu}] {line}'
            else:
                self.stream.write(line + '\n')
            # Limit to terminal width so a long row cannot wrap into another.
            width = max(1, shutil.get_terminal_size((120, 24)).columns - 1)
            for row in self.rows:
                self.stream.write('\r\x1b[2K' + row[:width] + '\n')
            self.stream.flush()


def display_worker_output(stream, gpu, lock, terminal=None):
    """Convert each worker's carriage-return bars to labelled terminal lines."""
    last_progress = {}
    with stream:
        for raw in stream:
            line = raw.strip()
            if not line:
                continue
            progress = '%' in line and '|' in line
            if terminal is not None:
                if progress:
                    # A single fixed row per GPU shows its current progress bar.
                    terminal.write(gpu, line, progress=True)
                else:
                    if not line.startswith(f'[FaceReclaim GPU {gpu}]'):
                        line = f'[GPU {gpu}] {line}'
                    terminal.write(gpu, line)
                continue
            # Non-interactive terminals retain readable, throttled plain lines.
            if line.startswith(('DDIM Sampler:', 'Data shape for DDIM sampling is',
                                'Running DDIM Sampling with')):
                continue
            if '%' in line and '|' in line:
                stage = line.split(':', 1)[0]
                now = time.monotonic()
                if now - last_progress.get(stage, float('-inf')) < 10:
                    continue
                last_progress[stage] = now
            if not line.startswith(f'[FaceReclaim GPU {gpu}]'):
                line = f'[GPU {gpu}] {line}'
            with lock:
                print(line, flush=True)


def launch_workers(count):
    import torch
    visible = torch.cuda.device_count()
    if visible < count:
        raise RuntimeError(f'Requested {count} GPUs but only {visible} are visible in this job')
    processes = []
    previous_handlers = {}
    queue_dir = tempfile.mkdtemp(prefix='facereclaim_queue_')
    display_pool = ThreadPoolExecutor(max_workers=count)
    display_futures = []
    display_lock = threading.Lock()
    terminal = TerminalProgress(count) if sys.stdout.isatty() else None

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, interrupted)
        for rank in range(count):
            env = dict(os.environ, PYTHONUNBUFFERED='1')
            processes.append(subprocess.Popen(
                worker_command(sys.argv[1:], count, rank, queue_dir),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors='replace', bufsize=1, env=env,
                start_new_session=(os.name == 'posix')))
            display_futures.append(display_pool.submit(
                display_worker_output, processes[-1].stdout, rank, display_lock, terminal))
        while True:
            statuses = [process.poll() for process in processes]
            failure = next((code for code in statuses if code not in (None, 0)), None)
            if failure is not None:
                raise RuntimeError(f'A GPU worker failed with exit code {failure}')
            if all(code == 0 for code in statuses):
                break
            time.sleep(.2)
    finally:
        for process in processes:
            if os.name == 'posix':
                # Include mask extraction grandchildren in cancellation.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            elif process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == 'posix':
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
                process.wait()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        display_pool.shutdown(wait=True)
        shutil.rmtree(queue_dir)
        for future in display_futures:
            future.result()


def main():
    args = parse_args()
    if getattr(args, 'gpus', 1) > 1:
        launch_workers(args.gpus)
        return
    jobs = build_jobs(args)
    jobs = shard_jobs(jobs, getattr(args, 'num_shards', 1), getattr(args, 'shard_index', 0))
    # Fail before GPU initialization if any requested dataset/weight is missing.
    for source, target in jobs:
        if not source.is_dir():
            raise FileNotFoundError(source)
        if source.resolve() == target.resolve() or source.resolve() in target.resolve().parents:
            raise ValueError('Output must be outside the input directory: ' + str(target))
    if getattr(args, 'job_queue', None):
        jobs = queued_jobs(jobs, args.job_queue, args.gpu)
    for key in ('config_path', 'ckpt_path', 'vae_weights', 'arcface_weights', 'gnet_weights', 'face_weights', 'clip_path'):
        if not Path(getattr(args, key)).exists():
            raise FileNotFoundError(f'{key}: {getattr(args, key)}')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('This pipeline requires a CUDA GPU')
    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda', args.gpu)
    if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('This GPU does not support bf16')
    if args.tmp_dir:
        Path(args.tmp_dir).mkdir(parents=True, exist_ok=True)
    model, encoded_text = None, None
    for source, output in jobs:
        started = time.perf_counter()
        backfill = args.ratio < -1
        if backfill:
            paths, _ = selected_paths_with_backfill(source, args)
        else:
            paths = selected_paths(source, args)
        chosen = len(paths)
        pairs = [(p, p.relative_to(source.resolve())) for p in paths]
        pairs = [(p, name) for p, name in pairs if args.overwrite or not (output/name).exists()]
        skipped_existing = chosen - len(pairs)
        if pairs and not backfill:
            pairs, _ = filter_readable_pairs(pairs)
        print(f'[FaceReclaim GPU {args.gpu}] {source}: selected={chosen}, pending={len(pairs)}, '
              f'skipped_existing={skipped_existing}', flush=True)
        if not pairs:
            continue
        output.mkdir(parents=True, exist_ok=True)
        paths, names = map(list, zip(*pairs))
        with tempfile.TemporaryDirectory(prefix='facereclaim_flip_', dir=args.tmp_dir) as temp:
            work = Path(temp)
            print(f'[FaceReclaim] Temporary work_dir: {work}', flush=True)
            materialize(paths, work, args)
            prepared = time.perf_counter()
            generate_masks(work, args)
            masked = time.perf_counter()
            if model is None:
                model = load_model(args, device)
                with torch.no_grad():
                    encoded_text = model.text_encoder_model.encode([TEXT]).detach()
            loaded = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device)
            restore(model, paths, names, work, output, args, device, encoded_text)
            torch.cuda.synchronize(device)
            finished = time.perf_counter()
            timing = dict(prepare_seconds=prepared-started, mask_seconds=masked-prepared,
                          load_seconds=loaded-masked, restore_seconds=finished-loaded,
                          total_seconds=finished-started, restored=len(paths),
                          images_per_second=len(paths)/(finished-loaded),
                          peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3)
            print(f'[FaceReclaim GPU {args.gpu}] {source} timing: {timing}', flush=True)


if __name__ == '__main__':
    main()
