# ms1mv3_local_rec_512.py
import os
from PIL import Image
import io
import numpy as np
from torchvision import transforms as T
from torch.utils.data import TensorDataset
import cv2
from omegaconf import ListConfig
from tqdm import tqdm
import random
random.seed(42)
import re


def get_all_image_files(directory, extensions=None):
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    
    image_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_files.append(os.path.abspath(os.path.join(root, file)))
    return image_files


def sample_source_ids(image_files, max_source_ids, seed=42):
    """Sample a fixed number of source identities for each target directory."""
    if max_source_ids is None or max_source_ids <= 0:
        return image_files

    sampler = random.Random(seed)
    source_dirs = {}
    source_pattern = re.compile(r"^(\d+)_(\d+)$")
    for image_path in image_files:
        source_dir = os.path.dirname(image_path)
        match = source_pattern.fullmatch(os.path.basename(source_dir))
        if match is None:
            continue
        target_id, source_id = match.groups()
        target_key = (os.path.dirname(source_dir), target_id)
        source_dirs.setdefault(target_key, {})[int(source_id)] = source_dir

    if not source_dirs:
        return image_files

    selected_dirs = set()
    for target_key in sorted(source_dirs):
        dirs_by_source = source_dirs[target_key]
        source_ids = sorted(dirs_by_source)
        selected_sources = sampler.sample(
            source_ids,
            k=min(max_source_ids, len(source_ids)),
        )
        selected_dirs.update(dirs_by_source[source_id] for source_id in selected_sources)

    all_source_dirs = {
        source_dir
        for dirs_by_source in source_dirs.values()
        for source_dir in dirs_by_source.values()
    }
    filtered = [
        image_path
        for image_path in image_files
        if (
            os.path.dirname(image_path) not in all_source_dirs
            or os.path.dirname(image_path) in selected_dirs
        )
    ]
    return filtered


def select_images_per_pair(image_files, ratio):
    """Keep a fixed prefix of images inside each target-source pair directory."""
    if ratio == 0 or ratio == 1:
        return image_files

    images_by_pair = {}
    for image_path in image_files:
        pair_dir = os.path.dirname(image_path)
        images_by_pair.setdefault(pair_dir, []).append(image_path)

    sampled_images = []
    for pair_dir in sorted(images_by_pair):
        pair_images = sorted(images_by_pair[pair_dir])
        image_count = len(pair_images)
        if ratio > 0:
            keep_count = int(image_count * ratio) if ratio <= 1 else int(ratio)
            keep_count = min(max(keep_count, 0), image_count)
            selected = pair_images[:keep_count]
        elif ratio >= -1:
            selected = pair_images[int(image_count * ratio):]
        else:
            selected = pair_images[int(ratio):]
        sampled_images.extend(selected)

    return sampled_images


class faceDataset_ID(TensorDataset):
    def __init__(
        self,
        root_path,
        ratio,
        form_gallery,
        logger=None,
        return_parsing=False,
        max_source_ids=5,
        source_sampling_seed=42,
    ):
        self.logger=logger
        self.root_path = root_path
        self.ratio=ratio
        self.idbook = {}
        self.real_idbook = {}
        self.fake_idbook = {}
        self.scenebook={}
        self.form_gallery = form_gallery
        self.return_parsing = return_parsing
        self.mean=(0.5, 0.5, 0.5)
        self.std=(0.5, 0.5, 0.5)

        self.id_transform = T.Compose([
            T.Resize((112, 112)),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)  
        ])
        
        if self.return_parsing:
            self.parsing_transform = T.Compose([
                T.Resize((512, 512), interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.image_list = []
        if isinstance(root_path, ListConfig):
            root_path_list = list(root_path)
            for path in root_path_list:
                image_paths = get_all_image_files(directory=path)
                self.image_list.extend(image_paths)
        else:
            image_paths = get_all_image_files(directory=root_path)
            self.image_list.extend(image_paths)
        image_count_before_source_limit = len(self.image_list)
        self.image_list = sample_source_ids(
            self.image_list,
            max_source_ids=max_source_ids,
            seed=source_sampling_seed,
        )
        image_count_before_ratio = len(self.image_list)
        self.image_list = select_images_per_pair(
            self.image_list,
            ratio=ratio,
        )
        if len(self.image_list) != image_count_before_source_limit:
            print(
                "Source limit| images: "
                f"{image_count_before_source_limit} -> {len(self.image_list)} "
                f"(sampled {max_source_ids} source IDs per target, "
                f"seed={source_sampling_seed})"
            )
        if len(self.image_list) != image_count_before_ratio:
            print(
                "Pair ratio| images: "
                f"{image_count_before_ratio} -> {len(self.image_list)} "
                f"(ratio={ratio}, keep prefix per pair)"
            )
        random.shuffle(self.image_list)
        for index, path in enumerate(tqdm(self.image_list, desc="classifying image paths")):
            if 'cdfv2' in self.image_list[index] or 'id' in self.image_list[index]:
                ori_id = int(self.image_list[index].split('/')[-2][2:]) if 'real' in self.image_list[index] else int(self.image_list[index].split('/')[-2].split('_')[0][2:])
                if self.form_gallery and 'real' in self.image_list[index]:
                    scene_id = int(self.image_list[index].split('/')[-1].split('_')[0])  
                    if scene_id not in self.scenebook:
                        self.scenebook[scene_id] = [self.image_list[index]]
                    else:
                        self.scenebook[scene_id].append(self.image_list[index])
            elif 'vgg2' in self.image_list[index]:
                ori_id=int(self.image_list[index].split('/')[-2].lstrip('n')) if 'real' in self.image_list[index] else int(self.image_list[index].split('/')[-2].split('_')[0].lstrip('n'))
            else:
                # import pdb;pdb.set_trace()
                ori_id=int(self.image_list[index].split('/')[-2]) if 'real' in self.image_list[index] else int(self.image_list[index].split('/')[-2].split('_')[0])
                # 1. 处理总表
                
            img_path = self.image_list[index]
            self.idbook.setdefault(ori_id, []).append(img_path)

            # 2. 分类处理 real/fake
            if 'real' in img_path:
                self.real_idbook.setdefault(ori_id, []).append(img_path)
            else:
                self.fake_idbook.setdefault(ori_id, []).append(img_path)
        print(f"Before ratio|Total IDs|images: {len(self.idbook)}:{len(self.image_list)} ")
        print(f"Before ratio|Total real IDs: {len(self.real_idbook)}")
        print(f"Before ratio|Total fake IDs: {len(self.fake_idbook)}")
        # if self.form_gallery and 'cdfv2' in self.image_list[0]:
        #     self.image_list = self.scenebook[0]
        # else:
        self.real_image_list = [item for sublist in self.real_idbook.values() for item in sublist]
        self.fake_image_list = [item for sublist in self.fake_idbook.values() for item in sublist]
        self.image_list = self.real_image_list + self.fake_image_list
        # for key in self.idbook:
        #     print(f"After ratio| ID {key} has {len(self.idbook[key])} images.")
        print(f"After ratio| Total IDs: {len(self.idbook)}")
        print(f"After ratio| Total scene: {len(self.scenebook)} ")
        print(f"After ratio| Total real images: {len(self.real_image_list)}")
        print(f"After ratio| Total fake images: {len(self.fake_image_list)}")
        print(f"After ratio| Total images: {len(self.image_list)}")

    def __getitem__(self, item):
        while True:
            try:
                fake_path=self.image_list[item]
                fake_img_pil = Image.fromarray(cv2.imread(fake_path)[:, :, ::-1]) # BGR to RGB
                fake_img = self.id_transform(fake_img_pil)
                
                if self.return_parsing:
                    parsing_img = self.parsing_transform(fake_img_pil)
                if 'celeba' in fake_path:
                    ori_id=int(self.image_list[item].split('/')[-2]) if 'real' in self.image_list[item] else int(self.image_list[item].split('/')[-2].split('_')[0])
                   
                    ret={
                        'image': fake_img,
                        'sec_label': ori_id
                    }
                   
                elif 'vgg2' in fake_path:
                    ori_id=int(self.image_list[item].split('/')[-2].lstrip('n')) if 'real' in self.image_list[item] else int(self.image_list[item].split('/')[-2].split('_')[0].lstrip('n'))
                    ret={
                        'image': fake_img,
                        'sec_label': ori_id
                    }
                elif 'cdfv2' in fake_path or 'id' in fake_path:
                    ori_id = int(self.image_list[item].split('/')[-2][2:]) if 'real' in self.image_list[item] else int(self.image_list[item].split('/')[-2].split('_')[0][2:])
                    ret={
                        'image': fake_img,
                        'sec_label': ori_id
                    }
                if self.return_parsing:
                    ret['parsing_image'] = parsing_img
                return ret
            except Exception as e:
                item = random.randint(0, len(self.image_list) - 1)


    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    root = '/your/path/to/ms1mv3'
