# ms1mv3_local_rec_512.py
import os
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import TensorDataset
import cv2
from omegaconf import ListConfig
from tqdm import tqdm
import random
random.seed(42)



def get_all_image_files(directory, extensions=None):
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    
    image_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_files.append(os.path.abspath(os.path.join(root, file)))
    return image_files
class faceDataset_ID(TensorDataset):
    def __init__(self, root_path,ratio,form_gallery,logger=None, return_parsing=False):
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
        # import pdb;pdb.set_trace()
        if not self.form_gallery:
            if ratio > 0:
                # soft ratio
                if ratio<=1:
                    for key in self.idbook:
                        # self.idbook[key] = self.idbook[key][:int(len(self.idbook[key]) * ratio)]
                        self.real_idbook[key] = self.real_idbook[key][:int(len(self.real_idbook[key]) * ratio)] if key in self.real_idbook else []
                        self.fake_idbook[key] = self.fake_idbook[key][:int(len(self.fake_idbook[key]) * ratio)] if key in self.fake_idbook else []
                # hard ratio
                else:
                    for key in self.idbook:
                        # self.idbook[key] = self.idbook[key][:int(ratio)]
                        self.real_idbook[key] = self.real_idbook[key][:int(ratio)] if key in self.real_idbook else []
                        self.fake_idbook[key] = self.fake_idbook[key][:int(ratio)] if key in self.fake_idbook else []
            else:
                if ratio>=-1:
                    for key in self.idbook:
                        # self.idbook[key] = self.idbook[key][int(len(self.idbook[key]) * ratio):]
                        self.real_idbook[key] = self.real_idbook[key][int(len(self.real_idbook[key]) * ratio):] if key in self.real_idbook else []
                        self.fake_idbook[key] = self.fake_idbook[key][int(len(self.fake_idbook[key]) * ratio):] if key in self.fake_idbook else []
                else:
                    for key in self.idbook:
                        # self.idbook[key] = self.idbook[key][int(ratio):]
                        self.real_idbook[key] = self.real_idbook[key][int(ratio):] if key in self.real_idbook else []
                        self.fake_idbook[key] = self.fake_idbook[key][int(ratio):] if key in self.fake_idbook else []
        else:
            for key in self.idbook:
                self.real_idbook[key] = self.real_idbook[key][:int(ratio)] if key in self.real_idbook else []
                self.fake_idbook[key] = self.fake_idbook[key][:int(ratio)] if key in self.fake_idbook else []
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
                    # fake_id=ori_id if 'real' in self.image_list[item] else int(self.image_list[item].split('/')[-2].split('_')[1])
                    # if 'real' not in fake_path and not self.form_gallery:
                    #     base_dataset_path=fake_path.replace('deepfakes/image','datasets/kl').rsplit('/', 3)[0]+'/real'
                    #     source_path=base_dataset_path+f'/{fake_id}/'+f"{fake_path.split('/')[-1].split('.')[0].split('_')[1]}.jpg"
                    #     target_path=base_dataset_path+f'/{ori_id}/'+f"{fake_path.split('/')[-1].split('.')[0].split('_')[0]}.jpg"
                    #     source_img = Image.fromarray(cv2.imread(source_path)[:, :, ::-1]) # BGR to RGB
                    #     source_img = self.id_transform(source_img)
                    #     target_img = Image.fromarray(cv2.imread(target_path)[:, :, ::-1]) # BGR to RGB
                    #     target_img = self.id_transform(target_img)
                    # ret={
                    #     'image': fake_img,
                    #     'image_path': fake_path,
                    #     'source_path': source_path,
                    #     'target_path': target_path,
                    #     'pri_image': source_img,
                    #     'sec_image': target_img,
                    #     'pri_label': fake_id,
                    #     'sec_label': ori_id
                    # }
                    ret={
                        'image': fake_img,
                        'sec_label': ori_id
                    }
                    # else:
                    #     ret={
                    #         'image': fake_img,
                    #         'image_path': fake_path,
                    #         'source_path': fake_path,
                    #         'target_path': fake_path,
                    #         'pri_image': fake_img,
                    #         'sec_image': torch.zeros_like(fake_img),
                    #         'pri_label': ori_id,
                    #         'sec_label': ori_id
                    #     }
                elif 'vgg2' in fake_path:
                    ori_id=int(self.image_list[item].split('/')[-2].lstrip('n')) if 'real' in self.image_list[item] else int(self.image_list[item].split('/')[-2].split('_')[0].lstrip('n'))
                    # fake_id=ori_id if 'real' in self.image_list[item] else int(self.image_list[item].split('/')[-2].split('_')[1].lstrip('n'))
                    ret={
                        'image': fake_img,
                        'sec_label': ori_id
                    }
                elif 'cdfv2' in fake_path or 'id' in fake_path:
                    ori_id = int(self.image_list[item].split('/')[-2][2:]) if 'real' in self.image_list[item] else int(self.image_list[item].split('/')[-2].split('_')[0][2:])
                    # fake_id=ori_id if 'real' in self.image_list[item] else int(self.image_list[item].split('/')[-2].split('_')[1][2:])
                    ret={
                        'image': fake_img,
                        'sec_label': ori_id
                    }
                if self.return_parsing:
                    ret['parsing_image'] = parsing_img
                return ret
            except Exception as e:
                # print(f"Warning: Skipping bad image at {self.image_list[item]} due to {e}")
                item = random.randint(0, len(self.image_list) - 1)


    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    root = '/your/path/to/ms1mv3'