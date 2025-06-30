from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image

import random
import h5py
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from scipy import ndimage
from PIL import Image
from torchvision import transforms
from io import BytesIO
import cv2
from random import random, choice, shuffle
from scipy.ndimage.filters import gaussian_filter
import pickle
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}

def recursively_read(rootdir, must_contain):
    out = []
    for r, d, f in os.walk(rootdir):
        for file in f:
            full_path = os.path.join(r, file)
            if must_contain in full_path:
                out.append(full_path)
    return out

def get_list(path, must_contain=''):
    if ".pickle" in path:
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        image_list = [ item for item in image_list if must_contain in item ]
    else:
        image_list = recursively_read(path, must_contain)
    return image_list

def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")

def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)

def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]

def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    img = np.array(img)
    out.close()
    return img

jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)

rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}

def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return transforms.functional.resize(img, opt.loadSize, interpolation=rz_dict[interp])

def data_augment(img, opt):
    img = np.array(img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)

class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, train=True):
        super(NPY_datasets, self)
        if train:
            images_list = sorted(os.listdir(path_Data+'train/images/'))
            masks_list = sorted(os.listdir(path_Data+'train/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'train/images/' + images_list[i]
                mask_path = path_Data+'train/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.train_transformer
        else:
            images_list = sorted(os.listdir(path_Data+'val/images/'))
            masks_list = sorted(os.listdir(path_Data+'val/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'val/images/' + images_list[i]
                mask_path = path_Data+'val/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.test_transformer
        
    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)
    


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

class UE_datasets(Dataset):
    def __init__(self, path_Data, config, train=True):
        super(UE_datasets, self)
        if train:
            images_list = sorted(os.listdir(path_Data+'/train/edited_image/'))
            masks_list = sorted(os.listdir(path_Data+'/train/mask_image/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'/train/edited_image/' + images_list[i]
                mask_path = path_Data+'/train/mask_image/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.train_transformer
        else:
            images_list = sorted(os.listdir(path_Data+'/val/edited_image/'))
            masks_list = sorted(os.listdir(path_Data+'/val/mask_image/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'/val/edited_image/' + images_list[i]
                mask_path = path_Data+'/val/mask_image/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.test_transformer
        
    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)



class RealFakeDataset(Dataset):
    def __init__(self, opt, split="train"):  # ✅ split: "train" / "val" / "test" / "custom_test"
        assert split in ["train", "val", "test", "custom_test"]
        self.opt = opt
        self.split = split

        # ===============================
        # ✅ Step 1: 加载图像路径
        # ===============================
        if split in ["train", "val"]:
            real_list, fake_list = self.load_train_val(opt, split)
        elif split in ["test", "custom_test"]:
            real_list, fake_list = self.load_test(opt, split)

        # ===============================
        # ✅ Step 2: 生成标签字典
        # ===============================
        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = 0
        for i in fake_list:
            self.labels_dict[i] = 1

        self.total_list = real_list + fake_list
        shuffle(self.total_list)


        # ===============================
        # ✅ Step 3: 构建 transform
        # ===============================
        if split == "train":
            crop_func = transforms.RandomCrop(opt.cropSize)
        elif opt.no_crop:
            crop_func = transforms.Lambda(lambda img: img)
        else:
            crop_func = transforms.CenterCrop(opt.cropSize)

        if split == "train" and not opt.no_flip:
            flip_func = transforms.RandomHorizontalFlip()
        else:
            flip_func = transforms.Lambda(lambda img: img)

        if split == "test" and opt.no_resize:
            rz_func = transforms.Lambda(lambda img: img)
        else:
            rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))

        stat_from = "imagenet" if opt.arch.lower().startswith("imagenet") else "clip"
        print(f"mean and std stats are from: {stat_from}")

        if '2b' not in opt.arch:
            print("using Official CLIP's normalization")
            self.transform = transforms.Compose([
                rz_func,
                transforms.Lambda(lambda img: data_augment(img, opt)),
                crop_func,
                flip_func,
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
            ])
        else:
            print("Using CLIP 2B transform")
            self.transform = None  

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        img_path = self.total_list[idx]
        label = self.labels_dict[img_path]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def load_train_val(self, opt, split):
        real_list, fake_list = [], []
        pickle_name = "train.pickle" if split == "train" else "val.pickle"

        if opt.data_mode == 'ours':
            real_list = get_list(os.path.join(opt.real_list_path, pickle_name))
            fake_list = get_list(os.path.join(opt.fake_list_path, pickle_name))

        elif opt.data_mode == 'wang2020':
            temp = 'train/' if split == 'train' else 'test/'
            real_list = get_list(os.path.join(opt.wang2020_data_path, temp), must_contain='0_real')
            fake_list = get_list(os.path.join(opt.wang2020_data_path, temp), must_contain='1_fake')

        elif opt.data_mode == 'ours_wang2020':
            real_list = get_list(os.path.join(opt.real_list_path, pickle_name))
            fake_list = get_list(os.path.join(opt.fake_list_path, pickle_name))
            temp = 'train/progan' if split == 'train' else 'test/progan'
            real_list += get_list(os.path.join(opt.wang2020_data_path, temp), must_contain='0_real')
            fake_list += get_list(os.path.join(opt.wang2020_data_path, temp), must_contain='1_fake')

        elif opt.data_mode == 'RFNT':
            datasets_list = ['fdmas_sample']
            folder = 'train' if split == 'train' else 'valid'
            dataset_path = "./datasets/"
            for dataset_index in datasets_list:
                real_list += get_list(os.path.join(dataset_path, dataset_index, folder, "0_real"))
                fake_list += get_list(os.path.join(dataset_path, dataset_index, folder, "1_fake"))

        else:
            raise ValueError(f"Unsupported data_mode: {opt.data_mode}")

        return real_list, fake_list

    def load_test(self, opt, split):
        real_list, fake_list = [], []
        generator_path = opt.data_path  # ✅ 正确：由外部传入 gen_path，而不是 test_root

        if not os.path.exists(generator_path):
            raise FileNotFoundError(f"Test data path not found: {generator_path}")

        # 检查是否存在子目录（多层结构，如 cyclegan/apple）
        subdirs = [
            os.path.join(generator_path, d)
            for d in os.listdir(generator_path)
            if os.path.isdir(os.path.join(generator_path, d))
        ]
        has_subcategories = any(
            os.path.isdir(os.path.join(subdir, "0_real")) or os.path.isdir(os.path.join(subdir, "1_fake"))
            for subdir in subdirs
        )

        if has_subcategories:
            # 多层结构：遍历每个子目录收集 0_real 和 1_fake
            for subdir in subdirs:
                real_dir = os.path.join(subdir, "0_real")
                fake_dir = os.path.join(subdir, "1_fake")

                if os.path.isdir(real_dir):
                    real_list += [
                        os.path.join(real_dir, f)
                        for f in os.listdir(real_dir)
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))
                    ]
                if os.path.isdir(fake_dir):
                    fake_list += [
                        os.path.join(fake_dir, f)
                        for f in os.listdir(fake_dir)
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))
                    ]
        else:
            # 单层结构：直接在 generator_path 下找 0_real 和 1_fake
            real_dir = os.path.join(generator_path, "0_real")
            fake_dir = os.path.join(generator_path, "1_fake")

            if os.path.isdir(real_dir):
                real_list += [
                    os.path.join(real_dir, f)
                    for f in os.listdir(real_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ]
            if os.path.isdir(fake_dir):
                fake_list += [
                    os.path.join(fake_dir, f)
                    for f in os.listdir(fake_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ]

        generator_name = os.path.basename(generator_path)
        print(f"[{split.upper()}/{generator_name}] Loaded {len(real_list)} real and {len(fake_list)} fake samples.")
        return real_list, fake_list