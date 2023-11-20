import glob
import os
from random import choice

from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import torch
import numpy as np
import json

from data.transforms import sync_resize, splicing, copy_move_by_mask, random_crop, random_flip_and_rot90, get_feature, \
    gen_edge_mask, jpeg_compression, gaussian_blur, resize, gaussian_noise, get_transforms, median_blur, center_crop


# _______________________________________数据路劲加载__________________________________ #
def read_txt(txt_path):
    img_paths = []
    mask_paths = []
    with open(txt_path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            img_paths.append(line.split(' ')[0])
            mask_paths.append(line.split(' ')[1])
    return img_paths, mask_paths


# 同步打乱两个数组
def shuffle_two_array(a, b, seed=None):
    state = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(a)
    np.random.set_state(state)
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(b)
    return a, b


# 过滤 CASIA2 里面img 和 msak 不匹配的图片 #
def filter_img(img_paths, mask_paths, filter_str=None):
    print('Start filter images ......')
    new_img_paths = []
    new_mask_paths = []
    for i in range(len(img_paths)):
        try:
            if 'Au' in img_paths[i]:
                new_img_paths.append(img_paths[i])
            else:
                image = cv2.imread(img_paths[i])
                mask = cv2.imread(mask_paths[i])
                if image.shape[0] == mask.shape[0] and image.shape[1] == mask.shape[1]:
                    if filter_str is not None:
                        if filter_str in img_paths[i]:
                            new_img_paths.append(img_paths[i])
                            new_mask_paths.append(mask_paths[i])
                    else:
                        new_img_paths.append(img_paths[i])
                        new_mask_paths.append(mask_paths[i])
        except:
            print(img_paths[i], mask_paths[i])
    print('Finished filter images!')
    print('New:', len(new_img_paths), len(new_mask_paths))
    return new_img_paths, new_mask_paths


def get_data_from_path(root_path, data_name='CASIA', data_type='train', use_au=False, au_ratio=0.5,
                       train_ratio=0.75):
    if data_name == 'CASIA':
        if data_type == 'test':
            test_root_path = os.path.join(root_path, 'CASIA', 'CASIA1')
            # test_root_path = os.path.join(root_path, 'CASIA', 'CASIA1_new')
            test_img_paths = sorted(glob.glob(test_root_path + '/Tp_new/*'))
            test_mask_paths = sorted(glob.glob(test_root_path + '/mask_new/*'))
            if use_au:
                au_img = sorted(glob.glob(test_root_path + '/Au/*'))
                au_img = au_img[:int(len(au_img) * au_ratio)]
                test_img_paths += au_img
            print(len(test_img_paths), len(test_mask_paths))
            return test_img_paths, test_mask_paths
        else:
            train_root_path = os.path.join(root_path, 'CASIA', 'CASIA2')
            train_img_paths = sorted(glob.glob(train_root_path + '/Tp/*'))
            train_mask_paths = sorted(glob.glob(train_root_path + '/CASIA2gt/*'))
            if use_au:
                au_img = sorted(glob.glob(train_root_path + '/Au/*'))
                au_img = au_img[:int(len(au_img) * au_ratio)]
                train_img_paths += au_img
            print(len(train_img_paths), len(train_mask_paths))
            return train_img_paths, train_mask_paths
    elif data_name == 'Columbia':
        img_root_paths = os.path.join(root_path, 'Columbia')
        img_paths = sorted(glob.glob(img_root_paths + '/4cam_splc/*.tif'))
        mask_paths = sorted(glob.glob(img_root_paths + '/columbia_mask/*.jpg'))
        if data_type == 'train':
            train_img_paths = img_paths[:int(len(img_paths) * train_ratio)]
            train_mask_paths = mask_paths[:int(len(img_paths) * train_ratio)]
            print(len(train_img_paths), len(train_mask_paths))
            return train_img_paths, train_mask_paths
        else:
            test_img_paths = img_paths[int(len(img_paths) * train_ratio):]
            test_mask_paths = mask_paths[int(len(img_paths) * train_ratio):]
            print(len(test_img_paths), len(test_mask_paths))
            return test_img_paths, test_mask_paths
    elif data_name == 'COVERAGE':
        img_root_paths = os.path.join(root_path, 'COVERAGE')
        img_paths = sorted(glob.glob(img_root_paths + '/image/*t.tif'))
        mask_paths = sorted(glob.glob(img_root_paths + '/mask/*forged.tif'))
        if data_type == 'train':
            train_img_paths = img_paths[:int(len(img_paths) * train_ratio)]
            train_mask_paths = mask_paths[:int(len(img_paths) * train_ratio)]
            print(len(train_img_paths), len(train_mask_paths))
            return train_img_paths, train_mask_paths
        else:
            test_img_paths = img_paths[int(len(img_paths) * train_ratio):]
            test_mask_paths = mask_paths[int(len(img_paths) * train_ratio):]
            print(len(test_img_paths), len(test_mask_paths))
            return test_img_paths, test_mask_paths
    elif data_name == 'NIST16':
        img_root_paths = os.path.join(root_path, 'NIST2016')
        img_paths = sorted(glob.glob(img_root_paths + '/tamper/manipulation_copy/*'))
        mask_paths = sorted(glob.glob(img_root_paths + '/mask/manipulation_copy/*'))
        if data_type == 'train':
            train_img_paths = img_paths[:404]
            train_mask_paths = mask_paths[:404]
            print(len(train_img_paths), len(train_mask_paths))
            return train_img_paths, train_mask_paths
        else:
            test_img_paths = img_paths[404:]
            test_mask_paths = mask_paths[404:]
            print(len(test_img_paths), len(test_mask_paths))
            return test_img_paths, test_mask_paths
    elif data_name == 'IMD2020':
        img_root_paths = os.path.join(root_path, 'IMD2020')
        if data_type == 'train':
            train_img_paths = sorted(glob.glob((img_root_paths + '/train/tamper/*')))
            train_mask_paths = sorted(glob.glob((img_root_paths + '/train/mask/*')))
            print(len(train_img_paths), len(train_mask_paths))
            return train_img_paths, train_mask_paths
        else:
            test_img_paths = sorted(glob.glob((img_root_paths + '/test/tamper/*')))
            test_mask_paths = sorted(glob.glob((img_root_paths + '/test/mask/*')))
            print(len(test_img_paths), len(test_mask_paths))
            return test_img_paths, test_mask_paths
    elif data_name == 'DEFACTO':
        img_root_paths = os.path.join(root_path, 'DEFACTO')
        img_paths, mask_paths = read_txt(txt_path=os.path.join(img_root_paths, 'defacto.txt'))
        for i in range(len(img_paths)):
            img_paths[i] = os.path.join(img_root_paths, img_paths[i])
            mask_paths[i] = os.path.join(img_root_paths, mask_paths[i])
        if data_type == 'train':
            train_img_paths = img_paths[:int(len(img_paths) * train_ratio)]
            train_mask_paths = mask_paths[:int(len(img_paths) * train_ratio)]
            print(len(train_img_paths), len(train_mask_paths))
            return train_img_paths, train_mask_paths
        else:
            test_img_paths = img_paths[int(len(img_paths) * train_ratio):]
            test_mask_paths = mask_paths[int(len(img_paths) * train_ratio):]
            print(len(test_img_paths), len(test_mask_paths))
            return test_img_paths, test_mask_paths
    else:
        assert False, 'data_name should be [CASIA, Columbia, COVERAGE, NIST16, IMD2020, DEFACTO]'


class CASIADataset(Dataset):
    def __init__(self, root_path='/data2/hulh/dataset', transform=None, is_random_crop=True, crop_shape=[256, 256],
                 is_one_hot=False, data_type='train', is_resize=False, use_au=False, use_edge=False, feature_name=None,
                 self_aug=False, self_random=False, predict=False, au_ratio=0.5):
        assert feature_name in [None, 'hog', 'sobel']
        self.is_random_crop = is_random_crop
        self.crop_shape = crop_shape
        self.transform = transform
        self.is_one_hot = is_one_hot
        self.data_type = data_type
        self.is_resize = is_resize
        self.use_edge = use_edge
        self.use_au = use_au
        self.feature_name = feature_name
        self.self_aug = self_aug
        self.self_random = self_random
        self.predict = predict
        self.use_edge = False if self.feature_name else use_edge  # 使用了手工特征只能就不用edge

        self.img_paths, self.mask_paths = get_data_from_path(root_path=root_path, data_name='CASIA',
                                                             data_type=data_type,
                                                             use_au=use_au,
                                                             au_ratio=au_ratio)

        if data_type == 'train':
            self.img_paths, self.mask_paths = filter_img(self.img_paths, self.mask_paths, filter_str=None)

        print('Total images: ', len(self.img_paths), len(self.mask_paths))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        shape = image.shape[:2]
        mask_path = self.mask_paths[index]
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) // 255
        label = 1 if np.max(mask) > 0 else 0


        assert image.shape[0] == mask.shape[0] or image.shape[1] == mask.shape[1], \
            'the image doesn\'t match the mask! image_path:' + image.shape + mask.shape
        if self.is_resize:
            image, mask = sync_resize(image, mask, resize_shape=self.crop_shape)

        if self.is_random_crop and self.data_type == 'train':
            if self.self_aug and label == 1:
                image, mask = self.splicing(image, mask, self_random=self.self_random)
                image, mask = copy_move_by_mask(image, mask, self_random=self.self_random)
            image, mask = random_crop(image, mask, crop_shape=self.crop_shape)
            image, mask = random_flip_and_rot90(image, mask, is_rot=False)
            image = jpeg_compression(image)  # 很关键的数据增强
            image, mask = median_blur(image, mask)
        elif self.is_random_crop:
            image, mask = center_crop(image, mask, crop_shape=self.crop_shape)

        if self.feature_name:
            feature = get_feature(image, feature_name=self.feature_name)
            feature = np.array(feature)

        if self.transform is not None:
            image = self.transform(Image.fromarray(image))

        if self.use_edge:
            edge_mask = gen_edge_mask(mask)
            # edge_mask = cv2.resize(edge_mask, (edge_mask.shape[1] // 4, edge_mask.shape[0] // 4))
        if self.is_one_hot:
            mask = np.array([mask == 0, mask == 1], dtype=int)
            if self.use_edge:
                edge_mask = np.array([edge_mask == 0, edge_mask == 1], dtype=int)
        else:
            mask = np.expand_dims(mask, axis=0)
            if self.use_edge:
                edge_mask = np.expand_dims(edge_mask, axis=0)
        if self.use_edge:
            if self.use_au:
                return image, mask, edge_mask, label
            return image, mask, edge_mask
        if self.feature_name:
            if self.predict:
                return image, mask, feature, shape
            else:
                return image, mask, feature
        if self.use_au:
            return image, mask, label
        return image, mask

    def splicing(self, image, mask, p=0.5, self_random=False):
        if np.random.random() > p:
            # print(img.sum(), img.shape)
            cmix_path = np.random.choice(self.img_paths, size=1)[0]
            while np.max(cv2.imread(
                    cmix_path.replace('Tp/', 'CASIA2gt/').replace('.jpg', '_gt.png').replace('.tif', '_gt.png'))) == 0:
                cmix_path = np.random.choice(self.img_paths, size=1)[0]
            cmix_img = cv2.imread(cmix_path)
            cmix_img = cv2.cvtColor(cmix_img, cv2.COLOR_BGR2RGB)

            if 'Au/' in cmix_path:
                cmix_mask = np.zeros((cmix_img.shape[0], cmix_img.shape[1]))
            else:
                mask_path = cmix_path.replace('Tp/', 'CASIA2gt/').replace('.jpg', '_gt.png').replace('.tif', '_gt.png')
                cmix_mask = cv2.imread(mask_path)
                cmix_mask = cv2.cvtColor(cmix_mask, cv2.COLOR_BGR2GRAY) // 255  # 把 [0  255](0: 黑色, 255: 白色) 转 [0 1]

            image, mask = splicing(image, mask, cmix_img, cmix_mask, self_random=self_random)

        return image, mask


class ColumbiaDataset(Dataset):
    def __init__(self, root_path='/data2/hulh/dataset', transform=None, is_random_crop=True, crop_shape=[256, 256],
                 is_one_hot=False,
                 data_type='train', is_resize=False, use_edge=False, feature_name=None, self_aug=False,
                 self_random=False, predict=False):
        assert feature_name in [None, 'hog', 'sobel']
        self.is_random_crop = is_random_crop
        self.crop_shape = crop_shape
        self.transform = transform
        self.is_one_hot = is_one_hot
        self.data_type = data_type
        self.is_resize = is_resize
        self.use_edge = use_edge
        self.feature_name = feature_name
        self.self_aug = self_aug
        self.self_random = self_random
        self.predict = predict
        self.use_edge = False if self.feature_name else use_edge  # 使用了手工特征只能就不用edge

        self.img_paths, self.mask_paths = get_data_from_path(root_path=root_path, data_name='Columbia',
                                                             data_type=data_type)

        print('Total images: ', len(self.img_paths), len(self.mask_paths))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        shape = image.shape[:2]

        mask_path = self.mask_paths[index]
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) // 255  # 把 [0  255](0: 黑色, 255: 白色) 转 [0 1]

        assert image.shape[0] == mask.shape[0] or image.shape[1] == mask.shape[1], \
            'the image doesn\'t match the mask! image_path:' + img_path
        if self.is_resize:
            image, mask = sync_resize(image, mask, resize_shape=self.crop_shape)

        if self.is_random_crop and self.data_type == 'train':
            if self.self_aug:
                image, mask = self.splicing(image, mask, self_random=self.self_random)
                image, mask = copy_move_by_mask(image, mask, self_random=self.self_random)
            image, mask = random_crop(image, mask, crop_shape=self.crop_shape)
            image, mask = random_flip_and_rot90(image, mask, is_rot=True)

        if self.feature_name:
            feature = get_feature(image, feature_name=self.feature_name)
            feature = np.array(feature)

        if self.transform is not None:
            image = self.transform(Image.fromarray(image))

        if self.use_edge:
            edge_mask = gen_edge_mask(mask)
            # edge_mask = cv2.resize(edge_mask, (edge_mask.shape[1] // 4, edge_mask.shape[0] // 4))
        if self.is_one_hot:
            mask = np.array([mask == 0, mask == 1], dtype=int)
            if self.use_edge:
                edge_mask = np.array([edge_mask == 0, edge_mask == 1], dtype=int)
        else:
            mask = np.expand_dims(mask, axis=0)
            if self.use_edge:
                edge_mask = np.expand_dims(edge_mask, axis=0)
        if self.use_edge:
            return image, mask, edge_mask
        if self.feature_name:
            if self.predict:
                return image, mask, feature, shape
            else:
                return image, mask, feature

        return image, mask

    def splicing(self, image, mask, p=0.5, self_random=False):
        if np.random.random() > p:
            # print(img.sum(), img.shape)
            cmix_path = np.random.choice(self.img_paths, size=1)[0]
            cmix_img = cv2.imread(cmix_path)
            cmix_img = cv2.cvtColor(cmix_img, cv2.COLOR_BGR2RGB)

            if 'Au/' in cmix_path:
                cmix_mask = np.zeros((cmix_img.shape[0], cmix_img.shape[1]))
            else:
                mask_path = cmix_path.replace('Tp/', 'CASIA2gt/').replace('.jpg', '_gt.png').replace('.tif', '_gt.png')
                cmix_mask = cv2.imread(mask_path)
                cmix_mask = cv2.cvtColor(cmix_mask, cv2.COLOR_BGR2GRAY) // 255  # 把 [0  255](0: 黑色, 255: 白色) 转 [0 1]

            image, mask = splicing(image, mask, cmix_img, cmix_mask, self_random=self_random)

        return image, mask


class COVERAGEDataset(Dataset):
    def __init__(self, root_path='/data2/hulh/dataset', transform=None, is_random_crop=True, crop_shape=[256, 256],
                 is_one_hot=False,
                 data_type='train', is_resize=False, use_edge=False, feature_name=None, self_aug=False,
                 self_random=False, predict=False):
        assert feature_name in [None, 'hog', 'sobel']
        self.is_random_crop = is_random_crop
        self.crop_shape = crop_shape
        self.transform = transform
        self.is_one_hot = is_one_hot
        self.data_type = data_type
        self.is_resize = is_resize
        self.use_edge = use_edge
        self.feature_name = feature_name
        self.self_aug = self_aug
        self.self_random = self_random
        self.predict = predict
        self.use_edge = False if self.feature_name else use_edge  # 使用了手工特征只能就不用edge

        self.img_paths, self.mask_paths = get_data_from_path(root_path=root_path, data_name='COVERAGE',
                                                             data_type=data_type)

        print('Total images: ', len(self.img_paths), len(self.mask_paths))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        shape = image.shape[:2]

        mask_path = self.mask_paths[index]
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) // 255  # 把 [0  255](0: 黑色, 255: 白色) 转 [0 1]

        assert image.shape[0] == mask.shape[0] or image.shape[1] == mask.shape[1], \
            'the image doesn\'t match the mask! image_path:' + img_path
        if self.is_resize:
            image, mask = sync_resize(image, mask, resize_shape=self.crop_shape)

        if self.is_random_crop and self.data_type == 'train':
            if self.self_aug:
                image, mask = self.splicing(image, mask, self_random=self.self_random)
                image, mask = copy_move_by_mask(image, mask, self_random=self.self_random)
            image, mask = random_crop(image, mask, crop_shape=self.crop_shape)
            image, mask = random_flip_and_rot90(image, mask, is_rot=True)

        if self.feature_name:
            feature = get_feature(image, feature_name=self.feature_name)
            feature = np.array(feature)

        if self.transform is not None:
            image = self.transform(Image.fromarray(image))

        if self.use_edge:
            edge_mask = gen_edge_mask(mask)
            # edge_mask = cv2.resize(edge_mask, (edge_mask.shape[1] // 4, edge_mask.shape[0] // 4))
        if self.is_one_hot:
            mask = np.array([mask == 0, mask == 1], dtype=int)
            if self.use_edge:
                edge_mask = np.array([edge_mask == 0, edge_mask == 1], dtype=int)
        else:
            mask = np.expand_dims(mask, axis=0)
            if self.use_edge:
                edge_mask = np.expand_dims(edge_mask, axis=0)
        if self.use_edge:
            return image, mask, edge_mask
        if self.feature_name:
            if self.predict:
                return image, mask, feature, shape
            else:
                return image, mask, feature

        return image, mask

    def splicing(self, image, mask, p=0.5, self_random=False):
        if np.random.random() > p:
            # print(img.sum(), img.shape)
            cmix_path = np.random.choice(self.img_paths, size=1)[0]
            cmix_img = cv2.imread(cmix_path)
            cmix_img = cv2.cvtColor(cmix_img, cv2.COLOR_BGR2RGB)

            if 'Au/' in cmix_path:
                cmix_mask = np.zeros((cmix_img.shape[0], cmix_img.shape[1]))
            else:
                mask_path = cmix_path.replace('Tp/', 'CASIA2gt/').replace('.jpg', '_gt.png').replace('.tif', '_gt.png')
                cmix_mask = cv2.imread(mask_path)
                cmix_mask = cv2.cvtColor(cmix_mask, cv2.COLOR_BGR2GRAY) // 255  # 把 [0  255](0: 黑色, 255: 白色) 转 [0 1]

            image, mask = splicing(image, mask, cmix_img, cmix_mask, self_random=self_random)

        return image, mask


class NISTDataset(Dataset):
    def __init__(self, root_path='/data2/hulh/dataset', transform=None, is_random_crop=True, crop_shape=[256, 256],
                 is_one_hot=False,
                 data_type='train', is_resize=False, use_edge=False, feature_name=None, self_aug=False,
                 self_random=False, predict=False, test_post=False):
        assert feature_name in [None, 'hog', 'sobel']
        self.is_random_crop = is_random_crop
        self.crop_shape = crop_shape
        self.transform = transform
        self.is_one_hot = is_one_hot
        self.test_post = test_post
        self.data_type = data_type
        self.is_resize = is_resize
        self.use_edge = use_edge
        self.feature_name = feature_name
        self.self_aug = self_aug
        self.self_random = self_random
        self.predict = predict
        self.use_edge = False if self.feature_name else use_edge  # 使用了手工特征只能就不用edge

        self.img_paths, self.mask_paths = get_data_from_path(root_path=root_path, data_name='NIST16',
                                                             data_type=data_type)

        print('Total images: ', len(self.img_paths), len(self.mask_paths))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        shape = image.shape[:2]

        mask_path = self.mask_paths[index]
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) // 255  # 把 [0  255](0: 黑色, 255: 白色) 转 [0 1]

        assert image.shape[0] == mask.shape[0] or image.shape[1] == mask.shape[1], \
            'the image doesn\'t match the mask! image_path:' + img_path
        if self.is_resize:
            image, mask = sync_resize(image, mask, resize_shape=self.crop_shape)

        if self.is_random_crop and self.data_type == 'train':
            if self.self_aug:
                image, mask = self.splicing(image, mask, self_random=self.self_random)
                image, mask = copy_move_by_mask(image, mask, self_random=self.self_random)
            image, mask = random_crop(image, mask, crop_shape=self.crop_shape)
            image, mask = random_flip_and_rot90(image, mask, is_rot=True)

        if self.feature_name:
            feature = get_feature(image, feature_name=self.feature_name)
            feature = np.array(feature)

        if self.test_post and self.data_type == 'test':
            image = jpeg_compression(image, qfs=[100], p=1.0)
            image, mask = gaussian_blur(image, mask, p=1.0, param=15)
            image = resize(image, rate=0.25, p=1)
            image = gaussian_noise(image, sd=15)

        if self.transform is not None:
            image = self.transform(Image.fromarray(image))

        if self.use_edge:
            edge_mask = gen_edge_mask(mask)
            # edge_mask = cv2.resize(edge_mask, (edge_mask.shape[1] // 4, edge_mask.shape[0] // 4))
        if self.is_one_hot:
            mask = np.array([mask == 0, mask == 1], dtype=int)
            if self.use_edge:
                edge_mask = np.array([edge_mask == 0, edge_mask == 1], dtype=int)
        else:
            mask = np.expand_dims(mask, axis=0)
            if self.use_edge:
                edge_mask = np.expand_dims(edge_mask, axis=0)
        if self.use_edge:
            return image, mask, edge_mask
        if self.feature_name:
            if self.predict:
                return image, mask, feature, shape
            else:
                return image, mask, feature

        return image, mask

    def splicing(self, image, mask, p=0.5, self_random=False):
        if np.random.random() > p:
            # print(img.sum(), img.shape)
            cmix_path = np.random.choice(self.img_paths, size=1)[0]
            cmix_img = cv2.imread(cmix_path)
            cmix_img = cv2.cvtColor(cmix_img, cv2.COLOR_BGR2RGB)

            if 'Au/' in cmix_path:
                cmix_mask = np.zeros((cmix_img.shape[0], cmix_img.shape[1]))
            else:
                mask_path = cmix_path.replace('Tp/', 'CASIA2gt/').replace('.jpg', '_gt.png').replace('.tif', '_gt.png')
                cmix_mask = cv2.imread(mask_path)
                cmix_mask = cv2.cvtColor(cmix_mask, cv2.COLOR_BGR2GRAY) // 255  # 把 [0  255](0: 黑色, 255: 白色) 转 [0 1]

            image, mask = splicing(image, mask, cmix_img, cmix_mask, self_random=self_random)

        return image, mask


class IMDDataset(Dataset):
    def __init__(self, root_path='/data2/hulh/dataset', transform=None, is_random_crop=True, crop_shape=[256, 256],
                 is_one_hot=False,
                 data_type='train', is_resize=False, use_edge=False, feature_name=None, self_aug=False,
                 self_random=False, predict=False):
        assert feature_name in [None, 'hog', 'sobel']
        self.is_random_crop = is_random_crop
        self.crop_shape = crop_shape
        self.transform = transform
        self.is_one_hot = is_one_hot
        self.data_type = data_type
        self.is_resize = is_resize
        self.use_edge = use_edge
        self.feature_name = feature_name
        self.self_aug = self_aug
        self.self_random = self_random
        self.predict = predict
        self.use_edge = False if self.feature_name else use_edge  # 使用了手工特征只能就不用edge

        self.img_paths, self.mask_paths = get_data_from_path(root_path=root_path, data_name='IMD2020',
                                                             data_type=data_type)

        print('Total images: ', len(self.img_paths), len(self.mask_paths))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        shape = image.shape[:2]

        mask_path = self.mask_paths[index]
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) // 255  # 把 [0  255](0: 黑色, 255: 白色) 转 [0 1]

        assert image.shape[0] == mask.shape[0] or image.shape[1] == mask.shape[1], \
            'the image doesn\'t match the mask! image_path:' + img_path
        if self.is_resize:
            image, mask = sync_resize(image, mask, resize_shape=self.crop_shape)

        if self.is_random_crop and self.data_type == 'train':
            if self.self_aug:
                image, mask = self.splicing(image, mask, self_random=self.self_random)
                image, mask = copy_move_by_mask(image, mask, self_random=self.self_random)
            image, mask = random_crop(image, mask, crop_shape=self.crop_shape)
            image, mask = random_flip_and_rot90(image, mask, is_rot=True)

        if self.feature_name:
            feature = get_feature(image, feature_name=self.feature_name)
            feature = np.array(feature)

        if self.transform is not None:
            image = self.transform(Image.fromarray(image))

        if self.use_edge:
            edge_mask = gen_edge_mask(mask)
            # edge_mask = cv2.resize(edge_mask, (edge_mask.shape[1] // 4, edge_mask.shape[0] // 4))
        if self.is_one_hot:
            mask = np.array([mask == 0, mask == 1], dtype=int)
            if self.use_edge:
                edge_mask = np.array([edge_mask == 0, edge_mask == 1], dtype=int)
        else:
            mask = np.expand_dims(mask, axis=0)
            if self.use_edge:
                edge_mask = np.expand_dims(edge_mask, axis=0)
        if self.use_edge:
            return image, mask, edge_mask
        if self.feature_name:
            if self.predict:
                return image, mask, feature, shape
            else:
                return image, mask, feature
        label = 1
        return image, mask, label

    def splicing(self, image, mask, p=0.5, self_random=False):
        if np.random.random() > p:
            # print(img.sum(), img.shape)
            cmix_path = np.random.choice(self.img_paths, size=1)[0]
            cmix_img = cv2.imread(cmix_path)
            cmix_img = cv2.cvtColor(cmix_img, cv2.COLOR_BGR2RGB)

            if 'Au/' in cmix_path:
                cmix_mask = np.zeros((cmix_img.shape[0], cmix_img.shape[1]))
            else:
                mask_path = cmix_path.replace('Tp/', 'CASIA2gt/').replace('.jpg', '_gt.png').replace('.tif', '_gt.png')
                cmix_mask = cv2.imread(mask_path)
                cmix_mask = cv2.cvtColor(cmix_mask, cv2.COLOR_BGR2GRAY) // 255  # 把 [0  255](0: 黑色, 255: 白色) 转 [0 1]

            image, mask = splicing(image, mask, cmix_img, cmix_mask, self_random=self_random)

        return image, mask


class DEFACTODataset(Dataset):
    def __init__(self, root_path='/data2/hulh/dataset', transform=None, is_random_crop=True, crop_shape=[256, 256],
                 is_one_hot=False,
                 data_type='train', is_resize=False, use_edge=False, feature_name=None, self_aug=False,
                 self_random=False, predict=False):
        assert feature_name in [None, 'hog', 'sobel']
        self.is_random_crop = is_random_crop
        self.crop_shape = crop_shape
        self.transform = transform
        self.is_one_hot = is_one_hot
        self.data_type = data_type
        self.is_resize = is_resize
        self.use_edge = use_edge
        self.feature_name = feature_name
        self.self_aug = self_aug
        self.self_random = self_random
        self.predict = predict
        self.use_edge = False if self.feature_name else use_edge  # 使用了手工特征只能就不用edge

        self.img_paths, self.mask_paths = get_data_from_path(root_path=root_path, data_name='DEFACTO',
                                                             data_type=data_type)

        print('Total images: ', len(self.img_paths), len(self.mask_paths))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        shape = image.shape[:2]

        mask_path = self.mask_paths[index]
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) // 255  # 把 [0  255](0: 黑色, 255: 白色) 转 [0 1]

        assert image.shape[0] == mask.shape[0] or image.shape[1] == mask.shape[1], \
            'the image doesn\'t match the mask! image_path:' + img_path
        if self.is_resize:
            image, mask = sync_resize(image, mask, resize_shape=self.crop_shape)

        if self.is_random_crop and self.data_type == 'train':
            if self.self_aug:
                image, mask = self.splicing(image, mask, self_random=self.self_random)
                image, mask = copy_move_by_mask(image, mask, self_random=self.self_random)
            image, mask = random_crop(image, mask, crop_shape=self.crop_shape)
            image, mask = random_flip_and_rot90(image, mask, is_rot=True)

        if self.feature_name:
            feature = get_feature(image, feature_name=self.feature_name)
            feature = np.array(feature)

        if self.transform is not None:
            image = self.transform(Image.fromarray(image))

        if self.use_edge:
            edge_mask = gen_edge_mask(mask)
            # edge_mask = cv2.resize(edge_mask, (edge_mask.shape[1] // 4, edge_mask.shape[0] // 4))
        if self.is_one_hot:
            mask = np.array([mask == 0, mask == 1], dtype=int)
            if self.use_edge:
                edge_mask = np.array([edge_mask == 0, edge_mask == 1], dtype=int)
        else:
            mask = np.expand_dims(mask, axis=0)
            if self.use_edge:
                edge_mask = np.expand_dims(edge_mask, axis=0)
        if self.use_edge:
            return image, mask, edge_mask
        if self.feature_name:
            if self.predict:
                return image, mask, feature, shape
            else:
                return image, mask, feature

        return image, mask

    def splicing(self, image, mask, p=0.5, self_random=False):
        if np.random.random() > p:
            # print(img.sum(), img.shape)
            cmix_path = np.random.choice(self.img_paths, size=1)[0]
            cmix_img = cv2.imread(cmix_path)
            cmix_img = cv2.cvtColor(cmix_img, cv2.COLOR_BGR2RGB)

            if 'Au/' in cmix_path:
                cmix_mask = np.zeros((cmix_img.shape[0], cmix_img.shape[1]))
            else:
                mask_path = cmix_path.replace('Tp/', 'CASIA2gt/').replace('.jpg', '_gt.png').replace('.tif', '_gt.png')
                cmix_mask = cv2.imread(mask_path)
                cmix_mask = cv2.cvtColor(cmix_mask, cv2.COLOR_BGR2GRAY) // 255  # 把 [0  255](0: 黑色, 255: 白色) 转 [0 1]

            image, mask = splicing(image, mask, cmix_img, cmix_mask, self_random=self_random)

        return image, mask


if __name__ == '__main__':
    xdl = CASIADataset(transform=get_transforms(data_type='train'), is_resize=True, crop_shape=[256, 256],
                       data_type='train', use_edge=False)
    dataloader = DataLoader(xdl, batch_size=4, shuffle=False, num_workers=1)
    for i, data in enumerate(dataloader):
        image, label = data
        print(image.shape, label.shape)
        if i == 10:
            break
