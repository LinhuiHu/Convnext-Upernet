import math
import os
import cv2
import numpy as np
from faker import Faker
from random import choice
import random
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from albumentations import Normalize, Compose
import albumentations as albu
from skimage import feature, filters, morphology



# 数据增强
def get_transforms(data_type='train', is_albu=False, size=[512, 512]):
    if not is_albu:
        default_data_transforms = {
            'train': transforms.Compose([
                # transforms.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.2),
                # transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), fillcolor=0, scale=(0.9, 1.1), shear=None),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        return default_data_transforms[data_type]
    else:
        if data_type == 'train':
            albu_transforms = albu.Compose([
                albu.OneOf([
                    albu.HorizontalFlip(p=0.5),
                    albu.Rotate(limit=[90, 90], p=0.5),
                    albu.Rotate(limit=[270, 270], p=0.5),
                ], p=0.5),
                albu.OneOf([
                    albu.GaussNoise(p=0.2),
                    albu.GaussianBlur(p=0.2),
                    albu.MedianBlur(p=0.2)
                ], p=0.5),
                albu.ImageCompression(quality_lower=75, quality_upper=100, p=0.5),
                # albu.RandomCrop(height=size[0], width=size[1]),
                # albu.RandomResizedCrop(size[0], size[1], scale=(0.8, 1.0)),
                # albu.Resize(size[0], size[1], p=1),
                albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                # ToTensorV2()
            ])
        elif data_type == 'val':
            albu_transforms = albu.Compose([
                # albu.Resize(size[0], size[1], p=1),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                # ToTensorV2()
            ])
        else:
            albu_transforms = albu.Compose([
                # albu.Resize(size[0], size[1], p=1),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                # ToTensorV2()
            ])

        return albu_transforms


# ______________________________________自定义数据增强_____________________________________________
# 添加背景
def add_background(img, target_shape, padding_number=0):
    # PIL：（宽，高） cv2：（高，宽）
    original_shape = img.shape

    new_target_shape = [max(original_shape[0], target_shape[0]), max(original_shape[1], target_shape[1])]
    if len(original_shape) == 3:
        target_img = np.zeros(shape=(new_target_shape[0], new_target_shape[1], 3), dtype=img.dtype) * padding_number
    else:
        target_img = np.zeros(shape=(new_target_shape[0], new_target_shape[1]), dtype=img.dtype) * padding_number

    # print(original_shape, target_img.shape)
    h_center = new_target_shape[0] // 2
    w_center = new_target_shape[1] // 2
    h_start = max(0, h_center - original_shape[0] // 2)
    w_start = max(0, w_center - original_shape[1] // 2)
    if len(original_shape) == 3:
        target_img[h_start: h_start + original_shape[0], w_start: w_start + original_shape[1], :] = img
    else:
        target_img[h_start: h_start + original_shape[0], w_start: w_start + original_shape[1]] = img

    return target_img


# 随机裁剪
def random_crop(img, mask=None, crop_shape=[512, 512]):
    # 如果图片小于crop shape, 则进行加背景
    if img.shape[0] < crop_shape[0] or img.shape[1] < crop_shape[1]:
        img = add_background(img=img, target_shape=crop_shape, padding_number=0)
        if mask is not None:
            mask = add_background(img=mask, target_shape=crop_shape, padding_number=0)

    # 随机裁剪 crop_shape
    original_shape = img.shape
    start_h = np.random.randint(0, original_shape[0] - crop_shape[0] + 1)
    start_w = np.random.randint(0, original_shape[1] - crop_shape[1] + 1)
    crop_img = img[start_h: start_h + crop_shape[0], start_w: start_w + crop_shape[1], :]
    if mask is not None:
        crop_mask = mask[start_h: start_h + crop_shape[0], start_w: start_w + crop_shape[1]]
        return crop_img, crop_mask
    # print('------', start_h, start_w, crop_shape[1])

    return crop_img


# 中心裁剪
def center_crop(img, mask=None, crop_shape=[512, 512]):
    if img.shape[0] < crop_shape[0] or img.shape[1] < crop_shape[1]:
        img = add_background(img=img, target_shape=crop_shape, padding_number=0)
        if mask is not None:
            mask = add_background(img=mask, target_shape=crop_shape, padding_number=0)

    center = [img.shape[0] / 2, img.shape[1] / 2]
    x = center[1] - crop_shape[1] / 2
    y = center[0] - crop_shape[0] / 2

    crop_img = img[int(y):int(y + crop_shape[0]), int(x):int(x + crop_shape[1])]
    if mask is not None:
        crop_mask = mask[int(y):int(y + crop_shape[0]), int(x):int(x + crop_shape[1])]
        return crop_img, crop_mask
    return crop_img


# 随机旋转翻转
def random_flip_and_rot90(img, mask, is_rot=True):
    # axis=0 垂直翻转, axis=1 水平翻转
    if np.random.random() > 0.5:
        img = np.flip(img, axis=0)
        mask = np.flip(mask, axis=0)
    if np.random.random() > 0.5:
        img = np.flip(img, axis=1)
        mask = np.flip(mask, axis=1)
    if np.random.random() > 0.5 and is_rot:
        rot_num = np.random.randint(1, 4)
        img = np.rot90(img, k=rot_num, axes=(0, 1))
        mask = np.rot90(mask, k=rot_num, axes=(0, 1))

    # ValueError: some of the strides of a given numpy array are negative.
    # This is currently not supported, but will be added in future releases.
    # 解决需要加入下面两行代码
    img = np.ascontiguousarray(img)
    mask = np.ascontiguousarray(mask)

    return img, mask


# 同步resize
def sync_resize(img, mask, resize_shape):
    new_img = cv2.resize(img, [resize_shape[1], resize_shape[0]])
    new_mask = cv2.resize(mask, [resize_shape[1], resize_shape[0]])

    return new_img, new_mask


# 手工特征
def get_feature(img, feature_name='hog'):
    """
    :param img: RGB图像
    :param feature_name:
    :return: 归一化特征
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img = data.camera()
    if feature_name == 'hog':
        out = feature.hog(img, orientations=6, pixels_per_cell=[8, 8], cells_per_block=[2, 2], visualize=True)[1] / 255
    elif feature_name == 'sobel':
        out = filters.sobel(img)
    else:
        out = None

    return out


# 边缘信息
def gen_edge_mask(mask, kernel_size=5):
    kernel = morphology.disk(kernel_size)
    erosion_mask = morphology.erosion(mask, kernel)
    edge_mask = erosion_mask - mask

    return edge_mask // 255


# Jpeg压缩
def jpeg_compression(img, qfs=None, p=0.5):
    if np.random.random() > p:
        return img
    if qfs is not None:
        compress_val = choice(qfs)
    else:
        qfs = [60, 70, 80, 90, 95]
        compress_val = choice(qfs)
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


# 中值模糊
def median_blur(img, mask, param=5):
    if np.random.random() > 0.5:
        if param is None:
            param = choice([3, 5, 7, 17, 21])
        new_img = cv2.medianBlur(img, param)
        return new_img, mask
    else:
        return img, mask


# 高斯模糊
def gaussian_blur(img, mask, p=1.0, param=None):
    if np.random.random() < p:
        if param is None:
            param = choice([3, 5, 7, 17, 21])
        new_img = cv2.GaussianBlur(img, (param, param), param * 1.0 / 6)
        return new_img, mask
    else:
        return img, mask


def resize(forgery, rate=None, p=0.5):
    if np.random.rand() <= p:
        ori_size = forgery.shape[:2]
        forgery = cv2.resize(forgery, (int(ori_size[1] * rate), int(ori_size[0] * rate)), cv2.INTER_AREA)
        forgery = cv2.resize(forgery, (ori_size[1], ori_size[0]), cv2.INTER_AREA)
    return forgery


def gaussian_noise(forgery, mean=0, sd=5, p=1):
    if np.random.rand() < p:
        size = forgery.shape
        noise = np.random.normal(mean, sd, size)
        # print(noise)
        forgery = forgery + noise
        forgery = np.clip(forgery, 0, 255)
    return forgery


def tamper_crop(img=None, mask=None, crop_shape=[224, 224]):
    tamper_index = np.where(mask > 0)

    tamper_y_min = np.min(np.array(tamper_index[0]))
    tamper_y_max = np.max(np.array(tamper_index[0]))
    tamper_x_min = np.min(np.array(tamper_index[1]))
    tamper_x_max = np.max(np.array(tamper_index[1]))
    return tamper_y_max - tamper_y_min, tamper_x_max - tamper_x_min


# -----------------------------------自监督数据增强------------------------------- #
# 拼接
def splicing(orig_img, orig_mask, tamper_img, tamper_mask, seamless_p=0.2, choice_box_v2=False, self_random=False):
    if len(np.unique(tamper_mask)) == 1:
        return orig_img, orig_mask

    tamper_index = np.where(tamper_mask > 0)
    if tamper_index[0].size == 0 or tamper_index[1].size == 0:
        # zero-size array to reduction operation maximum which has no identity
        return orig_img, orig_mask

    tamper_y_min = np.min(np.array(tamper_index[0]))
    tamper_y_max = np.max(np.array(tamper_index[0]))
    tamper_x_min = np.min(np.array(tamper_index[1]))
    tamper_x_max = np.max(np.array(tamper_index[1]))

    tamper_img_item = tamper_img[tamper_y_min:tamper_y_max, tamper_x_min:tamper_x_max, :]
    tamper_item_mask = tamper_mask[tamper_y_min:tamper_y_max, tamper_x_min:tamper_x_max]

    h, w = tamper_y_max - tamper_y_min, tamper_x_max - tamper_x_min
    orig_h, orig_w = orig_mask.shape

    if h >= orig_h or w >= orig_w:
        if h / orig_h > w / orig_w:
            w = int(w * (1.0 * (orig_h // 2) / h))
            h = orig_h // 2
        else:
            h = int(h * (1.0 * (orig_w // 2) / w))
            w = orig_w // 2

    if w <= 0 or h <= 0:
        return orig_img, orig_mask
    if w > 2 and h > 2 and self_random:
        scale = 0.5 + np.random.random() / 2
        h = int(h * scale)
        w = int(w * scale)
    tamper_img_item = cv2.resize(tamper_img_item, (w, h))
    tamper_item_mask = cv2.resize(tamper_item_mask, (w, h))

    rand_y_start, rand_x_start = np.random.randint(0, orig_h - h), np.random.randint(0, orig_w - w)

    # tamper_item=cv2.bitwise_and(tamper_img_item,tamper_img_item,mask=tamper_item_mask)
    tamper_full_mask = np.zeros(orig_mask.shape, dtype=np.uint8)
    tamper_full_img = np.zeros(orig_img.shape, dtype=np.uint8)

    tamper_full_img[rand_y_start:rand_y_start + h, rand_x_start:rand_x_start + w, :] = tamper_img_item
    tamper_full_mask[rand_y_start:rand_y_start + h, rand_x_start:rand_x_start + w] = tamper_item_mask

    try:
        tamper_full_img = cv2.bitwise_and(tamper_full_img, tamper_full_img, mask=tamper_full_mask)
    except:
        print(tamper_full_img.shape, tamper_full_mask.shape)
    change_mask = tamper_full_mask | orig_mask
    result_img = orig_img - cv2.bitwise_and(orig_img, orig_img, mask=tamper_full_mask) + tamper_full_img

    if random.random() < seamless_p:  # 一定的概率泊松融合
        try:
            orig_index = np.where(tamper_full_mask > 0)
            if orig_index[0].size == 0 or orig_index[1].size == 0:
                # zero-size array to reduction operation maximum which has no identity
                return result_img, change_mask

            orig_y_min = np.min(np.array(orig_index[0]))
            orig_y_max = np.max(np.array(orig_index[0]))
            orig_x_min = np.min(np.array(orig_index[1]))
            orig_x_max = np.max(np.array(orig_index[1]))
            center = ((orig_x_min + orig_x_max) // 2, (orig_y_min + orig_y_max) // 2,)
            if orig_x_max - orig_x_min > 2 and orig_y_max - orig_y_min > 2 and (orig_x_max - orig_x_min) * (
                    orig_y_max - orig_y_min) > 10:
                result_img = cv2.seamlessClone(result_img, orig_img, tamper_full_mask * 255, center, cv2.NORMAL_CLONE)
        except:
            # 可能会出现的bug:cv2.error: OpenCV(4.1.2) /io/opencv/modules/core/src/matrix_wrap.cpp:1659: error: (-215:Assertion failed) !fixedSize() in function 'release
            pass

    return result_img, change_mask


# 基于文本框的拼接
def splicing_by_textbox(orig_img, orig_mask, tamper_img, tamper_mask, tamper_boxes):
    if len(np.unique(tamper_mask)) == 1 or len(tamper_boxes) == 0:
        return orig_img, orig_mask

    box = tamper_boxes[np.random.choice(len(tamper_boxes), 1)[0]]  # 随机选一个文字区域pr
    tamper_y_min, tamper_y_max, tamper_x_min, tamper_x_max = box[1], box[1] + box[3], box[0], box[0] + box[2]

    tamper_img_item = tamper_img[tamper_y_min:tamper_y_max, tamper_x_min:tamper_x_max, :]
    tamper_item_mask = np.ones(tamper_img_item.shape[:2], dtype='uint8')

    h, w = tamper_y_max - tamper_y_min, tamper_x_max - tamper_x_min
    orig_h, orig_w = orig_mask.shape

    if h >= orig_h or w >= orig_w:
        if h / orig_h > w / orig_w:
            w = int(w * (1.0 * (orig_h // 2) / h))
            h = orig_h // 2
        else:
            h = int(h * (1.0 * (orig_w // 2) / w))
            w = orig_w // 2
    if w <= 0 or h <= 0:
        return orig_img, orig_mask

    tamper_img_item = cv2.resize(tamper_img_item, (w, h))
    tamper_item_mask = cv2.resize(tamper_item_mask, (w, h))

    rand_y_start, rand_x_start = np.random.randint(0, orig_h - h), np.random.randint(0, orig_w - w)

    orig_img[rand_y_start:rand_y_start + h, rand_x_start:rand_x_start + w, :] = tamper_img_item
    orig_mask[rand_y_start:rand_y_start + h, rand_x_start:rand_x_start + w] = tamper_item_mask

    return orig_img, orig_mask


# 随机形状的copy-move
def random_copy_move(img, mask, p=0.5):
    """
    a. Randomly crop-and-paste image region(s) of different shapes (circle, triangle, rectangle and arbitrary boundaries).
    b.Cropped image region(s) can be processed with resizing, rotation or other distortion then be pasted to generate a spliced image.
    """
    if random.random() > 7:
        return img, mask

    def circle(shape):
        h, w, c = shape
        canvas = np.zeros(shape=shape, dtype="uint8")
        radius = np.random.randint(30, high=50)
        center_x_1 = np.random.randint(radius, high=w - radius)
        center_y_1 = np.random.randint(radius, high=h - radius)

        center_x_2 = np.random.randint(radius, high=w - radius)
        center_y_2 = np.random.randint(radius, high=h - radius)

        cv2.circle(canvas, (center_x_1, center_y_1), radius, (255, 255, 255), -1)
        crop_mask = canvas // 255
        crop_mask = crop_mask[center_y_1 - radius:center_y_1 + radius, center_x_1 - radius:center_x_1 + radius, :]

        box_1 = [center_y_1 - radius, center_y_1 + radius, center_x_1 - radius, center_x_1 + radius]
        box_2 = [center_y_2 - radius, center_y_2 + radius, center_x_2 - radius, center_x_2 + radius]

        return canvas, crop_mask, box_1, box_2

    def rectangle(shape):
        h, w, c = shape
        canvas = np.zeros(shape=shape, dtype="uint8")
        # r_w = np.random.randint(30, high=60)
        # r_h = np.random.randint(30, high=60)
        # 参考Dave的
        cut_rat_w = random.random() * 0.1 + 0.05
        cut_rat_h = random.random() * 0.1 + 0.05
        r_w = int(w * cut_rat_w)
        r_h = int(h * cut_rat_h)

        start_x_1 = np.random.randint(0, high=w - r_w)
        start_y_1 = np.random.randint(0, high=h - r_h)

        start_x_2 = np.random.randint(0, high=w - r_w)
        start_y_2 = np.random.randint(0, high=h - r_h)

        cv2.rectangle(canvas, (start_x_1, start_y_1), (start_x_1 + r_w, start_y_1 + r_h), (255, 255, 255), -1)
        crop_mask = canvas // 255
        crop_mask = crop_mask[start_y_1:start_y_1 + r_h, start_x_1:start_x_1 + r_w, :]

        box_1 = [start_y_1, start_y_1 + r_h, start_x_1, start_x_1 + r_w]
        box_2 = [start_y_2, start_y_2 + r_h, start_x_2, start_x_2 + r_w]

        return canvas, crop_mask, box_1, box_2

    def triangle(shape):
        h, w, c = shape
        canvas = np.zeros(shape=shape, dtype="uint8")
        t_w = np.random.randint(30, high=60)
        t_h = np.random.randint(30, high=60)
        start_x_1 = np.random.randint(0, high=w - t_w)
        start_y_1 = np.random.randint(0, high=h - t_h)

        start_x_2 = np.random.randint(0, high=w - t_w)
        start_y_2 = np.random.randint(0, high=h - t_h)

        point_1 = [start_x_1 + t_w // 2, start_y_1]
        point_2 = [start_x_1, start_y_1 + t_h]
        point_3 = [start_x_1 + t_w, start_y_1 + t_h]

        points = np.array([point_1, point_2, point_3], np.int32)  # 多边形的顶点坐标
        # cv2.polylines(canvas, [points], True, (255, 255, 255), -1)  # 画任意多边形
        cv2.fillPoly(canvas, [points], (255, 255, 255))  # 画任意多边形
        crop_mask = canvas // 255
        crop_mask = crop_mask[start_y_1:start_y_1 + t_h, start_x_1:start_x_1 + t_w, :]

        box_1 = [start_y_1, start_y_1 + t_h, start_x_1, start_x_1 + t_w]
        box_2 = [start_y_2, start_y_2 + t_h, start_x_2, start_x_2 + t_w]

        return canvas, crop_mask, box_1, box_2

    shapes = ['circle', 'rectangle', 'triangle']  #
    # index = np.random.randint(0, len(shapes))
    index = 1
    if shapes[index] == 'circle':
        canvas, crop_mask, box_1, box_2 = circle(img.shape)
    elif shapes[index] == 'rectangle':
        canvas, crop_mask, box_1, box_2 = rectangle(img.shape)
    else:
        canvas, crop_mask, box_1, box_2 = triangle(img.shape)

    img[box_1[0]:box_1[1], box_1[2]:box_1[3], :] = (1 - crop_mask) * img[box_1[0]:box_1[1], box_1[2]:box_1[3], :] + \
                                                   crop_mask * img[box_2[0]:box_2[1], box_2[2]:box_2[3], :]

    mask = mask + cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) // 255

    return img, mask


# 复制粘贴
def copy_move_by_mask(tamper_img, tamper_mask, p=1.0, choice_box_v2=False, self_random=False):
    if random.random() > p or len(np.unique(tamper_mask)) == 1:
        return tamper_img, tamper_mask

    if not choice_box_v2:
        tamper_index = np.where(tamper_mask > 0)
        tamper_y_min = np.min(np.array(tamper_index[0]))
        tamper_y_max = np.max(np.array(tamper_index[0]))
        tamper_x_min = np.min(np.array(tamper_index[1]))
        tamper_x_max = np.max(np.array(tamper_index[1]))
    else:
        box = random_choice_box(tamper_mask)
        if box is not None:
            tamper_y_min, tamper_y_max, tamper_x_min, tamper_x_max = box
        else:
            return tamper_img, tamper_mask

    h, w = tamper_y_max - tamper_y_min, tamper_x_max - tamper_x_min
    orig_h, orig_w = tamper_mask.shape
    if h * w == 0 or (h * w) / (orig_h * orig_w) > 0.2 or orig_h - h <= 0 or orig_w - w <= 0:
        return tamper_img, tamper_mask
    rand_y_start, rand_x_start = np.random.randint(0, orig_h - h), np.random.randint(0, orig_w - w)
    # 篡改位置复制一份
    if random.random() > 0.5:
        tamper_img_item = tamper_img[tamper_y_min:tamper_y_max, tamper_x_min:tamper_x_max, :]
        tamper_item_mask = tamper_mask[tamper_y_min:tamper_y_max, tamper_x_min:tamper_x_max]
        if self_random:
            method = [0, 1]
            num = random.randint(1, 2)
            index = random.sample(method, num)
            for i in index:
                if i == 0:  # 使用随机crop
                    crop_h = np.random.randint(0, tamper_y_max - tamper_y_min)
                    crop_w = np.random.randint(0, tamper_x_max - tamper_x_min)
                    tamper_full_mask = np.zeros(tamper_mask.shape, dtype=np.uint8)
                    tamper_full_img = np.zeros(tamper_img.shape, dtype=np.uint8)

                    tamper_full_img[rand_y_start:rand_y_start + crop_h, rand_x_start:rand_x_start + crop_w,
                    :] = tamper_img_item[:crop_h, :crop_w]
                    tamper_full_mask[rand_y_start:rand_y_start + crop_h,
                    rand_x_start:rand_x_start + crop_w] = tamper_item_mask[:crop_h, :crop_w]

                    tamper_full_img = cv2.bitwise_and(tamper_full_img, tamper_full_img, mask=tamper_full_mask)

                    change_mask = tamper_full_mask | tamper_mask
                    result_img = tamper_img - cv2.bitwise_and(tamper_img, tamper_img,
                                                              mask=tamper_full_mask) + tamper_full_img
                if i == 1:  # 使用随机resize
                    scale = 0.5 + np.random.random() / 2
                    if h > 2 and w > 2:
                        h = int(scale * h)
                        w = int(scale * w)
                    tamper_full_mask = np.zeros(tamper_mask.shape, dtype=np.uint8)
                    tamper_full_img = np.zeros(tamper_img.shape, dtype=np.uint8)

                    tamper_full_img[rand_y_start:rand_y_start + h, rand_x_start:rand_x_start + w, :] = cv2.resize(
                        tamper_img_item, (w, h))
                    tamper_full_mask[rand_y_start:rand_y_start + h, rand_x_start:rand_x_start + w] = cv2.resize(
                        tamper_item_mask, (w, h))

                    tamper_full_img = cv2.bitwise_and(tamper_full_img, tamper_full_img, mask=tamper_full_mask)

                    change_mask = tamper_full_mask | tamper_mask
                    result_img = tamper_img - cv2.bitwise_and(tamper_img, tamper_img,
                                                              mask=tamper_full_mask) + tamper_full_img
        ########################原有的#################
        # if is_crop:
        #     h = np.random.randint(32, tamper_y_max - tamper_y_min)
        #     w = np.random.randint(32, tamper_x_max - tamper_x_min)
        #     tamper_full_mask = np.zeros(tamper_mask.shape, dtype=np.uint8)
        #     tamper_full_img = np.zeros(tamper_img.shape, dtype=np.uint8)
        #
        #     tamper_full_img[rand_y_start:rand_y_start + h, rand_x_start:rand_x_start + w, :] = tamper_img_item[:h, :w]
        #     tamper_full_mask[rand_y_start:rand_y_start + h, rand_x_start:rand_x_start + w] = tamper_item_mask[:h, :w]
        #
        #     tamper_full_img = cv2.bitwise_and(tamper_full_img, tamper_full_img, mask=tamper_full_mask)
        #
        #     change_mask = tamper_full_mask | tamper_mask
        #     result_img = tamper_img - cv2.bitwise_and(tamper_img, tamper_img, mask=tamper_full_mask) + tamper_full_img
        # if is_resize:
        #     scale = 0.5 + np.random.random() / 2
        #     h = int(scale * h)
        #     w = int(scale * w)
        #     tamper_full_mask = np.zeros(tamper_mask.shape, dtype=np.uint8)
        #     tamper_full_img = np.zeros(tamper_img.shape, dtype=np.uint8)
        #
        #     tamper_full_img[rand_y_start:rand_y_start + h, rand_x_start:rand_x_start + w, :] = cv2.resize(
        #         tamper_img_item, (w, h))
        #     tamper_full_mask[rand_y_start:rand_y_start + h, rand_x_start:rand_x_start + w] = cv2.resize(
        #         tamper_item_mask, (w, h))
        #
        #     tamper_full_img = cv2.bitwise_and(tamper_full_img, tamper_full_img, mask=tamper_full_mask)
        #
        #     change_mask = tamper_full_mask | tamper_mask
        #     result_img = tamper_img - cv2.bitwise_and(tamper_img, tamper_img, mask=tamper_full_mask) + tamper_full_img
        #
        else:
            tamper_full_mask = np.zeros(tamper_mask.shape, dtype=np.uint8)
            tamper_full_img = np.zeros(tamper_img.shape, dtype=np.uint8)

            tamper_full_img[rand_y_start:rand_y_start + h, rand_x_start:rand_x_start + w, :] = tamper_img_item
            tamper_full_mask[rand_y_start:rand_y_start + h, rand_x_start:rand_x_start + w] = tamper_item_mask

            try:
                tamper_full_img = cv2.bitwise_and(tamper_full_img, tamper_full_img, mask=tamper_full_mask)
            except:
                print(tamper_full_img.shape, tamper_full_mask.shape)
            change_mask = tamper_full_mask | tamper_mask
            result_img = tamper_img - cv2.bitwise_and(tamper_img, tamper_img, mask=tamper_full_mask) + tamper_full_img
        return result_img, change_mask
    else:
        tamper_img_org_item = tamper_img[tamper_y_min:tamper_y_max, tamper_x_min:tamper_x_max, :]
        tamper_img_item = tamper_img[rand_y_start:rand_y_start + h, rand_x_start:rand_x_start + w, :]
        tamper_item_mask = tamper_mask[tamper_y_min:tamper_y_max, tamper_x_min:tamper_x_max]

        tamper_img[tamper_y_min:tamper_y_max, tamper_x_min:tamper_x_max, :] = \
            cv2.bitwise_and(tamper_img_org_item, tamper_img_org_item, mask=(1 - tamper_item_mask)) + \
            cv2.bitwise_and(tamper_img_item, tamper_img_item, mask=tamper_item_mask)

        return tamper_img, tamper_mask


# 基于文本框的复制粘贴
def copy_move_by_textbox(tamper_img, tamper_mask, boxes, p=1.0):
    if random.random() > p or len(np.unique(tamper_mask)) == 1 or len(boxes) == 0:
        return tamper_img, tamper_mask

    box = boxes[np.random.choice(len(boxes), 1)[0]]  # 随机选一个文字区域pr
    tamper_y_min, tamper_y_max, tamper_x_min, tamper_x_max = box[1], box[1] + box[3], box[0], box[0] + box[2]

    h, w = tamper_y_max - tamper_y_min, tamper_x_max - tamper_x_min
    orig_h, orig_w = tamper_mask.shape
    if h * w == 0 or (h * w) / (orig_h * orig_w) > 0.2 or orig_h - h <= 0 or orig_w - w <= 0:
        return tamper_img, tamper_mask
    rand_y_start, rand_x_start = np.random.randint(0, orig_h - h), np.random.randint(0, orig_w - w)
    image_new = tamper_img.copy()
    mask_new = np.zeros(tamper_mask.shape[:2], dtype='uint8')
    # 篡改位置复制一份
    if random.random() > 0.5:
        image_new[rand_y_start:rand_y_start + h, rand_x_start:rand_x_start + w, :] = \
            tamper_img[tamper_y_min:tamper_y_max, tamper_x_min:tamper_x_max, :]
        mask_new[rand_y_start:rand_y_start + h, rand_x_start:rand_x_start + w] = 1
    else:
        image_new[tamper_y_min:tamper_y_max, tamper_x_min:tamper_x_max, :] = \
            tamper_img[rand_y_start:rand_y_start + h, rand_x_start:rand_x_start + w, :]
        mask_new[tamper_y_min:tamper_y_max, tamper_x_min:tamper_x_max] = 1

    mask_new = np.where(mask_new == 1, mask_new, tamper_mask)  # 新的mask

    return image_new, mask_new


# 修复
def inpainting(img, mask, p=0.2):
    if np.random.random() > p:
        return img
    result = cv2.inpaint(img, mask, 20, cv2.INPAINT_TELEA)

    return result


# 修复: 先对图像resize到一个小的尺寸inpainting,然后再放大回去，增加速度
def inpainting_v2(img, mask, p=1.0, in_size=64):
    if np.random.random() > p:
        return img
    img_rs, mask_rs = sync_resize(img, mask, resize_shape=[in_size, in_size])
    result_rs = cv2.inpaint(img_rs, mask_rs, 20, cv2.INPAINT_TELEA)
    # result_rs = cv2.inpaint(img_rs, mask_rs, 20, cv2.INPAINT_NS)
    result = cv2.resize(result_rs, (img.shape[1], img.shape[0]))
    # result = img * (1-mask) + result * mask
    result = cv2.bitwise_and(img, img, mask=(1 - mask)) + cv2.bitwise_and(result, result, mask=mask)

    return result


# 基于文本框的数据增强
def inpainting_by_textbox(img, mask, boxes, p=1.0, in_size=64):
    if np.random.random() > p or len(boxes) == 0:
        return img, mask

    box = boxes[np.random.choice(len(boxes), 1)[0]]  # 随机选一个文字区域pr
    tamper_y_min, tamper_y_max, tamper_x_min, tamper_x_max = box[1], box[1] + box[3], box[0], box[0] + box[2]

    mask_new = np.zeros(mask.shape[:2], dtype='uint8')
    mask_new[tamper_y_min:tamper_y_max, tamper_x_min:tamper_x_max] = 1
    img_new = inpainting_v2(img, mask_new, p=1.0, in_size=in_size)

    mask_new = np.where(mask_new == 1, mask_new, mask)  # 新的mask

    return img_new, mask_new


# 随机插入文字，随机文字（包括人名，公司名，地址，电话和日期），随机颜色，随机大小，随机字体和随机位置
def random_insert_text(image, return_bbox=False, return_mask=False, show_loc=False, color=None):
    """
    在图像上添加中文字符
    :param image:  图像, BGR 图像
    :param return_bbox: 返回插入文字所在的位置， bbox=[x, y, w, h]
    :param show_loc: 显示插入文字的位置，用矩形框起来
    :return:
    """

    def random_gen_text():
        # 随机生成文字， 参考博客 https://blog.csdn.net/weixin_43566535/article/details/107339159
        # csdn[标题：Python中第三方库-Faker；作者：南枝向暖北枝寒MA]
        fake = Faker("zh_CN")
        # 生成句子
        # article = fake.text(max_nb_chars=15)  # 生成一篇文章,数字代表文章最大长度，一般小于这个数
        # article1 = re.findall('[\u4e00-\u9fa5]{0,}', article)
        # article = ''.join(article1)
        # print(f'article:{article}', fake.name(), fake.company(), fake.address()[:-7], fake.phone_number(), fake.date())
        text_per_wight = 1  # 占位：中文1，英文0.5
        index = random.randint(0, 4)
        if index == 0:
            text = fake.name()
        elif index == 1:
            text = fake.company()
        elif index == 2:
            text = fake.address()[:-7]
        elif index == 3:
            text = fake.phone_number()
        else:
            text = fake.date()
        # texts = [fake.name(), fake.company(), fake.address()[:-7], fake.phone_number(), fake.date()]
        # text = random.choice(texts)

        return text

    # 生成随机参数
    text = random_gen_text()  # 随机生成文字， gbk编码
    if color is None:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # 随机选择颜色
    else:
        color = color  # 直插入指定的颜色
    h, w, c = image.shape
    font_size = random.choice([5 * i for i in range(2, 15)])  # 10, 15, ..., 65, 70
    font_size = min([font_size, h // 2, w // len(text)])  # 随机选择字体大小
    # text_w, text_h = font_size*len(text), font_size
    bytes_len = len(text.encode('gbk'))  # 统计占位长度，中文和字母数字占位不一样
    text_w, text_h = int(font_size * 0.5 * bytes_len), font_size
    loc = (random.randint(0, w - text_w), random.randint(0, h - text_h))  # 随机选择位置
    # 生成插入文字的图像
    new_image = put_chinese_on_img(image, text, loc, color, font_size)
    # print(text, color, font_size, loc, bytes_len)
    # 框住文字
    x1, y1, x2, y2 = loc[0], loc[1], loc[0] + text_w, loc[1] + text_h
    if show_loc:
        print(text, color, font_size, loc, bytes_len)
        cv2.rectangle(new_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
    if return_bbox and return_mask:
        bbox = [x1, y1, loc[0] + text_w, text_h]
        mask = new_image - image
        mask = mask[:, :, 0] + mask[:, :, 1] + mask[:, :, 2]
        mask[mask > 1] = 1

        return new_image, mask, bbox
    elif return_bbox:
        bbox = [x1, y1, loc[0] + text_w, text_h]
        return new_image, bbox
    elif return_mask:
        mask = new_image - image
        mask = mask[:, :, 0] + mask[:, :, 1] + mask[:, :, 2]
        mask[mask > 1] = 1

        return new_image, mask

    return new_image


# 插入中文到图像中
def put_chinese_on_img(img, text, loc, color=(255, 0, 0), font_size=40, font_file=None):
    """
    在图像上添加中文字符
    :param img:  图像， RGB
    :param text: 文本
    :param loc: 位置
    :param color:  颜色
    :param font_size: 字体大小
    :param font_file: 字体文件
    :return:
    """
    pilimg = Image.fromarray(img)

    # PIL图片上打印汉字
    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    if font_file is None:
        current_work_dir = os.path.dirname(__file__)  # 当前文件所在的目录
        # 随机选择字体
        font_files = os.listdir(os.path.join(current_work_dir, 'fonts'))
        font_files = [os.path.join(current_work_dir, 'fonts', font_file) for font_file in font_files]
        font_file = random.choice(font_files)
    font = ImageFont.truetype(font_file, font_size, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
    draw.text(loc, text, color, font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体

    # PIL图片转cv2 图片
    cv2charimg = np.array(pilimg)

    return cv2charimg


if __name__ == '__main__':
    # mask = cv2.imread('/data2/hulh/dataset/Columbia/columbia_mask/canong3_canonxt_sub_01_edgemask.jpg', 0)
    # H, W = mask.shape
    # n, m = math.ceil(H / 224), math.ceil(W / 224)
    # num = 0
    # for i in range(n):
    #     for j in range(m):
    #         if 224 * (i + 1) < H and 224 * (j + 1) < W:
    #             patch_img = mask[i * 224:224 * (i + 1), j * 224:224 * (j + 1)]
    #             num += 1
    #         elif 224 * (i + 1) >= H and 224 * (j + 1) >= W:
    #             patch_img = mask[-224:, -224:]
    #             num += 1
    #         elif 224 * (i + 1) >= H:
    #             patch_img = mask[-224:, j * 224:224 * (j + 1)]
    #             num += 1
    #         elif 224 * (j + 1) >= W:
    #             patch_img = mask[i * 224:224 * (i + 1), -224:]
    #             num += 1
    #         cv2.imwrite(f'/data2/hulh/workplace/ForgeryDemo_neat/data/samples/patches/patch_{num:02d}.png', patch_img)
    #         print(num)
    # H, W = 568, 757
    # mask = np.zeros([H, W])
    # n, m = math.ceil(H / 224), math.ceil(W / 224)
    # for root, dirs, files in os.walk('/data2/hulh/workplace/ForgeryDemo_neat/data/samples/patches'):
    #     files = sorted(files)
    #     num = 0
    #     for i in range(n):
    #         for j in range(m):
    #             temp = cv2.imread(os.path.join(root, files[num]), 0) // 128
    #             if 224 * (i + 1) < H and 224 * (j + 1) < W:
    #                 mask[i * 224:224 * (i + 1), j * 224:224 * (j + 1)] += temp
    #             elif 224 * (i + 1) >= H and 224 * (j + 1) >= W:
    #                 mask[-224:, -224:] += temp
    #             elif 224 * (i + 1) >= H:
    #                 mask[-224:, j * 224:224 * (j + 1)] += temp
    #             elif 224 * (j + 1) >= W:
    #                 mask[i * 224:224 * (i + 1), -224:] += temp
    #             num += 1
    #
    # mask[mask > 0.5] = 1
    # cv2.imwrite('test.png', mask.astype(np.uint8) * 255)
    for root, dirs, files in os.walk('/data2/hulh/dataset/CASIA/CASIA2_crop_224/CASIA2gt'):
        num = 0
        for file in files:
            mask = cv2.imread(os.path.join(root, file))
            if len(np.unique(mask)) == 1:
                num += 1
                print(file)
        print(num)
