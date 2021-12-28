""" Transforms Factory
Factory methods for building image transforms for use with TIMM (PyTorch Image Models)

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
import random
import torch
from torchvision import transforms
import numpy
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT
from timm.data.auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform
from timm.data.transforms import _pil_interp, RandomResizedCropAndInterpolation, ToNumpy, ToTensor
from timm.data.random_erasing import RandomErasing
from PIL import Image

#自定義一個旋轉固定角度的transform
class rotTransforms(object):
    def __call__(self, img):  
        #angle = random.choice([-90,0,90])
        angle = random.choice(list(range(-45,45,3))+[0]*25)
        img = transforms.functional.rotate(img, angle)
        return img

#自定義一個專為中文字的透視變換的transform
class chinese_perTransforms(object):
        
    #PIL透視變換用
    def find_coeffs(self,source_coords, target_coords):
        matrix = []
        for s, t in zip(source_coords, target_coords):
            matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0]*t[0], -s[0]*t[1]])
            matrix.append([0, 0, 0, t[0], t[1], 1, -s[1]*t[0], -s[1]*t[1]])
        A = numpy.matrix(matrix, dtype=numpy.float)
        B = numpy.array(source_coords).reshape(8)
        res = numpy.dot(numpy.linalg.inv(A.T * A) * A.T, B)
        return numpy.array(res).reshape(8)
    
    def __call__(self, img):  
        width, height = img.size
        #梯形、平行四邊形、不變
        #[左上，右上，右下，左下]

        shape=random.choice([[(0, 0), (width-1, 0), (int(5*(width-1)/6), height-1), (int((width-1)/6), height-1)],
        [(int((width-1)/6), 0), (int(5*(width-1)/6), 0),  ((width-1), height-1),(0, height-1)],
        [(0, int((height-1)/6)), ((width-1), 0), ((width-1), height-1), (0, int(5*(height-1)/6))],
        [(0, 0), ((width-1), int((height-1)/6)), ((width-1),int(5*(height-1)/6)), (0, height-1)],
        #左高右低平行四邊形
        [(0, 0), ((width-1), int((height-1)/5)),(width-1, height-1), (0, int(4*(height-1)/5))],
        [(int((width-1)/5), 0), (int(4*(width-1)/5), int((height-1)/5)),(int(4*(width-1)/5), (height-1)), (int((width-1)/5), int(4*(height-1)/5))],
        [(int((width-1)/3), 0), (int(2*(width-1)/3), int((height-1)/3)),(int(2*(width-1)/3), (height-1)), (int((width-1)/3), int(2*(height-1)/3))],
        #右高左低平行四邊形
        [(0, int((height-1)/5)), ((width-1), 0),(width-1, int(4*(height-1)/5)), (0, height-1)],
        [(int((width-1)/5), int((height-1)/5)), (int(4*(width-1)/5), 0),(int(4*(width-1)/5), int(4*(height-1)/5)), (int((width-1)/5), (height-1))],
        [(int((width-1)/3), int((height-1)/3)), (int(2*(width-1)/3), 0),(int(2*(width-1)/3), int(2*(height-1)/3)), (int((width-1)/3), (height-1))],
        #原始形狀
        [(0, 0), (width-1, 0), (width-1, height-1), (0, height-1)],
        [(0, 0), (width-1, 0), (width-1, height-1), (0, height-1)],
        #[(0, 0), (width-1, 0), (width-1, height-1), (0, height-1)],
        ])

        try:
            coeffs = self.find_coeffs([(0, 0), (width-1, 0), (width-1, height-1), (0, height-1)],shape)
            img=img.transform(img.size, Image.PERSPECTIVE, coeffs,
                        Image.BICUBIC)
        except:
            img.show()
            return img
        else:
            return img

#自定義一個透視變換的transform
class perTransforms(object):
        
    #PIL透視變換用
    def find_coeffs(self,source_coords, target_coords):
        matrix = []
        for s, t in zip(source_coords, target_coords):
            matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0]*t[0], -s[0]*t[1]])
            matrix.append([0, 0, 0, t[0], t[1], 1, -s[1]*t[0], -s[1]*t[1]])
        A = numpy.matrix(matrix, dtype=numpy.float)
        B = numpy.array(source_coords).reshape(8)
        res = numpy.dot(numpy.linalg.inv(A.T * A) * A.T, B)
        return numpy.array(res).reshape(8)
    
    def __call__(self, img):  
        width, height = img.size
        #梯形、平行四邊形
        #[左上，右上，右下，左下]

        shape=random.choice([[(0, 0), (width-1, 0), (int(5*(width-1)/6), height-1), (int((width-1)/6), height-1)],
        [(int((width-1)/6), 0), (int(5*(width-1)/6), 0),  ((width-1), height-1),(0, height-1)],
        [(0, int((height-1)/6)), ((width-1), 0), ((width-1), height-1), (0, int(5*(height-1)/6))],
        [(0, 0), ((width-1), int((height-1)/6)), ((width-1),int(5*(height-1)/6)), (0, height-1)],
        [(0, 0), (width-1, 0), (width-1, height-1), (0, height-1)]
        ])
        try:
            coeffs = self.find_coeffs([(0, 0), (width-1, 0), (width-1, height-1), (0, height-1)],shape)
            img=img.transform(img.size, Image.PERSPECTIVE, coeffs,
                        Image.BICUBIC)
        except:
            img.show()
            return img
        else:
            return img

def transforms_noaug_train(
        img_size=224,
        interpolation='bilinear',
        use_prefetcher=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
):
    if interpolation == 'random':
        # random interpolation not supported with no-aug
        interpolation = 'bilinear'
    tfl = [
        transforms.Resize(img_size, _pil_interp(interpolation)),
        transforms.CenterCrop(img_size)
    ]
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        tfl += [ToNumpy()]
    else:
        tfl += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ]
    return transforms.Compose(tfl)


def transforms_imagenet_train(
        img_size=224,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        interpolation='random',
        use_prefetcher=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        separate=False,
):
    """
    If separate==True, the transforms are returned as a tuple of 3 separate transforms
    for use in a mixing dataset that passes
     * all data through the first (primary) transform, called the 'clean' data
     * a portion of the data through the secondary transform
     * normalizes and converts the branches above with the third, final transform
    """
    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range (0.08，1.0)
    ratio = tuple(ratio or (3./4., 4./3.))  # default imagenet ratio range
    #隨機連續旋轉角度[-90,90]
    #[transforms.RandomRotation(90)]
    #自定義旋轉固定角度之[rotTransforms()]
    #自定義透視變換之[perTransforms()]
    #自定義中文字透視變換之[chinese_perTransforms()]
    
    primary_tfl =[chinese_perTransforms()]+[rotTransforms()]+[RandomResizedCropAndInterpolation(img_size, scale= (0.75, 1.0), ratio=ratio, interpolation=interpolation)]
    
    #針對中文字元
    hflip=0
    
    if hflip > 0.:
        primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.:
        primary_tfl += [transforms.RandomVerticalFlip(p=vflip)]

    secondary_tfl = []
    if auto_augment:
        assert isinstance(auto_augment, str)
        if isinstance(img_size, (tuple, list)):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != 'random':
            aa_params['interpolation'] = _pil_interp(interpolation)
        if auto_augment.startswith('rand'):
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
        elif auto_augment.startswith('augmix'):
            aa_params['translate_pct'] = 0.3
            secondary_tfl += [augment_and_mix_transform(auto_augment, aa_params)]
        else:
            secondary_tfl += [auto_augment_transform(auto_augment, aa_params)]
    elif color_jitter is not None:
        # color jitter is enabled when not using AA
        if isinstance(color_jitter, (list, tuple)):
            # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
            # or 4 if also augmenting hue
            assert len(color_jitter) in (3, 4)
        else:
            # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
            color_jitter = (float(color_jitter),) * 3
        #原code
        #secondary_tfl += [transforms.ColorJitter(*color_jitter)]
        #0919測試
        secondary_tfl += [transforms.ColorJitter(0.4,0.4,0.4,hue=0.2)]

    final_tfl = []
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        final_tfl += [ToNumpy()]
    else:
        final_tfl += [
            transforms.ToTensor(),
            transforms.Normalize(
             mean=torch.tensor(mean),
             std=torch.tensor(std))
                
        ]
        if re_prob > 0.:
            final_tfl.append(
                RandomErasing(re_prob, mode=re_mode, max_count=re_count, num_splits=re_num_splits, device='cpu'))

    if separate:
        return transforms.Compose(primary_tfl), transforms.Compose(secondary_tfl), transforms.Compose(final_tfl)
    else:
        return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)


def transforms_imagenet_eval(
        img_size=224,
        crop_pct=None,
        interpolation='bilinear',
        use_prefetcher=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD):
    crop_pct = crop_pct or DEFAULT_CROP_PCT

    if isinstance(img_size, (tuple, list)):
        assert len(img_size) == 2
        if img_size[-1] == img_size[-2]:
            # fall-back to older behaviour so Resize scales to shortest edge if target is square
            scale_size = int(math.floor(img_size[0] / crop_pct))
        else:
            scale_size = tuple([int(x / crop_pct) for x in img_size])
    else:
        scale_size = int(math.floor(img_size / crop_pct))

    tfl = [
        #transforms.Resize(scale_size, _pil_interp(interpolation)),
        #防止漏掉劑量數字
        transforms.Resize([scale_size,scale_size], _pil_interp(interpolation)),
        transforms.CenterCrop(img_size),
    ]
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        tfl += [ToNumpy()]
    else:
        tfl += [
            transforms.ToTensor(),
            transforms.Normalize(
                     mean=torch.tensor(mean),
                     std=torch.tensor(std))
        ]

    return transforms.Compose(tfl)


def create_transform(
        input_size,
        is_training=False,
        use_prefetcher=False,
        no_aug=False,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        crop_pct=None,
        tf_preprocessing=False,
        separate=False):

    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    if tf_preprocessing and use_prefetcher:
        assert not separate, "Separate transforms not supported for TF preprocessing"
        from timm.data.tf_preprocessing import TfPreprocessTransform
        transform = TfPreprocessTransform(
            is_training=is_training, size=img_size, interpolation=interpolation)
    else:
        if is_training and no_aug:
            assert not separate, "Cannot perform split augmentation with no_aug"
            transform = transforms_noaug_train(
                img_size,
                interpolation=interpolation,
                use_prefetcher=use_prefetcher,
                mean=mean,
                std=std)
        elif is_training:
            transform = transforms_imagenet_train(
                img_size,
                scale=scale,
                ratio=ratio,
                hflip=hflip,
                vflip=vflip,
                color_jitter=color_jitter,
                auto_augment=auto_augment,
                interpolation=interpolation,
                use_prefetcher=use_prefetcher,
                mean=mean,
                std=std,
                re_prob=re_prob,
                re_mode=re_mode,
                re_count=re_count,
                re_num_splits=re_num_splits,
                separate=separate)
        else:
            assert not separate, "Separate transforms not supported for validation preprocessing"
            transform = transforms_imagenet_eval(
                img_size,
                interpolation=interpolation,
                use_prefetcher=use_prefetcher,
                mean=mean,
                std=std,
                crop_pct=crop_pct)

    return transform
