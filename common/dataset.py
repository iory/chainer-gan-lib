import os

import numpy as np
from PIL import Image
import chainer
from chainer.dataset import dataset_mixin


class Cifar10Dataset(dataset_mixin.DatasetMixin):
    def __init__(self, test=False):
        d_train, d_test = chainer.datasets.get_cifar10(ndim=3, withlabel=False, scale=1.0)
        if test:
            self.ims = d_test
        else:
            self.ims = d_train
        self.ims = self.ims * 2 - 1.0  # [-1.0, 1.0]
        print("load cifar-10.  shape: ", self.ims.shape)

    def __len__(self):
        return self.ims.shape[0]

    def get_example(self, i):
        return self.ims[i]


def image_to_np(img):
    img = img.convert('RGB')
    img = np.asarray(img, dtype=np.uint8)
    img = img.transpose((2, 0, 1)).astype("f")
    if img.shape[0] == 1:
        img = np.broadcast_to(img, (3, img.shape[1], img.shape[2]))
    img = (img - 127.5)/127.5
    return img


def preprocess_image(img, crop_width=256, img2np=True):
    wid = min(img.size[0], img.size[1])
    ratio = crop_width / wid + 1e-4
    img = img.resize((int(ratio * img.size[0]), int(ratio * img.size[1])), Image.BILINEAR)
    x_l = (img.size[0]) // 2 - crop_width // 2
    x_r = x_l + crop_width
    y_u = 0
    y_d = y_u + crop_width
    img = img.crop((x_l, y_u, x_r, y_d))

    if img2np:
        img = image_to_np(img)
    return img


def find_all_files(directory):
    """http://qiita.com/suin/items/cdef17e447ceeff6e79d"""
    for root, dirs, files in os.walk(directory):
        yield root
        for file in files:
            yield os.path.join(root, file)


class ImagenetDataset(dataset_mixin.DatasetMixin):
    def __init__(self, file_list, crop_width=256):
        self.crop_width = crop_width
        self.image_files = file_list
        print(len(self.image_files))

    def __len__(self):
        return len(self.image_files)

    def get_example(self, i):
        np.random.seed()
        img = None

        while img is None:
            # print(i,id)
            try:
                fn = "%s" % (self.image_files[i])
                img = Image.open(fn)
            except Exception as e:
                print(i, fn, str(e))
        return preprocess_image(img, crop_width=self.crop_width)


class CelebA(chainer.dataset.DatasetMixin):
    def __init__(self, dataset_home='/home/iory/dataset', image_size=64, image_type='sigmoid', nodivide=False, type='train'):
        self.image_type = image_type
        self.nodivide = nodivide
        self.name = 'celeba'
        self.n_imgs = 202599
        self.n_attrs = 40
        self.image_size = image_size
        self.data_dir = os.path.join(dataset_home, self.name)
        self._npz_path = os.path.join(self.data_dir, self.name + '.npz')
        self.img_dir = os.path.join(self.data_dir, 'img_align_celeba')

    def __len__(self):
        return self.n_imgs

    def get_image(self, idx):
        img_path = os.path.join(self.img_dir, '%.6d.jpg' % (idx + 1))
        return Image.open(img_path)

    def get_attributes(self, idx):
        return None

    def get_example(self, i):
        image = self.get_image(i)
        attr = self.get_attributes(i)

        offset_x = np.random.randint(8) + 13
        offset_y = np.random.randint(8) + 33
        w = 144
        h = 144
        image = np.asarray(image.convert('RGB').
                           crop((offset_x, offset_y, offset_x + w, offset_y + h)).
                           resize((self.image_size, self.image_size)))

        image = image.astype(np.float32).transpose((2, 0, 1))

        # pre-process
        if not self.nodivide:
            if self.image_type == 'tanh':
                image = image / 127.5 - 1
            elif self.image_type == 'sigmoid':
                image /= 255.
            else:
                raise ValueError('invalid image type')

        return image
