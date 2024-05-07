import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import numpy as np
from PIL import Image
from torchvision.transforms import ColorJitter

from . import flow_transforms


class FlowAugmentor:
    def __init__(
        self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True, pwc_aug=False
    ):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5 / 3.14
        )
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5
        self.pwc_aug = pwc_aug
        if self.pwc_aug:
            print("[Using pwc-style spatial augmentation]")

    def color_transform(self, img1, img2):
        """Photometric augmentation"""

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(
                self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8
            )
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """Occlusion augmentation"""

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0 : y0 + dy, x0 : x0 + dx, :] = mean_color

        return img1, img2

    def spatial_transform(self, img1, img2, flow):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), (self.crop_size[1] + 8) / float(wd)
        )

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(
                img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR
            )
            img2 = cv2.resize(
                img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR
            )
            flow = cv2.resize(
                flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR
            )
            flow = flow * [scale_x, scale_y]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.v_flip_prob:  # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        if img1.shape[0] == self.crop_size[0]:
            y0 = 0
        else:
            y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        if img1.shape[1] == self.crop_size[1]:
            x0 = 0
        else:
            x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
        img2 = img2[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
        flow = flow[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]

        return img1, img2, flow

    def __call__(self, img1, img2, flow):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        if self.pwc_aug:
            th, tw = self.crop_size
            schedule = [0.5, 1.0]  # initial coeff, final_coeff, half life
            difficulty = np.random.uniform(0, 1)
            schedule_coeff = schedule[0] + (schedule[1] - schedule[0]) * (
                2 / (1 + np.exp(-1.0986 * difficulty)) - 1
            )
            spatial_augmentor = flow_transforms.SpatialAug(
                [th, tw],
                scale=[0.4, 0.03, 0.2],
                rot=[0.4, 0.03],
                trans=[0.4, 0.03],
                squeeze=[0.3, 0.0],
                schedule_coeff=schedule_coeff,
                order=1,
                black=False,
            )
            flow = np.concatenate(
                [flow, np.ones((flow.shape[0], flow.shape[1], 1))], axis=-1
            )
            augmented, flow_valid = spatial_augmentor([img1, img2], flow)
            flow = flow_valid[:, :, :2]
            img1 = augmented[0]
            img2 = augmented[1]

        else:
            img1, img2, flow = self.spatial_transform(img1, img2, flow)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)

        return img1, img2, flow


class SparseFlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3 / 3.14
        )
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(
            self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8
        )
        img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2):
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0 : y0 + dy, x0 : x0 + dx, :] = mean_color

        return img1, img2

    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid >= 1]
        flow0 = flow[valid >= 1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:, 0]).astype(np.int32)
        yy = np.round(coords1[:, 1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def spatial_transform(self, img1, img2, flow, valid):
        pad_t = 0
        pad_b = 0
        pad_l = 0
        pad_r = 0
        if self.crop_size[0] > img1.shape[0]:
            # pad_t = self.crop_size[0] - img1.shape[0]
            pad_b = self.crop_size[0] - img1.shape[0]
        if self.crop_size[1] > img1.shape[1]:
            print("[In kitti data, padding along width axis now!]")
            pad_r = self.crop_size[1] - img1.shape[1]
        if pad_b != 0 or pad_r != 0 or pad_t != 0:
            img1 = np.pad(
                img1,
                ((pad_t, pad_b), (pad_l, pad_r), (0, 0)),
                "constant",
                constant_values=((0, 0), (0, 0), (0, 0)),
            )
            img2 = np.pad(
                img2,
                ((pad_t, pad_b), (pad_l, pad_r), (0, 0)),
                "constant",
                constant_values=((0, 0), (0, 0), (0, 0)),
            )
            flow = np.pad(
                flow,
                ((pad_t, pad_b), (pad_l, pad_r), (0, 0)),
                "constant",
                constant_values=((0, 0), (0, 0), (0, 0)),
            )
            valid = np.pad(
                valid,
                ((pad_t, pad_b), (pad_l, pad_r)),
                "constant",
                constant_values=((0, 0), (0, 0)),
            )
        # randomly sample scale

        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht), (self.crop_size[1] + 1) / float(wd)
        )

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(
                img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR
            )
            img2 = cv2.resize(
                img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR
            )
            flow, valid = self.resize_sparse_flow_map(
                flow, valid, fx=scale_x, fy=scale_y
            )

        if self.do_flip:
            if np.random.rand() < 0.5:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
                valid = valid[:, ::-1]

        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
        img2 = img2[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
        flow = flow[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
        valid = valid[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
        return img1, img2, flow, valid

    def __call__(self, img1, img2, flow, valid):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow, valid = self.spatial_transform(img1, img2, flow, valid)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)

        return img1, img2, flow, valid


class ImageAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5 / 3.14
        )
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        """Photometric augmentation"""

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(
                self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8
            )
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def spatial_transform(self, img1, img2):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), (self.crop_size[1] + 8) / float(wd)
        )

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(
                img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR
            )
            img2 = cv2.resize(
                img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR
            )

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]

            if np.random.rand() < self.v_flip_prob:  # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]

        if img1.shape[0] == self.crop_size[0]:
            y0 = 0
        else:
            y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        if img1.shape[1] == self.crop_size[1]:
            x0 = 0
        else:
            x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
        img2 = img2[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]

        return img1, img2

    def __call__(self, img1, img2):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.spatial_transform(img1, img2)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)

        return img1, img2
