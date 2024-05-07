import itertools
import json
import os
import os.path as osp
import random
from glob import glob
from pathlib import Path

import numpy as np
import parse
import torch
import torch.utils.data as data
from utils import frame_utils
from utils.augmentor import AmodalFlowAugmentor, FlowAugmentor, SparseFlowAugmentor

from core.utils.labels import ID_MAP_AMODAL as AMSYNTHDRIVE_ID_MAP_AMODAL
from core.utils.labels import ID_MAP_FULL as AMSYNTHDRIVE_ID_MAP_FULL


class FlowDataset(data.Dataset):
    def __init__(
        self,
        aug_params=None,
        sparse=False,
        amodal=False,
        amodal_layers=10,
        show_extra_info=False,
    ):
        super().__init__()

        self.augmentor = None
        self.sparse = sparse
        self.amodal = amodal
        self.amodal_layers = amodal_layers
        self.show_extra_info = show_extra_info

        self.augmentor = None
        if aug_params is not None:
            if amodal:
                self.augmentor = AmodalFlowAugmentor(**aug_params)
            elif sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.sseg_list = []
        self.image_list = []
        self.extra_info = []
        self.semantic_id_map_full = {}
        self.semantic_id_map_amodal = {}

    def __getitem__(self, index):
        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        if self.amodal:
            # Note: This is a bit of a hack... we don't have sparse
            # ground-truth for amodal, so our "valid" masks are actually
            # amodal/object masks. But we should treat/augment them the same
            # way
            flow = [frame_utils.read_gen_flow(p) for p in self.flow_list[index]]
            flow, valid = zip(*flow)

            id_map_full = np.vectorize(self.semantic_id_map_full.__getitem__)
            id_map_amodal = np.vectorize(self.semantic_id_map_amodal.__getitem__)

            sseg = [frame_utils.read_semantics(p) for p in self.sseg_list[index]]
            sseg = [id_map_full(s) for s in sseg[:2]] + [
                id_map_amodal(s) for s in sseg[2:]
            ]
        else:
            flow, valid = frame_utils.read_gen_flow(self.flow_list[index])
            flow, valid = [flow], [valid]

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = [np.array(f).astype(np.float32) for f in flow]
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.amodal:
                img1, img2, flow, valid, sseg = self.augmentor(img1, img2, flow, valid, sseg)
            elif self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow[0], valid[0])
                flow, valid = [flow], [valid]
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow[0])
                flow, valid = [flow], [np.ones(flow.shape[:2])]

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = [torch.from_numpy(f).permute(2, 0, 1).float() for f in flow]
        valid = [torch.from_numpy(v).float() for v in valid]

        if self.amodal:
            sseg = [torch.from_numpy(s).long() for s in sseg]

            # pad for amodal
            flow = flow + [torch.zeros_like(flow[0])] * max(
                0, self.amodal_layers - len(flow)
            )
            valid = valid + [torch.zeros_like(valid[0])] * max(
                0, self.amodal_layers - len(valid)
            )
            sseg = sseg + [255 * torch.ones_like(sseg[0])] * max(
                0, self.amodal_layers - len(sseg)
            )

            # "valid" masks are actually amodal masks... so unpack and re-pack
            # them accordingly. Note that we don't have any sparse ground truth
            # yet, so set the actual valid masks accordingly
            flow_full, flow_empty, *flow_amodal = flow
            _, _, *mask_amodal = valid

            flow = [flow_full, flow_empty] + list(zip(flow_amodal, mask_amodal))
            valid = [(f[0].abs() < 1000) & (f[1].abs() < 1000) for f in flow[:2]] + [
                (f[0].abs() < 1000) & (f[1].abs() < 1000) for f in flow_amodal
            ]

        else:
            flow = flow[0]
            valid = valid[0]

            if valid is None:
                valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000).float()

        if self.show_extra_info:
            if self.amodal:
                return img1, img2, flow, valid, sseg, self.extra_info[index]
            else:
                return img1, img2, flow, valid, self.extra_info[index]
        else:
            if self.amodal:
                return img1, img2, flow, valid, sseg
            else:
                return img1, img2, flow, valid

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)


class MpiSintel_submission(FlowDataset):
    def __init__(
        self, aug_params=None, split="test", root="datasets/Sintel", dstype="clean"
    ):
        super(MpiSintel_submission, self).__init__(aug_params)
        flow_root = osp.join(root, split, "flow")
        image_root = osp.join(root, split, dstype)

        if split == "test":
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, "*.png")))
            for i in range(len(image_list) - 1):
                self.image_list += [[image_list[i], image_list[i + 1]]]
                self.extra_info += [(scene, i)]  # scene and frame_id

            if split != "test":
                self.flow_list += sorted(glob(osp.join(flow_root, scene, "*.flo")))


class MpiSintel(FlowDataset):
    def __init__(
        self, aug_params=None, split="training", root="datasets/Sintel", dstype="clean"
    ):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, "flow")
        image_root = osp.join(root, split, dstype)

        if split == "test":
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, "*.png")))
            for i in range(len(image_list) - 1):
                self.image_list += [[image_list[i], image_list[i + 1]]]
                self.extra_info += [(scene, i)]  # scene and frame_id

            if split != "test":
                self.flow_list += sorted(glob(osp.join(flow_root, scene, "*.flo")))


class FlyingChairs(FlowDataset):
    def __init__(
        self, aug_params=None, split="train", root="datasets/FlyingChairs_release/data"
    ):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, "*.ppm")))
        flows = sorted(glob(osp.join(root, "*.flo")))
        assert len(images) // 2 == len(flows)

        split_list = np.loadtxt("chairs_split.txt", dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split == "training" and xid == 1) or (
                split == "validation" and xid == 2
            ):
                self.flow_list += [flows[i]]
                self.image_list += [[images[2 * i], images[2 * i + 1]]]


class FlyingThings3D(FlowDataset):
    def __init__(
        self,
        aug_params=None,
        root="datasets/FlyingThings3D",
        dstype="frames_cleanpass",
        split="training",
    ):
        super(FlyingThings3D, self).__init__(aug_params)

        split_dir = "TRAIN" if split == "training" else "TEST"
        for cam in ["left"]:
            for direction in ["into_future", "into_past"]:
                image_dirs = sorted(glob(osp.join(root, dstype, f"{split_dir}/*/*")))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(
                    glob(osp.join(root, f"optical_flow/{split_dir}/*/*"))
                )
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, "*.png")))
                    flows = sorted(glob(osp.join(fdir, "*.pfm")))
                    for i in range(len(flows) - 1):
                        if direction == "into_future":
                            self.image_list += [[images[i], images[i + 1]]]
                            self.flow_list += [flows[i]]
                        elif direction == "into_past":
                            self.image_list += [[images[i + 1], images[i]]]
                            self.flow_list += [flows[i + 1]]


class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split="training", root="datasets/KITTI"):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == "testing":
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, "image_2/*_10.png")))
        images2 = sorted(glob(osp.join(root, "image_2/*_11.png")))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split("/")[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]

        if split == "training":
            self.flow_list = sorted(glob(osp.join(root, "flow_occ/*_10.png")))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root="datasets/HD1k"):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(
                glob(os.path.join(root, "hd1k_flow_gt", "flow_occ/%06d_*.png" % seq_ix))
            )
            images = sorted(
                glob(os.path.join(root, "hd1k_input", "image_2/%06d_*.png" % seq_ix))
            )

            if len(flows) == 0:
                break

            for i in range(len(flows) - 1):
                self.flow_list += [flows[i]]
                self.image_list += [[images[i], images[i + 1]]]

            seq_ix += 1


class AmSynthDrive(FlowDataset):
    PATH_IMG = "full/images/{camera}/{sequence}/{camera}_full_{frame}_rgb.jpg"
    PATH_FLOW_FULL = "full/flow/{camera}/{sequence}/{camera}_full_{frame}_flow.png"
    PATH_FLOW_EMPTY = "empty/flow/{camera}/{sequence}/{camera}_empty_{frame}_flow.png"
    PATH_AMFLOW = "full/amodal_optical_flow/{camera}/{sequence}/{frame}/{camera}_full_{frame}_amflow.json"
    PATH_FULL_SSEG = (
        "full/semantic_seg/{camera}/{sequence}/{camera}_full_{frame}_seg.png"
    )
    PATH_EMPTY_SSEG = (
        "empty/semantic_seg/{camera}/{sequence}/{camera}_empty_{frame}_seg.png"
    )
    PATH_AMSSEG_INFO = "full/amodal_semantic_seg_for_flow/{camera}/{sequence}/{frame}/{camera}_full_{frame}_amsemseg.json"

    LAYERS = 2 + 8  # full + empty + 8*amodal

    def __init__(
        self,
        aug_params=None,
        root="datasets/AmSynthDrive",
        split_cfg="datasets/amsynthdrive.json",
        camera="front",
        amodal=False,
        split="train",
        show_extra_info=False,
    ):
        super().__init__(
            aug_params,
            amodal=amodal,
            amodal_layers=AmSynthDrive.LAYERS,
            show_extra_info=show_extra_info,
        )

        root = Path(root)
        split_cfg = Path(split_cfg)

        self.semantic_id_map_full = AMSYNTHDRIVE_ID_MAP_FULL
        self.semantic_id_map_amodal = AMSYNTHDRIVE_ID_MAP_AMODAL

        with open(split_cfg) as fd:
            split_cfg = json.load(fd)

        sequences = split_cfg[split]

        # Note: Select data based on path template variables, allows for lists, * is
        # wildcard
        #
        # e.g.:
        #   selector = {'camera': 'front'}
        #   selector = {'camera': ['front', 'back']}

        camera = [camera] if isinstance(camera, str) else camera
        frame = ["*"]

        selection = list(itertools.product(camera, sequences, frame))

        # get amflow.json file paths
        paths = set()
        for camera, sequence, frame in selection:
            glob_path = str(root / AmSynthDrive.PATH_AMFLOW)
            glob_path = glob_path.format(camera=camera, sequence=sequence, frame=frame)

            paths |= set(glob(glob_path))

        paths = [Path(p).relative_to(root) for p in paths]
        paths = sorted(paths)

        # get image, flow, and amodal flow file paths
        images = []
        full_flows = []
        empty_flows = []
        amodal_flows = []
        full_sseg = []
        empty_sseg = []
        amodal_sseg = []
        info = []

        for path in paths:
            attr = parse.parse(AmSynthDrive.PATH_AMFLOW, str(path))
            attr = attr.named

            frame1 = attr["frame"]
            frame2 = str(int(frame1, 10) + 1).zfill(4)

            path_img1 = root / AmSynthDrive.PATH_IMG.format_map(
                {**attr, "frame": frame1}
            )
            path_img2 = root / AmSynthDrive.PATH_IMG.format_map(
                {**attr, "frame": frame2}
            )

            path_flow_full = root / AmSynthDrive.PATH_FLOW_FULL.format_map(attr)
            path_flow_empty = root / AmSynthDrive.PATH_FLOW_EMPTY.format_map(attr)

            path_full_sseg = root / AmSynthDrive.PATH_FULL_SSEG.format_map(attr)
            path_empty_sseg = root / AmSynthDrive.PATH_EMPTY_SSEG.format_map(attr)
            path_amsseg_info = root / AmSynthDrive.PATH_AMSSEG_INFO.format_map(attr)

            with open(root / path) as fd:
                am_flow_info = json.load(fd)

            with open(path_amsseg_info) as fd:
                am_sseg_info = json.load(fd)

            amflow = [(k, v["file_name"]) for k, v in am_flow_info["flow"].items()]
            amflow = sorted(amflow)
            paths_amflow = [(root / path).parent / file for _, file in amflow]

            amsseg = [(k, v['file_name']) for k, v in am_sseg_info['seg'].items()]
            amsseg = sorted(amsseg)
            paths_amsseg = [path_amsseg_info.parent / file for _, file in amsseg]

            assert path_img1.exists()
            assert path_img2.exists()
            assert path_flow_full.exists()
            assert path_flow_empty.exists()
            assert path_full_sseg.exists()
            assert path_empty_sseg.exists()
            assert all([p.exists() for p in paths_amflow])
            assert all([p.exists() for p in paths_amsseg])

            images += [(str(path_img1), str(path_img2))]
            full_flows += [str(path_flow_full)]
            empty_flows += [str(path_flow_empty)]
            amodal_flows += [tuple(str(p) for p in paths_amflow)]
            info += [attr]

            full_sseg += [str(path_full_sseg)]
            empty_sseg += [str(path_empty_sseg)]
            amodal_sseg += [tuple(str(p) for p in paths_amsseg)]

        self.image_list = images
        self.extra_info = info

        if amodal:
            # construct list of tuples, containing in the following order paths for
            # - flow of full scene
            # - flow of background/stuff
            # - flows of things/objects (inline)
            self.flow_list = [
                (f, e, *a) for f, e, a in zip(full_flows, empty_flows, amodal_flows)
            ]
            self.sseg_list = [
                (f, e, *a) for f, e, a in zip(full_sseg, empty_sseg, amodal_sseg)
            ]
        else:
            self.flow_list = full_flows


def fetch_dataloader(args, TRAIN_DS="C+T+K+S+H"):
    """Create the data loader for the corresponding trainign set"""

    if args.stage == "chairs":
        aug_params = {
            "crop_size": args.image_size,
            "min_scale": -0.1,
            "max_scale": 1.0,
            "do_flip": True,
        }
        train_dataset = FlyingChairs(aug_params, split="training")

    elif args.stage == "things":
        aug_params = {
            "crop_size": args.image_size,
            "min_scale": -0.4,
            "max_scale": 0.8,
            "do_flip": True,
        }
        clean_dataset = FlyingThings3D(aug_params, dstype="frames_cleanpass")
        final_dataset = FlyingThings3D(aug_params, dstype="frames_finalpass")
        train_dataset = clean_dataset + final_dataset

    elif args.stage == "sintel":
        aug_params = {
            "crop_size": args.image_size,
            "min_scale": -0.2,
            "max_scale": 0.6,
            "do_flip": True,
        }
        things = FlyingThings3D(aug_params, dstype="frames_cleanpass")
        sintel_clean = MpiSintel(aug_params, split="training", dstype="clean")
        sintel_final = MpiSintel(aug_params, split="training", dstype="final")

        if TRAIN_DS == "C+T+K+S+H":
            kitti = KITTI(
                {
                    "crop_size": args.image_size,
                    "min_scale": -0.3,
                    "max_scale": 0.5,
                    "do_flip": True,
                }
            )
            hd1k = HD1K(
                {
                    "crop_size": args.image_size,
                    "min_scale": -0.5,
                    "max_scale": 0.2,
                    "do_flip": True,
                }
            )
            train_dataset = (
                100 * sintel_clean
                + 100 * sintel_final
                + 200 * kitti
                + 5 * hd1k
                + things
            )

        elif TRAIN_DS == "C+T+K/S":
            train_dataset = 100 * sintel_clean + 100 * sintel_final + things

    elif args.stage == "kitti":
        aug_params = {
            "crop_size": args.image_size,
            "min_scale": -0.2,
            "max_scale": 0.4,
            "do_flip": False,
        }
        train_dataset = KITTI(aug_params, split="training")

    elif args.stage == "amsynthdrive":
        aug_params = {
            "crop_size": args.image_size,
            "min_scale": -0.2,
            "max_scale": 0.4,
            "do_flip": False,
        }
        train_dataset = AmSynthDrive(
            aug_params, camera=["front", "back"], amodal=True, split="train"
        )

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )

    print("Training with %d image pairs" % len(train_dataset))
    return train_loader
