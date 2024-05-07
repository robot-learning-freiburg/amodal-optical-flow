import torch.nn as nn

from ..encoders import convnext_large, nat_base, twins_svt_large
from .cnn import BasicEncoder
from .decoder import MemoryDecoder
from .encoder import MemoryEncoder


class FlowFormer(nn.Module):
    def __init__(self, cfg):
        super(FlowFormer, self).__init__()

        H1, W1, H2, W2 = cfg.pic_size
        H_offset = (H1 - H2) // 2
        W_offset = (W1 - W2) // 2
        cfg.H_offset = H_offset
        cfg.W_offset = W_offset

        self.cfg = cfg

        self.memory_encoder = MemoryEncoder(cfg)
        self.memory_decoder = MemoryDecoder(cfg)
        if cfg.cnet == "twins":
            self.context_encoder = twins_svt_large(
                pretrained=self.cfg.pretrain, del_layers=cfg.del_layers
            )
        elif cfg.cnet == "basicencoder":
            self.context_encoder = BasicEncoder(output_dim=256, norm_fn="instance")
        elif cfg.cnet == "convnext":
            self.context_encoder = convnext_large(pretrained=self.cfg.pretrain)
        elif cfg.cnet == "nat":
            self.context_encoder = nat_base(pretrained=self.cfg.pretrain)

        if cfg.pretrain_mode:
            print("[In pretrain mode, freeze context encoder]")
            for param in self.context_encoder.parameters():
                param.requires_grad = False

    def forward(self, image1, image2, mask=None, output=None, flow_init=None):
        if self.cfg.pretrain_mode:
            raise RuntimeError(
                "Pretraining not supported, use pretrained FlowFormer++ weights instead"
            )
        else:
            # Following https://github.com/princeton-vl/RAFT/
            image1 = 2 * (image1 / 255.0) - 1.0
            image2 = 2 * (image2 / 255.0) - 1.0

            data = {}

            context, _ = self.context_encoder(image1)

            cost_memory, cost_patches = self.memory_encoder(
                image1, image2, data, context
            )

            flow_predictions = self.memory_decoder(
                cost_memory,
                context,
                data,
                flow_init=flow_init,
                cost_patches=cost_patches,
            )

            return flow_predictions
