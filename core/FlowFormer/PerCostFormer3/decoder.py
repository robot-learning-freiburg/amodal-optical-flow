import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from ...utils.utils import bilinear_sampler, coords_grid
from ...utils import labels
from .attention import (
    ExpPositionEmbeddingSine,
    LinearPositionEmbeddingSine,
    MultiHeadAttention,
)
from .gma import Attention
from .gru import BasicUpdateBlock, GMAUpdateBlock
from .sk import SKUpdateBlock6_Deep_nopoolres_AllDecoder


def initialize_flow(img):
    """Flow is represented as difference between two means flow = mean1 - mean0"""
    N, C, H, W = img.shape
    mean = coords_grid(N, H, W).to(img.device)
    mean_init = coords_grid(N, H, W).to(img.device)

    # optical flow computed as difference: flow = mean1 - mean0
    return mean, mean_init


class CrossAttentionLayer(nn.Module):
    # def __init__(self, dim, cfg, num_heads=8, attn_drop=0., proj_drop=0., drop_path=0., dropout=0.):
    def __init__(
        self,
        qk_dim,
        v_dim,
        query_token_dim,
        tgt_token_dim,
        flow_or_pe,
        num_heads=8,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        dropout=0.0,
        pe="linear",
        no_sc=False,
    ):
        super(CrossAttentionLayer, self).__init__()

        head_dim = qk_dim // num_heads
        self.scale = head_dim**-0.5
        self.query_token_dim = query_token_dim
        self.pe = pe

        self.norm1 = nn.LayerNorm(query_token_dim)
        self.norm2 = nn.LayerNorm(query_token_dim)
        self.multi_head_attn = MultiHeadAttention(qk_dim, num_heads)
        self.q, self.k, self.v = (
            nn.Linear(query_token_dim, qk_dim, bias=True),
            nn.Linear(tgt_token_dim, qk_dim, bias=True),
            nn.Linear(tgt_token_dim, v_dim, bias=True),
        )

        self.proj = nn.Linear(v_dim, query_token_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.ffn = nn.Sequential(
            nn.Linear(query_token_dim, query_token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_token_dim, query_token_dim),
            nn.Dropout(dropout),
        )
        self.flow_or_pe = flow_or_pe
        print("[Decoder flow_or_pe setting is: {}]".format(self.flow_or_pe))
        self.no_sc = no_sc
        if self.no_sc:
            print("[No short cut in cost decoding]")
        self.dim = qk_dim

    def forward(self, query, key, value, memory, query_coord):
        """
        query_coord [B, 2, H1, W1]
        """
        B, _, H1, W1 = query_coord.shape

        if key is None and value is None:
            key = self.k(memory)
            value = self.v(memory)

        # [B, 2, H1, W1] -> [BH1W1, 1, 2]
        query_coord = query_coord.contiguous()
        query_coord = (
            query_coord.view(B, 2, -1)
            .permute(0, 2, 1)[:, :, None, :]
            .contiguous()
            .view(B * H1 * W1, 1, 2)
        )
        if self.pe == "linear":
            query_coord_enc = LinearPositionEmbeddingSine(query_coord, dim=self.dim)
        elif self.pe == "exp":
            query_coord_enc = ExpPositionEmbeddingSine(query_coord, dim=self.dim)
        elif self.pe == "norm_linear":
            query_coord[:, :, 0:1] = query_coord[:, :, 0:1] / W1
            query_coord[:, :, 1:2] = query_coord[:, :, 1:2] / H1
            query_coord_enc = LinearPositionEmbeddingSine(
                query_coord, dim=self.dim, NORMALIZE_FACOR=2
            )

        short_cut = query
        if query is not None:
            query = self.norm1(query)

        if self.flow_or_pe == "and":
            q = self.q(query + query_coord_enc)
        elif self.flow_or_pe == "pe":
            q = self.q(query_coord_enc)
        elif self.flow_or_pe == "flow":
            q = self.q(query)
        else:
            print("[Wrong setting of flow_or_pe]")
            exit()
        k, v = key, value

        x = self.multi_head_attn(q, k, v)

        x = self.proj(x)
        # x = self.proj(torch.cat([x, short_cut],dim=2))
        if short_cut is not None and not self.no_sc:
            # print("short cut")
            x = short_cut + self.proj_drop(x)

        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x, k, v


class MemoryDecoderLayer(nn.Module):
    def __init__(self, dim, cfg):
        super(MemoryDecoderLayer, self).__init__()
        self.cfg = cfg
        self.patch_size = cfg.patch_size  # for converting coords into H2', W2' space

        query_token_dim, tgt_token_dim = cfg.query_latent_dim, cfg.cost_latent_dim
        qk_dim, v_dim = query_token_dim, query_token_dim

        self.cross_attend = CrossAttentionLayer(
            qk_dim,
            v_dim,
            query_token_dim,
            tgt_token_dim,
            flow_or_pe=cfg.flow_or_pe,
            dropout=cfg.dropout,
            pe=cfg.pe,
            no_sc=cfg.no_sc,
        )

    def forward(self, query, key, value, memory, coords1, size):
        """
        x:      [B*H1*W1, 1, C]
        memory: [B*H1*W1, H2'*W2', C]
        coords1 [B, 2, H2, W2]
        size: B, C, H1, W1
        1. Note that here coords0 and coords1 are in H2, W2 space.
           Should first convert it into H2', W2' space.
        2. We assume the upper-left point to be [0, 0], instead of letting center of upper-left patch to be [0, 0]
        """
        x_global, k, v = self.cross_attend(query, key, value, memory, coords1)
        B, C, H1, W1 = size
        C = self.cfg.query_latent_dim
        x_global = x_global.view(B, H1, W1, C).permute(0, 3, 1, 2)
        return x_global, k, v


class MemoryDecoder(nn.Module):
    def __init__(self, cfg):
        super(MemoryDecoder, self).__init__()
        dim = self.dim = cfg.query_latent_dim
        self.cfg = cfg

        if cfg.use_patch:
            print("[Using cost patch as local cost]")
            self.flow_token_encoder = nn.Conv2d(
                cfg.cost_latent_input_dim + 64, cfg.query_latent_dim, 1, 1
            )
        else:
            self.flow_token_encoder = nn.Sequential(
                nn.Conv2d(81 * cfg.cost_heads_num, dim, 1, 1),
                nn.GELU(),
                nn.Conv2d(dim, dim, 1, 1),
            )

        if self.cfg.fix_pe:
            print("[fix_pe: regress 8*8 block]")
            self.pretrain_head = nn.Sequential(
                nn.Conv2d(dim, dim * 2, 1, 1),
                nn.GELU(),
                nn.Conv2d(dim * 2, dim * 2, 1, 1),
                nn.GELU(),
                nn.Conv2d(dim * 2, 64, 1, 1),
            )
        elif self.cfg.gt_r > 0:
            print("[Using larger cost as gt, radius is {}]".format(self.cfg.gt_r))
            # self.pretrain_head = nn.Conv2d(dim, self.cfg.gt_r**2, 1, 1)
            self.pretrain_head = nn.Sequential(
                nn.Conv2d(dim, dim * 2, 1, 1),
                nn.GELU(),
                nn.Conv2d(dim * 2, dim * 2, 1, 1),
                nn.GELU(),
                # nn.Conv2d(dim*2, dim*2, 1, 1),
                # nn.GELU(),
                # nn.Conv2d(dim*2, dim*2, 1, 1),
                # nn.GELU(),
                # nn.Conv2d(dim*2, dim*2, 1, 1),
                # nn.GELU(),
                # nn.Conv2d(dim*2, dim*2, 1, 1),
                # nn.GELU(),
                nn.Conv2d(dim * 2, self.cfg.gt_r**2, 1, 1),
            )
        else:
            self.pretrain_head = nn.Sequential(
                nn.Conv2d(dim, dim * 2, 1, 1),
                nn.GELU(),
                nn.Conv2d(dim * 2, dim * 2, 1, 1),
                nn.GELU(),
                nn.Conv2d(dim * 2, 81, 1, 1),
            )

        self.proj = nn.Conv2d(cfg.encoder_latent_dim, 256, 1)
        self.depth = cfg.decoder_depth
        self.decoder_layer = MemoryDecoderLayer(dim, cfg)

        n_fullcls = labels.N_CLASSES_FULL
        n_amcls = labels.N_CLASSES_AMODAL

        if self.cfg.gma == "GMA":
            print("[Using GMA]")
            self.update_block = GMAUpdateBlock(self.cfg, hidden_dim=128)
            self.update_block_bg = GMAUpdateBlock(self.cfg, hidden_dim=128)
            self.update_block_am = GMAUpdateBlock(self.cfg, hidden_dim=128)
            self.att = Attention(
                args=self.cfg, dim=128, heads=1, max_pos_size=160, dim_head=128
            )
        elif self.cfg.gma == "GMA-SK":
            print("[Using GMA-SK]")
            self.update_block = SKUpdateBlock6_Deep_nopoolres_AllDecoder(
                args=self.cfg, hidden_dim=128
            )
            self.update_block_bg = SKUpdateBlock6_Deep_nopoolres_AllDecoder(
                args=self.cfg, hidden_dim=128
            )
            self.update_block_am = SKUpdateBlock6_Deep_nopoolres_AllDecoder(
                args=self.cfg, hidden_dim=128
            )
            self.att = Attention(
                args=self.cfg, dim=128, heads=1, max_pos_size=160, dim_head=128
            )
        else:
            print("[Not using GMA decoder]")
            self.update_block = BasicUpdateBlock(self.cfg, hidden_dim=128)
            self.update_block_bg = BasicUpdateBlock(self.cfg, hidden_dim=128)
            self.update_block_am = BasicUpdateBlock(self.cfg, hidden_dim=128)

        if self.cfg.r_16 > 0:
            raise ValueError("r_16 > 0 is unsupported")

        if self.cfg.quater_refine:
            raise ValueError("Using Quater Refinement is not supported")

        self.combine_bg = nn.Sequential(
            nn.Conv2d(
                2 * 128 + 2 + n_fullcls, 256, 3, padding=1
            ),  # 2x state + flow + semantics
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.combine_am0 = nn.Sequential(
            nn.Conv2d(
                2 * 128 + 2 + n_fullcls, 256, 3, padding=1
            ),  # 2x state + flow + semantics
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.combine_am1 = nn.Sequential(
            nn.Conv2d(
                4 * 128 + 2 + 2, 256, 3, padding=1
            ),  # 4x state + flow + 2x amodal mask
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.object_logits = nn.Sequential(
            nn.Conv2d(2 * 128 + 128, 256, 3, padding=1),  # 2x state + context
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_fullcls, 1, padding=0),
        )

        self.amodal_visible_logits = nn.Sequential(
            nn.Conv2d(4 * 128 + 128, 256, 3, padding=1),  # 4x state + context
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1, padding=0),
        )

        self.amodal_occlusion_logits = nn.Sequential(
            nn.Conv2d(
                4 * 128 + 128 + 1, 256, 3, padding=1
            ),  # 4x state + context + visible-mask
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1, padding=0),
        )

        self.amodal_score_logits = nn.Sequential(
            nn.Conv2d(128 + 2, 256, 3, padding=1),  # 1x state + flow
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1, padding=0),
        )

        self.amodal_class_logits = nn.Sequential(
            nn.Conv2d(128 + 2, 256, 3, padding=1),  # 1x state + flow
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_amcls, 1, padding=0),
        )

        self.proj_bg = nn.Conv2d(256, 128, 1)
        self.proj_am = nn.Conv2d(256, 128, 1)

    def upsample_field(self, flow, mask):
        """Upsample vector/feature field [H/8, W/8, C] -> [H, W, C] using convex combination"""
        N, C, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, C, 8 * H, 8 * W)

    def sample_feature_map(self, coords, feat_t_quater, r=1):
        H, W = feat_t_quater.shape[-2:]
        xgrid, ygrid = coords.split([1, 1], dim=-1)
        xgrid = 2 * xgrid / (W - 1) - 1
        ygrid = 2 * ygrid / (H - 1) - 1

        grid = torch.cat([xgrid, ygrid], dim=-1)
        img = F.grid_sample(feat_t_quater, grid, align_corners=True)

        if mask:
            mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
            return img, mask.float()

        return img

    def encode_flow_token(self, cost_maps, coords, r=4):
        """
        cost_maps   -   B*H1*W1, cost_heads_num, H2, W2
        coords      -   B, 2, H1, W1
        """
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        dx = torch.linspace(-r, r, 2 * r + 1)
        dy = torch.linspace(-r, r, 2 * r + 1)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

        centroid = coords.reshape(batch * h1 * w1, 1, 1, 2)
        delta = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
        coords = centroid + delta
        corr = bilinear_sampler(cost_maps, coords)

        corr = corr.view(batch, h1, w1, -1).permute(0, 3, 1, 2)
        return corr

    def encode_cost_query(self, cost, cost_patches, coords1, size):
        if self.cfg.use_patch:
            if self.cfg.detach_local:
                _local_cost = self.encode_flow_token(cost_patches, coords1 / 8.0, r=0)
                _local_cost = _local_cost.contiguous().detach()
                query = self.flow_token_encoder(_local_cost)
            else:
                query = self.flow_token_encoder(
                    self.encode_flow_token(cost_patches, coords1 / 8.0, r=0)
                )
        else:
            if self.cfg.detach_local:
                _local_cost = cost.contiguous().detach()
                query = self.flow_token_encoder(_local_cost)
            else:
                query = self.flow_token_encoder(cost)

        query = query.permute(0, 2, 3, 1).contiguous()
        query = query.view(size[0] * size[2] * size[3], 1, self.dim)

        return query

    def decode_correlation(
        self, coords0, coords1, query, key, value, cost_memory, cost_forward, size
    ):
        if self.cfg.use_rpe:
            query_coord = coords1 - coords0
        else:
            query_coord = coords1

        cost_global, key, value = self.decoder_layer(
            query, key, value, cost_memory, query_coord, size
        )

        corr = torch.cat([cost_global, cost_forward], dim=1)

        return corr, key, value

    def compute_flow_delta(self, fn, net, inp, corr, flow, attention):
        if self.cfg.gma is not None:
            net, up_mask, flow_delta = fn(net, inp, corr, flow, attention)
        else:
            net, up_mask, flow_delta = fn(net, inp, corr, flow)

        return net, up_mask, flow_delta

    def compute_score_logits(self, net, visible_mask, occlusion_mask):
        x = torch.cat((net, visible_mask, occlusion_mask), dim=1)
        return self.amodal_score_logits(x)

    def compute_amodal_class_logits(self, net, visible_mask, occlusion_mask):
        x = torch.cat((net, visible_mask, occlusion_mask), dim=1)
        return self.amodal_class_logits(x)

    def compute_object_logits(self, net_full, net_empty, ctx):
        x = torch.cat((net_full, net_empty, ctx), dim=1)
        return self.object_logits(x)

    def compute_amodal_visible_logits(
        self, net_full, net_empty, net_amodal, net_amodal_prev, ctx
    ):
        x = torch.cat((net_full, net_empty, net_amodal, net_amodal_prev, ctx), dim=1)
        return self.amodal_visible_logits(x)

    def compute_amodal_occlusion_logits(
        self, net_full, net_empty, net_amodal, net_amodal_prev, ctx, visible_mask
    ):
        x = torch.cat(
            (net_full, net_empty, net_amodal, net_amodal_prev, ctx, visible_mask), dim=1
        )
        return self.amodal_occlusion_logits(x)

    def forward(
        self,
        cost_memory,
        context,
        data={},
        flow_init=None,
        cost_patches=None,
    ):
        """
        memory: [B*H1*W1, H2'*W2', C]
        context: [B, D, H1, W1]
        """
        cost_maps = data["cost_maps"]
        coords0, coords1_fg = initialize_flow(context)

        coords1_bg = coords1_fg.detach().clone()
        coords1_am = [
            coords1_fg.detach().clone() for _ in range(self.cfg.amodal_layers)
        ]

        if flow_init is not None:
            raise RuntimeError("warm-start not supported")

        flow_predictions = []

        net_fg, inp = torch.split(self.proj(context), [128, 128], dim=1)
        net_fg = torch.tanh(net_fg)
        inp = torch.relu(inp)

        net_bg = torch.tanh(self.proj_bg(context))

        net_am = torch.tanh(self.proj_am(context))
        net_am = [None] + [net_am.clone() for _ in range(self.cfg.amodal_layers)]

        attention = None
        if self.cfg.gma is not None:
            attention = self.att(inp)

        size = net_fg.shape

        key, value = None, None

        for idx in range(self.depth):
            coords1_fg = coords1_fg.detach()
            coords1_bg = coords1_bg.detach()
            coords1_am = [c.detach() for c in coords1_am]

            # sample costs (forward flow) from cost map
            cost_fg = self.encode_flow_token(cost_maps, coords1_fg)
            cost_bg = self.encode_flow_token(cost_maps, coords1_bg)
            cost_am = [self.encode_flow_token(cost_maps, c) for c in coords1_am]

            # encode sampled costs into actual query
            query_fg = self.encode_cost_query(cost_fg, cost_patches, coords1_fg, size)
            query_bg = self.encode_cost_query(cost_bg, cost_patches, coords1_bg, size)
            query_am = [
                self.encode_cost_query(cv, cost_patches, c1, size)
                for cv, c1 in zip(cost_am, coords1_am)
            ]

            # decode corsts/correlation from cost memory
            corr_fg, key, value = self.decode_correlation(
                coords0, coords1_fg, query_fg, key, value, cost_memory, cost_fg, size
            )
            corr_bg, _, _ = self.decode_correlation(
                coords0, coords1_bg, query_bg, key, value, cost_memory, cost_bg, size
            )
            corr_am = [
                self.decode_correlation(
                    coords0, c1, q, key, value, cost_memory, cv, size
                )[0]
                for cv, c1, q in zip(cost_am, coords1_am, query_am)
            ]

            # current flow estimate
            flow_fg = coords1_fg - coords0
            flow_bg = coords1_bg - coords0
            flow_am = [c - coords0 for c in coords1_am]

            # full/foreground: compute flow delta, update and upsample flow
            net_fg, up_mask_fg, delta_fg = self.compute_flow_delta(
                self.update_block, net_fg, inp, corr_fg, flow_fg, attention
            )

            coords1_fg = coords1_fg + delta_fg
            flow_fg = coords1_fg - coords0
            flow_up_fg = self.upsample_field(flow_fg, up_mask_fg)

            # compute combined object/motion mask
            object_logits = self.compute_object_logits(net_fg, net_bg, inp)
            object_semantics = F.softmax(object_logits, dim=1)
            object_logits_up = self.upsample_field(object_logits, up_mask_fg)

            # background: compute flow delta, update and upsample flow
            net_bg = torch.cat((net_bg, net_fg, flow_fg, object_semantics), dim=1)
            net_bg = self.combine_bg(net_bg)
            net_bg, up_mask_bg, delta_bg = self.compute_flow_delta(
                self.update_block_bg, net_bg, inp, corr_bg, flow_bg, attention
            )

            coords1_bg = coords1_bg + delta_bg
            flow_up_bg = self.upsample_field(coords1_bg - coords0, up_mask_bg)

            # amodal flow: compute flow delta, update and upsample flow
            net_am[0] = torch.cat((net_bg, net_fg, flow_fg, object_semantics), dim=1)
            net_am[0] = self.combine_am0(net_am[0])

            res_am = []
            for i in range(self.cfg.amodal_layers):
                visible_logits = self.compute_amodal_visible_logits(
                    net_fg, net_bg, net_am[i + 1], net_am[i], inp
                )
                visible_mask = torch.sigmoid(visible_logits)

                occlusion_logits = self.compute_amodal_occlusion_logits(
                    net_fg, net_bg, net_am[i + 1], net_am[i], inp, visible_mask
                )
                occlusion_mask = torch.sigmoid(occlusion_logits)

                net = torch.cat((net_fg, net_bg, net_am[i + 1], net_am[i]), dim=1)
                net = torch.cat((net, flow_fg, visible_mask, occlusion_mask), dim=1)
                net = self.combine_am1(net)

                net, up_mask, delta = self.compute_flow_delta(
                    self.update_block_am, net, inp, corr_am[i], flow_am[i], attention
                )

                net_am[i + 1] = net

                coords1_am[i] = coords1_am[i] + delta
                flow_up = self.upsample_field(coords1_am[i] - coords0, up_mask)

                score = self.compute_score_logits(net, visible_mask, occlusion_mask)
                score_up = self.upsample_field(score, up_mask)

                amcls = self.compute_amodal_class_logits(net, visible_mask, occlusion_mask)
                amcls_up = self.upsample_field(amcls, up_mask)

                visible_logits_up = self.upsample_field(visible_logits, up_mask)
                occlusion_logits_up = self.upsample_field(occlusion_logits, up_mask)

                res_am.append(
                    (flow_up, score_up, amcls_up, visible_logits_up, occlusion_logits_up)
                )

            flow_predictions.append([flow_up_fg, flow_up_bg, object_logits_up] + res_am)

        if self.training:
            return flow_predictions
        else:
            # TODO: implement warm-start for the amodal setting
            return flow_predictions[-1], coords1_fg - coords0
