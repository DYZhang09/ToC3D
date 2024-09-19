from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import BACKBONES
from mmdet3d.models.builder import build_loss
from torch.utils.checkpoint import checkpoint

from .eva_vit import Block, Attention
from .eva_utils import (
    Backbone,
    PatchEmbed,
    get_abs_pos,
    window_partition,
    window_unpartition,
    VisionRotaryEmbeddingFast,
    VisionRotaryEmbeddingFastWithSelection
)
from projects.mmdet3d_plugin.models.backbones.toc3d_utils import MotionAwareQueryGuidedTokenSelector, ToC3DViTReturnType
from projects.mmdet3d_plugin.models.backbones.toc3d_utils import merge_tokens, batch_index_fill, batch_index_select 
from projects.mmdet3d_plugin.models.utils.gpu_timer import GLOBAL_TIMER


@BACKBONES.register_module()
class ToC3DEVAViT(Backbone):
    def __init__(
            self,
            img_size=1024,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4 * 2 / 3,
            qkv_bias=True,
            drop_path_rate=0.0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            use_abs_pos=True,
            use_rel_pos=False,
            rope=True,
            rope_acc=False,
            pt_hw_seq_len=16,
            intp_freq=True,
            window_size=0,
            global_window_size=20,
            use_checkpoint = True,
            # window_block_indexes=(),
            global_attn_indexes=(),
            residual_block_indexes=(),
            use_act_checkpoint=False,
            pretrain_img_size=224,
            pretrain_use_cls_token=True,
            out_feature="last_feat",
            return_intermediate = False,
            xattn=True,
            pruning_loc = None,
            pruning_score_type = 'attention',
            score_mask = True,
            pruning_attn_scale = True,
            pruning_num_queries=256,
            accelerate_global = True,
            token_ratio = None,
            use_represent_tokens = True,
            pc_range = None,
            token_selection_loss = None,
    ):
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            window_block_indexes (list): Indexes for blocks using window attention.
            residual_block_indexes (list): Indexes for blocks using conv propagation.
            use_act_checkpoint (bool): If True, use activation checkpointing.
            pretrain_img_size (int): input image size for pretraining models.
            pretrain_use_cls_token (bool): If True, pretrainig models use class token.
            out_feature (str): name of the feature from the last block.
        """
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.use_checkpoint = use_checkpoint
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        half_head_dim = embed_dim // num_heads // 2
        hw_seq_len = img_size // patch_size

        if rope:
            self.rope_win = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=window_size if intp_freq else None,
            )
            self.rope_glb = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len if intp_freq else None,
            )
        else:
            self.rope_win = None
            self.rope_glb = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        ##################### modules for ToC3D #########################
        self.pruning_loc = pruning_loc
        self.use_represent_tokens = use_represent_tokens
        self.accelerate_global = accelerate_global
        self.token_ratio = token_ratio
        if token_selection_loss is not None:
            self.token_selection_loss = build_loss(token_selection_loss)
        else:
            self.token_selection_loss = None
        assert len(list(set(pruning_loc).intersection(set(global_attn_indexes)))) == 0, \
            "The pruning score calculation layer cannot be the global attention layer"
        self.pruning_num_queries = pruning_num_queries
        self.pruning_attn_scale = pruning_attn_scale
        self.score_predictor = nn.ModuleList(
            [MotionAwareQueryGuidedTokenSelector(
                embed_dim=embed_dim,
                num_queries=pruning_num_queries,
                ratio=token_ratio[i],
                attn_scale=self.pruning_attn_scale,
                use_mask=score_mask,
                pc_range=pc_range,
                score_type=pruning_score_type
            ) for i in range(len(pruning_loc))]
        )

        # print some info
        if self.accelerate_global:
            print('*' * 20 + ' will prune tokens in global attention layer' + '*' * 20)
        if rope and rope_acc:
            print('*' * 20 + f' using RoPE with selection indexes ' + '*' * 20)
            self.rope_win_acc = VisionRotaryEmbeddingFastWithSelection(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=window_size if intp_freq else None,
            )
            self.rope_glb_acc = VisionRotaryEmbeddingFastWithSelection(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len if intp_freq else None,
            )
        else:
            self.rope_win_acc = self.rope_glb_acc = None 
        ###################################################################

        self.blocks = nn.ModuleList()
        for i in range(depth):
            accelerate = (len(self.pruning_loc) > 0) and (i >= self.pruning_loc[0]) and (
                (self.accelerate_global) or (i not in global_attn_indexes)
            )
            if rope:
                if accelerate and rope_acc:
                    block_rope = self.rope_glb_acc if i in global_attn_indexes else self.rope_win_acc
                elif accelerate and not rope_acc:
                    block_rope = None
                else:
                    block_rope = self.rope_glb if i in global_attn_indexes else self.rope_win
            else:
                block_rope = None
            block = ToC3DEVAViTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                window_size=window_size if i not in global_attn_indexes else global_window_size,
                use_residual_block=i in residual_block_indexes,
                rope=block_rope,
                accelerate=accelerate,
            )
            if use_act_checkpoint:
                from fairscale.nn.checkpoint import checkpoint_wrapper

                block = checkpoint_wrapper(block)
            self.blocks.append(block)

        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}
        self._out_features = [out_feature]

        self.return_intermediate = return_intermediate

        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def forward(
        self, 
        x, 
        temp_queries=None, 
        prev_exists=None, 
        temp_ref_points=None,
        temp_vel=None,
        temp_timestamp=None,
        temp_ego_pose=None,
        ego_pose_inv=None,
        *args, 
        **kwargs
    ):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
            )

        B, H, W = x.shape[0], x.shape[1], x.shape[2]
        pruning_loc = 0
        masks = torch.ones([B, H, W, 1], device=x.device)
        decisions = []
        attn_scores = []
        keep_idxes = []
        drop_idxes = []
        scores = None
        score_predictor = None
        override_ratio = None
        if self.return_intermediate:
            aux_outputs = list()

        GLOBAL_TIMER.event_start(f'ToC3D-StreamPETR-EVAViT/backbone')
        for i, blk in enumerate(self.blocks):
            if i in self.pruning_loc:
                score_predictor = self.score_predictor[pruning_loc]
                keep_idx, drop_idx, masks, scores, attn_score = score_predictor(
                    input_x = x,
                    mask = masks,
                    temp_queries=temp_queries,
                    temp_ref_points=temp_ref_points,
                    temp_vel=temp_vel,
                    temp_timestamp=temp_timestamp,
                    temp_ego_pose=temp_ego_pose,
                    do_sample=True,
                    override_ratio=None,
                    prev_exists=prev_exists,
                    ego_pose_inv=ego_pose_inv,
                )[-5:]

                pruning_loc += 1
                decisions.append(masks)
                if attn_score is not None:
                    attn_scores.append(attn_score)
                keep_idxes.append(keep_idx)
                drop_idxes.append(drop_idx)

            x =  checkpoint(blk, x, scores, score_predictor, override_ratio=override_ratio, use_represent_tokens=self.use_represent_tokens) if self.use_checkpoint \
                else blk(x, scores, score_predictor, override_ratio=override_ratio, use_represent_tokens=self.use_represent_tokens)

            if self.return_intermediate:
                aux_outputs.append(x.permute(0, 3, 1, 2))

        GLOBAL_TIMER.event_end(f'ToC3D-StreamPETR-EVAViT/backbone')
        outputs = {self._out_features[0]: x.permute(0, 3, 1, 2)}

        if len(decisions) == 0:
            decisions = None
        if len(attn_scores) == 0:
            attn_scores = None
        if len(keep_idxes) == 0:
            keep_idxes = None
        if len(drop_idxes) == 0:
            drop_idxes = None
        res = ToC3DViTReturnType(
            outputs, decisions, attn_scores, 
            keep_idx=keep_idxes, 
            drop_idx=drop_idxes,
            aux_outputs=aux_outputs if self.return_intermediate else None
        )
        return res

    def loss(
        self,
        pred_masks,
        gt_bboxes,
        *args, 
        **kwargs
    ):
        losses = dict()
        if self.token_selection_loss is not None:
            token_selection_loss = self.token_selection_loss(
                pred_mask=pred_masks,
                gt_bboxes=gt_bboxes,
            )
            losses.update(token_selection_loss)
        return losses


class ToC3DEVAViTBlock(Block):
    def __init__(
        self, 
        dim, 
        num_heads, 
        mlp_ratio=4 * 2 / 3, 
        qkv_bias=True, 
        drop_path=0, 
        norm_layer=..., 
        window_size=0, 
        use_residual_block=False, 
        rope=None, 
        accelerate: bool = False,
        *args, 
        **kwargs,
    ):
        super().__init__(
            dim, 
            num_heads, 
            mlp_ratio, 
            qkv_bias, 
            drop_path, 
            norm_layer, 
            window_size, 
            use_residual_block, 
            rope, 
        )

        self.accelerate = accelerate
        if self.accelerate:
            self.attn = ToC3DEVAAttention(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                rope=rope,
            )

    def forward_slow(
        self,
        slow_tokens: torch.Tensor,
        attn_select_index: torch.Tensor,
    ):
        shortcut = slow_tokens
        slow_tokens = self.norm1(slow_tokens)

        if self.attn.rope is not None:
            assert attn_select_index is not None
            raw_1 = self.attn(slow_tokens, attn_select_index) 
        else:
            raw_1 = self.attn(slow_tokens)
        slow_tokens = shortcut + raw_1

        shortcut = slow_tokens
        slow_tokens = self.norm2(slow_tokens)
        raw_2 = self.mlp(slow_tokens) 
        slow_tokens = shortcut + raw_2

        return slow_tokens, raw_1, raw_2

    def forward_fast(
        self,
        fast_tokens: torch.Tensor,
        attn_select_index: torch.Tensor = None,
    ):
        return fast_tokens

    def forward(
        self,
        x: torch.Tensor,
        scores: torch.Tensor = None,
        score_predictor: MotionAwareQueryGuidedTokenSelector = None,
        override_ratio = None,
        use_represent_tokens = True,
        *args, 
        **kwargs,
    ):
        if self.accelerate:
            assert scores is not None and scores.shape[:2] == x.shape[:2]
            assert score_predictor is not None

            B, H, W, C = x.shape

            # Window partition
            if self.window_size > 0:
                H, W = x.shape[1], x.shape[2]
                x, pad_hw = window_partition(x, self.window_size)
                scores, _ = window_partition(scores.unsqueeze(-1), self.window_size, pad_value=-1e6)

            # step 1: select the informative tokens
            override_ratio = override_ratio if self.training else None
            slow_score, fast_score, slow_index, fast_index, _ = score_predictor.sample(scores, override_ratio=override_ratio)

            slow_tokens = batch_index_select(x.flatten(1, 2), slow_index)  # B x N_slow x C

            # step 2: select the uninformative tokens and scores, generate the representative tokens
            fast_tokens = batch_index_select(x.flatten(1, 2), fast_index)  # B x N_fast x C
            
            if use_represent_tokens and fast_tokens.shape[1] > 0:
                represent_tokens = merge_tokens(fast_tokens, fast_score)  # B x 1 x C

                # step 3: slow update the informative tokens and representative tokens
                slow_tokens = torch.cat([slow_tokens, represent_tokens], dim=1)  # B x (N_slow + 1) x C

            if self.attn.rope is not None:
                if use_represent_tokens and fast_tokens.shape[1] > 0:
                    slow_attn_select_index = torch.ones([slow_index.shape[0], 1], device=slow_index.device) * slow_index.shape[-1]
                    slow_attn_select_index = torch.cat([slow_index, slow_attn_select_index], dim=-1).long()
                else:
                    slow_attn_select_index = slow_index.long()
                fast_attn_select_index = fast_index.long()
            else:
                slow_attn_select_index = None
                fast_attn_select_index = None

            # slow path
            slow_tokens, raw_1, raw_2 = self.forward_slow(slow_tokens, slow_attn_select_index)
            # fast path
            fast_tokens = self.forward_fast(fast_tokens, fast_attn_select_index)

            if use_represent_tokens and fast_tokens.shape[1] > 0:
                slow_tokens = slow_tokens[:, :-1]  # B x N_slow x C

                # step 4: fast update the uninformative tokens by copying the representative ones
                represent_tokens1 = raw_1[:, -1:]  # B x 1 x C
                represent_tokens2 = raw_2[:, -1:]  
                fast_tokens = fast_tokens + \
                                represent_tokens1.expand(-1, fast_tokens.shape[1], -1) + \
                                represent_tokens2.expand(-1, fast_tokens.shape[1], -1)  # B x N_fast x C

            # step 5: organize tokens to regular shape
            if fast_tokens.shape[1] > 0:
                x0 = torch.zeros_like(x).flatten(1, 2)
                x = batch_index_fill(x0, slow_tokens, fast_tokens, slow_index, fast_index).view(-1, self.window_size, self.window_size, C)
            else:
                x = slow_tokens

            # Reverse window partition
            if self.window_size > 0:
                x = window_unpartition(x, self.window_size, pad_hw, (H, W))
                # scores = window_unpartition(scores, self.window_size, pad_hw, (H, W)).squeeze(-1)

            if self.use_residual_block:
                x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

            return x
        
        else:
            x = super().forward(x)
            return x


class ToC3DEVAAttention(Attention):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_head_dim=None, rope=None):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_head_dim, rope)

    def forward(self, x, selected_idx=None, *args, **kwargs):
        reorganize = False
        if len(x.shape) == 4:
            B, H, W, C = x.shape
            x = x.view(B, -1, C)
            N = H * W
            reorganize = True
        else:
            assert len(x.shape) == 3
            B, N, C = x.shape

        q = F.linear(input=x, weight=self.q_proj.weight, bias=self.q_bias)
        k = F.linear(input=x, weight=self.k_proj.weight, bias=None)
        v = F.linear(input=x, weight=self.v_proj.weight, bias=self.v_bias)

        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # B, num_heads, N, C
        k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        ## rope
        if self.rope is not None:
            assert selected_idx is not None
            q = self.rope(q, selected_idx).type_as(v)
            k = self.rope(k, selected_idx).type_as(v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1).type_as(x)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        if reorganize:
            x = x.view(B, H, W, C)

        return x