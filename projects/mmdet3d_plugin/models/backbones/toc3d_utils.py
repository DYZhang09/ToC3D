from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F

from projects.mmdet3d_plugin.models.utils.positional_encoding import pos2posemb3d, pos2posemb1d, nerf_positional_encoding
from projects.mmdet3d_plugin.models.utils.misc import MLN, transform_reference_points


class ToC3DViTReturnType:
    def __init__(
        self,
        img_feats = None,
        token_masks = None,
        attn_scores = None,
        keep_idx = None,
        drop_idx = None,
        aux_outputs: list = None
    ) -> None:
        self.img_feats = img_feats
        self.token_masks = token_masks
        self.attn_scores = attn_scores
        self.keep_idx = keep_idx
        self.drop_idx = drop_idx
        self.aux_outputs = aux_outputs


def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError


def batch_index_fill(x, x1, x2, idx1, idx2):
    B, N, C = x.size()
    B, N1, C = x1.size()
    B, N2, C = x2.size()

    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1)
    idx1 = idx1 + offset * N
    idx2 = idx2 + offset * N

    x = x.reshape(B*N, C)

    x[idx1.reshape(-1)] = x1.reshape(B*N1, C)
    x[idx2.reshape(-1)] = x2.reshape(B*N2, C)

    x = x.reshape(B, N, C)
    return x


def merge_tokens(x_drop, score):
    # score B,N
    # scale
    weight = score / torch.sum(score, dim=1, keepdim=True)
    x_drop = weight.unsqueeze(-1) * x_drop
    return torch.sum(x_drop, dim=1, keepdim=True)


class TokenSelectorBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def score(self, input_x, mask, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def sample(self, pred_score: torch.Tensor, override_ratio=None):
        raise NotImplementedError

    @abstractmethod
    def forward(self, input_x, mask, *args, **kwargs):
        raise NotImplementedError


class ScoreBasedTokenSelector(TokenSelectorBase):
    """ Importance Score Predictor
    """
    def __init__(self, embed_dim=384, hard_score=False, ratio=0.5, use_mask=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.hard_score = hard_score
        self.use_mask = use_mask
        self.ratio = ratio
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.LogSoftmax(dim=-1)
        )

    def score(self, input_x, mask, *args, **kwargs):
        B, H, W, C = input_x.shape
        if self.use_mask:
            assert mask is not None
            input_x = (input_x * mask).view(B, H * W, C)
        else:
            input_x = input_x.view(B, H * W, C)
            
        x = self.in_conv(input_x)
        B, N, C = x.size()
        local_x = x[:, :, :C//2]
        global_x = torch.mean(x[:, :, C//2:], keepdim=True, dim=(1))
        x = torch.cat([local_x, global_x.expand(B, N, C//2)], dim=2)
        pred_score = self.out_conv(x)

        return pred_score

    def sample(self, pred_score: torch.Tensor, override_ratio=None):
        ratio = override_ratio if override_ratio is not None else self.ratio
        if len(pred_score.shape) == 4:
            pred_score = pred_score.flatten(1, 2)
        if len(pred_score.shape) == 3:
            score = pred_score[:, :, 0]  # B x (H * W)
        B, N = score.shape[:2]
        num_keep_node = int(N * ratio)
        sorted_score, sorted_idx = torch.sort(score, dim=1, descending=True)
        keep_score = sorted_score[:, :num_keep_node]
        keep_idx = sorted_idx[:, :num_keep_node]
        drop_score = sorted_score[:, num_keep_node:]
        drop_idx = sorted_idx[:, num_keep_node:]

        if self.training or (not self.hard_score):
            # differentiable
            new_mask = F.gumbel_softmax(pred_score, hard=False, dim=-1)[:, :, 0:1]
        else:
            new_mask = torch.zeros([B, N, 1], device=pred_score.device)
            new_mask = batch_index_fill(
                new_mask, 
                torch.ones([B, num_keep_node, 1], device=pred_score.device), 
                torch.zeros([B, N - num_keep_node, 1], device=pred_score.device),
                keep_idx,
                drop_idx
            )

        return keep_score, drop_score, keep_idx, drop_idx, new_mask

    def forward(self, input_x, mask, do_sample=True, override_ratio=None, *args, **kwargs):
        '''
        Args:
            input_x (torch.Tensor): B x H x W x C
            mask (torch.Tensor): B x H x W x 1. Defaults to None.
            ratio (float, optional): _description_. Defaults to 0.5.
            do_sample (bool): if do the top-k sampling, if not, only return the importance score of each token. 
                                Defaults to True.

        Returns:
            if do_sample is True:
                keep_score: B x N_keep
                drop_score: B x N_drop
                keep_idx: B x N_keep
                drop_idx: B x N_drop
                new_mask: B x H x W x 1
                score: B x H x W
            else:
                score: B x H x W
        '''
        B, H, W, C = input_x.shape
        pred_score = self.score(input_x, mask)
        score = pred_score[:, :, 0]  # B x (H * W)
        if do_sample:
            keep_score, drop_score, keep_idx, drop_idx, new_mask = self.sample(pred_score, override_ratio)
            new_mask = new_mask.view(B, H, W, 1)

            return keep_score, drop_score, keep_idx, drop_idx, new_mask, score.view(B, H, W), None
        else:
            return score.view(B, H, W)


class NaiveQueryGuidedTokenSelector(ScoreBasedTokenSelector):
    SUPPORTED_SCORE_TYPE = ['attention', 'score']

    def __init__(
        self, 
        embed_dim=384, 
        query_dim=256,
        num_queries=256, 
        hard_score=False, 
        use_mask=True,
        ratio=0.5, 
        attn_scale=True,
        score_type='attention',
    ):
        super().__init__(embed_dim=embed_dim, hard_score=hard_score, ratio=ratio, use_mask=use_mask)
        self.score_type = score_type
        if self.score_type not in self.SUPPORTED_SCORE_TYPE:
            raise NotImplementedError(f'Not supported score type: {self.score_type}, only support: {self.SUPPORTED_SCORE_TYPE}')
        if self.score_type != 'attention':
            print('*' * 20 + f' Using Non-default type: {self.score_type} for query guided token selection!!!' + '*' * 20)
        
        self.num_queries = num_queries
        self.scale = query_dim ** -0.5 if attn_scale else 1.0

        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim, query_dim)
        )

        if self.score_type == 'attention':
            self.aggregate = nn.Sequential(
                nn.Linear(num_queries, 2),
                nn.LogSoftmax(dim=-1)
            )
        if self.score_type == 'score':
            self.query_pool = nn.AdaptiveAvgPool1d(1)
            self.aggregate = nn.Sequential(
                MLPBlock(input_dim=2 * query_dim, hidden_dim=query_dim, out_dim=2),
                nn.LogSoftmax(dim=-1)
            )

    def query_based_score(self, input_x: torch.Tensor, mask, queries: torch.Tensor, *args, **kwargs):
        if self.use_mask:
            assert mask is not None
            input_x = (input_x * mask).flatten(1, 2)  # B x N x C
        else:
            input_x = input_x.flatten(1, 2)

        input_x = self.input_proj(input_x)  # B x N x C_Q
        queries_embed = queries.repeat_interleave(input_x.shape[0] // queries.shape[0], dim=0)  # B x Q x C_Q

        attention = None
        if self.score_type == 'attention':
            attention = torch.einsum('bnc,bqc->bnq', input_x, queries_embed) * self.scale  # B x N x Q
            pred_score = self.aggregate(attention)  # B x N x 2
        elif self.score_type == 'score':
            mean_query_embed = self.query_pool(queries_embed.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()  # B x 1 x C_Q
            mean_query_embed = mean_query_embed.expand_as(input_x).contiguous()
            x_query = torch.concat([input_x, mean_query_embed], dim=-1)  # B x N x (2 * C_Q)
            pred_score = self.aggregate(x_query)

        return pred_score, attention

    def score(
        self, 
        input_x: torch.Tensor, 
        mask, 
        temp_queries: torch.Tensor, 
        prev_exists: torch.Tensor,
        *args, **kwargs,
    ):
        if self.training:
            # a trick to fix the unused parameter problem
            super_score = super().score(input_x, mask)[..., :1]
            pred_score, attention = self.query_based_score(input_x, mask, temp_queries)
            pred_score = super_score * (1 - prev_exists) + pred_score * prev_exists
            attention = None if not prev_exists else attention
        else:
            attention = None
            if not prev_exists:
                pred_score = super().score(input_x, mask)
            else:
                pred_score = self.query_based_score(input_x, mask, temp_queries)[0]
        return pred_score, attention

    def forward(
        self,
        input_x,
        mask,
        temp_queries,
        do_sample=True,
        override_ratio=None,
        prev_exists: torch.Tensor = None,
        *args, **kwargs
    ):
        B, H, W, C = input_x.shape
        pred_score, attention = self.score(input_x, mask, temp_queries, prev_exists, *args, **kwargs)
        score = pred_score[:, :, 0]  # B x (H * W)
        if do_sample:
            keep_score, drop_score, keep_idx, drop_idx, new_mask = self.sample(pred_score, override_ratio)
            new_mask = new_mask.view(B, H, W, 1)

            return keep_score, drop_score, keep_idx, drop_idx, new_mask, score.view(B, H, W), attention
        else:
            return score.view(B, H, W)


class MotionAwareQueryGuidedTokenSelector(NaiveQueryGuidedTokenSelector):
    def __init__(
        self, 
        embed_dim=384, 
        query_dim=256, 
        num_queries=256, 
        hard_score=False, 
        use_mask=True,
        ratio=0.5,
        attn_scale=True,
        pc_range=None,
        score_type='attention'
    ):
        super().__init__(
            embed_dim, query_dim, num_queries, 
            hard_score=hard_score, 
            use_mask=use_mask, 
            ratio=ratio, 
            attn_scale=attn_scale,
            score_type=score_type
        )

        assert pc_range is not None
        self.pc_range = nn.Parameter(torch.tensor(pc_range), requires_grad=False)
        self.query_embedding = nn.Sequential(
            nn.Linear(query_dim*3//2, query_dim),
            nn.ReLU(),
            nn.Linear(query_dim, query_dim),
        )
        self.ego_pose_pe = MLN(180)
        self.ego_pose_queries = MLN(180)
        self.time_embedding = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.LayerNorm(query_dim)
        )

    def get_motion_aware_queries(
        self,
        temp_queries,
        temp_ref_points,
        temp_vel,
        temp_timestamp,
        temp_ego_pose,
        ego_pose_inv,
    ):
        # fuse motion info with queries to make queries motion-aware
        # 1. transform ref_points from global coords to current ego coords
        assert ego_pose_inv is not None
        temp_ref_points = transform_reference_points(temp_ref_points, ego_pose_inv, reverse=False)
        # 2. encode the ref points
        temp_ref_points = (temp_ref_points - self.pc_range[:3]) / (self.pc_range[3:6] - self.pc_range[0:3])
        temp_pos = self.query_embedding(pos2posemb3d(temp_ref_points)) 
        # 3. pass the encoded ref points together with the vel, timestamp and ego pose into the MLN to get the tmp_pos
        tmp_ego_motion = torch.cat([temp_vel, temp_timestamp, temp_ego_pose[..., :3, :].flatten(-2)], dim=-1).float()
        tmp_ego_motion = nerf_positional_encoding(tmp_ego_motion)
        temp_pos = self.ego_pose_pe(temp_pos, tmp_ego_motion)
        temp_pos += self.time_embedding(pos2posemb1d(temp_timestamp).float())

        temp_queries = self.ego_pose_queries(temp_queries, tmp_ego_motion)
        # 4. add tmp_pos with queries to make queries motion-aware
        temp_queries += temp_pos

        return temp_queries

    def score(
        self, 
        input_x: torch.Tensor, 
        mask, 
        queries: torch.Tensor, 
        temp_ref_points,
        temp_vel,
        temp_timestamp,
        temp_ego_pose,
        ego_pose_inv,
        prev_exists,
        *args, 
        **kwargs
    ):
        queries = self.get_motion_aware_queries(
            queries,
            temp_ref_points,
            temp_vel,
            temp_timestamp,
            temp_ego_pose,
            ego_pose_inv
        )

        return super().score(input_x, mask, queries, prev_exists, *args, **kwargs)

    def forward(
        self,
        input_x,
        mask,
        temp_queries,
        temp_ref_points,
        temp_vel,
        temp_timestamp,
        temp_ego_pose,
        ego_pose_inv,
        do_sample=True,
        override_ratio=None,
        prev_exists=None,
        *args,
        **kwargs
    ):
        B, H, W, C = input_x.shape

        pred_score, attention = self.score(
            input_x, mask,
            temp_queries,
            temp_ref_points,
            temp_vel,
            temp_timestamp,
            temp_ego_pose,
            ego_pose_inv,
            prev_exists
        )
        score = pred_score[:, :, 0]  # B x (H * W)
        if do_sample:
            keep_score, drop_score, keep_idx, drop_idx, new_mask = self.sample(pred_score, override_ratio)
            new_mask = new_mask.view(B, H, W, 1)

            return keep_score, drop_score, keep_idx, drop_idx, new_mask, score.view(B, H, W), attention
        else:
            return score.view(B, H, W)


class MLPBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        act = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))