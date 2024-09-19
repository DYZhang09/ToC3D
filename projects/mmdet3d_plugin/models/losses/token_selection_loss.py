from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import LOSSES, build_loss
from mmdet.core import multi_apply

from projects.mmdet3d_plugin.models.utils.token_select_vis import token_selection_vis


class TokenSelectionSemanticTargetAssigner(nn.Module):
    def __init__(
        self, 
        bg_mask_value = 0.0,
        patch_size = 16,
    ) -> None:
        super().__init__()

        self.bg_mask_value = bg_mask_value
        self.patch_size = patch_size

    @torch.no_grad()
    def _get_semantic_mask_target_single_view(
        self,
        gt_bboxes_single: torch.Tensor,
        mask_shape: Tuple[int],
    ):
        '''get the targets for semantic mask loss of a single view

        Args:
            token_ratio (List[float]): the list of desired token ratios
            gt_bboxes_single (torch.Tensor): the 2d gt_bboxes with shape num_gt x 4
            gt_centers2d (torch.Tensor): the center of 2d gt_bboxes with shape num_gt x 2
            gt_depths (torch.Tensor): the depth of 2d gt_bboxes with shape num_gt
            cross_attn_weights (torch.Tensor): the cross attention map with shape num_queries x (mask_h * mask_w)
            mask_shape (Tuple[int]): the shape of mask
        '''
        mask_h, mask_w = mask_shape
        mask_label = torch.ones(mask_h, mask_w, device=gt_bboxes_single.device) * self.bg_mask_value
        if gt_bboxes_single.shape[0] > 0:
            tl_x, tl_y, br_x, br_y = torch.split(gt_bboxes_single, 1, -1)  # num_gt x 1

            tl_x = torch.clamp(torch.floor((tl_x) / self.patch_size), min=0).long().flatten()
            tl_y = torch.clamp(torch.floor((tl_y) / self.patch_size), min=0).long().flatten()
            br_x = torch.clamp(torch.ceil((br_x) / self.patch_size), max=mask_w - 1).long().flatten()
            br_y = torch.clamp(torch.ceil((br_y) / self.patch_size), max=mask_h - 1).long().flatten()

            for i in range(gt_bboxes_single.shape[0]):
                mask_label[tl_y[i]:br_y[i], tl_x[i]:br_x[i]] = 1.0
            
        return (mask_label,)

    @torch.no_grad()
    def get_semantic_mask_target(
        self,
        gt_bboxes: List[torch.Tensor],
        mask_shape: Tuple[int, int],
    ):
        '''get the semantic target for semantic mask loss of a single sample with multiple views
            
        Args:
            token_ratio (List[float]): the list of desired token ratios, len is num_pruning_layers
            gt_bboxes (List[torch.Tensor]): the list of 2D gt_bboxes, len is num_views
            gt_centers2d (List[torch.Tensor]): the list of 2D centers of gt_bboxes, len is num_views
            gt_depths (List[torch.Tensor]): the lit of 2D depths of gt_bboxes, len is num_views 
            mask_shape (Tuple[int]): the shape of mask,
            token_ratio (List[float]): the list of token keep raitos, len is num_pruning_layers
            cross_attn_weights (List[torch.Tensor]): the cross attn map from decoder, len is num_views, defaults to None

        Returns:
            mask_labels (torch.Tensor): the mask labels with shape num_views x H x W
        '''
        (mask_labels, ) = multi_apply(
            self._get_semantic_mask_target_single_view,
            gt_bboxes,
            mask_shape = mask_shape,
        )
        mask_labels = torch.stack(mask_labels)
        return mask_labels


@LOSSES.register_module()
class TokenSelectionLoss(nn.Module):
    def __init__(
        self,
        patch_size = 16,
        semantic_loss = None,
        class_weights = None,
        bg_mask_value = 0.0,
    ):
        super(TokenSelectionLoss, self).__init__()
        self.patch_size = patch_size
        self.class_weights = class_weights

        self.semantic_loss_func = build_loss(semantic_loss) if semantic_loss is not None else None
        if semantic_loss is not None:
            self.semantic_target_assigner = TokenSelectionSemanticTargetAssigner(
                bg_mask_value=bg_mask_value,
                patch_size=self.patch_size
            )

    def semantic_loss_single(
        self,
        pred_mask: torch.Tensor,
        gt_bboxes: List[torch.Tensor],
    ):
        '''the semantic loss of a single sample 

        Args:
            pred_mask (torch.Tensor): the predicted token selection mask, with shape num_pruning_layers x num_views x H x W
            token_ratio (List[float]): the list of desired token ratios
            gt_bboxes (List[torch.Tensor]): the list of 2D gt_bboxes, len == num_views
            gt_centers2d (List[torch.Tensor]): the list of 2D centers of gt_bboxes, len == num_views
            gt_depths (List[torch.Tensor]): the list of 2D depths of gt_bboxes, len == num_views
        
        Returns:
            loss_semantic (torch.Tensor): the semantic loss of a single sample
        '''
        num_layers, num_views, mask_h, mask_w = pred_mask.shape
        mask_labels = self.semantic_target_assigner.get_semantic_mask_target(
            gt_bboxes=gt_bboxes, 
            mask_shape=(mask_h, mask_w), 
        )  # num_views x H x W
        mask_labels = mask_labels.unsqueeze(0).expand(num_layers, -1, -1, -1).contiguous()

        # calculate the loss weights for each token
        loss_weight = torch.ones_like(pred_mask)
        
        pred_mask = pred_mask.view(-1, 1)
        mask_labels = mask_labels.view_as(pred_mask)
        loss_weight = loss_weight.view_as(pred_mask)
        loss_semantic = self.semantic_loss_func(
            pred_mask,
            mask_labels,
            loss_weight
        )
        return loss_semantic
    
    def semantic_loss(
        self,
        pred_mask: torch.Tensor,
        gt_bboxes: Tuple[List[torch.Tensor]],
    ):
        '''Forward function for semantic loss

        Args:
            pred_mask (torch.Tensor): the predicted masks, shape is num_pruning_layers x B x N x H x W
            token_ratio (List[float]): the list of desired token ratio
            gt_bboxes (Tuple[List[torch.Tensor]]): the 2d gt bboxes, length = batch_size;
                each entry of list contains the gt_bboxes with shape num_gts x 4 of each view
            gt_centers2d (Tuple[List[torch.Tensor]]): the 2d gt centers, length = batch_size;
                each entry of list contains the 2d centers of gt_bboxes with shape num_gts x 2 of each view
            gt_depths (Tuple[List[torch.Tensor]]): the depths of 2d gts, length = batch_size;
                each entry of list contains the depths of 2D gt bboxes with shape num_gts

        Returns:
            semantic_loss (torch.Tensor): the semantic loss of total batch
        '''
        if self.semantic_loss_func is not None:
            all_gt_bboxes_list = [bboxes2d for i in gt_bboxes for bboxes2d in i]
            pred_mask = pred_mask.flatten(1, 2)  # num_pruning_layers x (B * N) x H x W

            semantic_loss = self.semantic_loss_single(
                pred_mask, 
                all_gt_bboxes_list, 
            )
            return semantic_loss

        else:
            return None

    def forward(
        self,
        pred_mask,
        gt_bboxes = None,
    ):
        total_loss = dict()

        semantic_loss = self.semantic_loss(pred_mask, gt_bboxes)
        if semantic_loss is not None:
            semantic_loss = torch.nan_to_num(semantic_loss)
            total_loss['semantic_loss'] = semantic_loss

        return total_loss
