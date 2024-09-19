# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.backbones.toc3d_utils import ToC3DViTReturnType
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.utils.misc import locations
from projects.mmdet3d_plugin.models.utils.token_select_vis import token_selection_vis
from projects.mmdet3d_plugin.models.utils.gpu_timer import GLOBAL_TIMER

@DETECTORS.register_module()
class Petr3D(MVXTwoStageDetector):
    """Petr3D."""

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 num_frame_head_grads=2,
                 num_frame_backbone_grads=2,
                 num_frame_losses=2,
                 stride=16,
                 position_level=0,
                 aux_2d_only=True,
                 single_test=False,
                 pretrained=None,
                 token_select_vis=False,
                 vis_num_sample=20,
                 vis_start_id=0,
                 vis_out_path='./token_vis',
                 test_time_print=False,
        ):
        super(Petr3D, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.prev_scene_token = None
        self.num_frame_head_grads = num_frame_head_grads
        self.num_frame_backbone_grads = num_frame_backbone_grads
        self.num_frame_losses = num_frame_losses
        self.single_test = single_test
        self.stride = stride
        self.position_level = position_level
        self.aux_2d_only = aux_2d_only

        # flags for prepare memory queries for backbone token selection
        self.query_backbone_selection = hasattr(self.img_backbone, 'pruning_loc') and \
            self.img_backbone.pruning_loc is not None

        # visualization flag for token selection
        self.token_select_vis = token_select_vis
        self.vis_num_sample = vis_num_sample
        self.vis_out_path = vis_out_path
        self.vis_start_id = vis_start_id
        self.cur_id = 0
        self.test_time_print = test_time_print

    def extract_img_feat(
        self, 
        img, 
        len_queue=1, 
        training_mode=False,
        gt_bboxes=None,
        centers2d=None,
        depths=None,
        prev_exists=None,
        ego_pose_inv=None,
    ):
        """Extract features of images."""
        B = img.size(0)

        if img is not None:
            if img.dim() == 6:
                img = img.flatten(1, 2)
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            if ego_pose_inv.dim() == 4:
                ego_pose_inv = ego_pose_inv.flatten(0, 1)

            gt_bboxes = [bboxes for i in gt_bboxes for bboxes in i] if gt_bboxes is not None else None
            centers2d = [ct2d for i in centers2d for ct2d in i] if centers2d is not None else None
            depths = [dep for i in depths for dep in i] if depths is not None else None
            
            # prepare the memory queries for backbone token selection
            if self.query_backbone_selection:
                # 1. use prev_exists to figure out if current frame is the first one of the sequence
                # 2. if prev_exists == False, then memory queries are set to None
                # 3. else, fetch the memory queries from pts_bbox_head.memory_embedding
                assert prev_exists is not None
                mid_frame = prev_exists.bool().flatten()[0].item()
                num_proposals = self.img_backbone.pruning_num_queries
                query_dim = self.pts_bbox_head.embed_dims
                if not mid_frame or self.pts_bbox_head.memory_embedding is None:
                    mem_queries = torch.zeros([B, num_proposals, query_dim], device=img.device)
                    mem_reference_point = torch.zeros([B, num_proposals, 3], device=img.device)
                    mem_timestamp = torch.zeros([B, num_proposals, 1], device=img.device)
                    mem_egopose = torch.zeros([B, num_proposals, 4, 4], device=img.device)
                    mem_velo = torch.zeros([B, num_proposals, 2], device=img.device)
                else:
                    mem_queries = self.pts_bbox_head.memory_embedding[:, :num_proposals, :].detach()
                    mem_reference_point = self.pts_bbox_head.memory_reference_point[:, :num_proposals, :].detach()
                    mem_timestamp = self.pts_bbox_head.memory_timestamp[:, :num_proposals, :].detach()
                    mem_egopose = self.pts_bbox_head.memory_egopose[:, :num_proposals, :, :].detach()
                    mem_velo = self.pts_bbox_head.memory_velo[:, :num_proposals, :].detach()
            else:
                mem_queries = None
                mem_reference_point = None
                mem_timestamp = None
                mem_egopose = None
                mem_velo = None
                mid_frame = False

            backbone_out = self.img_backbone(
                x=img, 
                gt_bboxes=gt_bboxes, 
                gt_centers2d=centers2d, 
                gt_depths=depths,
                temp_queries=mem_queries,
                temp_ref_points=mem_reference_point,
                temp_vel=mem_velo,
                temp_timestamp=mem_timestamp,
                temp_ego_pose=mem_egopose,
                prev_exists=mid_frame,
                ego_pose_inv=ego_pose_inv,
            )
            # for ToC3D
            if isinstance(backbone_out, ToC3DViTReturnType):
                token_masks = backbone_out.token_masks  # list of (BN, H, W, 1), len == num_pruning_layer
                attn_scores = backbone_out.attn_scores
                keep_idxes = backbone_out.keep_idx
                drop_idxes = backbone_out.drop_idx
                img_feats = backbone_out.img_feats

                if isinstance(img_feats, dict):
                    if 'aux_outputs' in img_feats:
                        aux_outputs = img_feats.pop('aux_outputs')
                    img_feats = list(img_feats.values())

            else:
                token_masks = None
                attn_scores = None
                keep_idxes = None
                drop_idxes = None
                if isinstance(backbone_out, dict):
                    if 'aux_outputs' in backbone_out:
                        aux_outputs = backbone_out.pop('aux_outputs')
                    img_feats = list(backbone_out.values())
                elif isinstance(backbone_out, list) or isinstance(backbone_out, tuple):
                    img_feats = backbone_out
                else:
                    raise NotImplementedError

        else:
            return None, None, None, None, None

        GLOBAL_TIMER.event_start('StreamPETR/img_neck')
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        BN, C, H, W = img_feats[self.position_level].size()
        if self.training or training_mode:
            img_feats_reshaped = img_feats[self.position_level].view(B, len_queue, int(BN/B / len_queue), C, H, W)

            if token_masks is not None:
                assert isinstance(token_masks, list), f'token_masks is not list but {type(token_masks)}'
                if len(token_masks) and isinstance(token_masks[0], list):
                    for i in range(len(token_masks)):
                        for j in range(len(token_masks[i])):
                            token_masks[i][j] = token_masks[i][j].view(B, len_queue, int(BN / B / len_queue), token_masks[i][j].shape[1], token_masks[i][j].shape[2])
                        if len(token_masks[i]):
                            token_masks[i] = torch.stack(token_masks[i])
                else:
                    for i in range(len(token_masks)):
                        token_masks[i] = token_masks[i].view(B, len_queue, int(BN / B / len_queue), token_masks[i].shape[1], token_masks[i].shape[2])
                    token_masks = torch.stack(token_masks)  # num_pruning_layer * B * len_queue * (BN / B / len_queue) * H * W
            if attn_scores is not None:
                assert isinstance(attn_scores, list), f'attn_scores is not list but {type(attn_scores)}'
                if len(attn_scores) and isinstance(attn_scores[0], list):
                    for i in range(len(attn_scores)):
                        for j in range(len(attn_scores[i])):
                            attn_scores[i][j] = attn_scores[i][j].view(B, len_queue, int(BN / B / len_queue), *attn_scores[i][j].shape[1:])
                        if len(attn_scores[i]):
                            attn_scores[i] = torch.stack(attn_scores[i])                           
                else:
                    for i in range(len(attn_scores)):
                        attn_scores[i] = attn_scores[i].view(B, len_queue, int(BN / B / len_queue), *attn_scores[i].shape[1:])
                    attn_scores = torch.stack(attn_scores)  # num_pruning_layer x B x len_queue x (BN / B / len_queue) x (H * W)
            if keep_idxes is not None:
                assert isinstance(keep_idxes, list), f'keep_idxes is not list but {type(keep_idxes)}'
                if len(keep_idxes) and isinstance(keep_idxes[0], list):
                    for i in range(len(keep_idxes)):
                        for j in range(len(keep_idxes[i])):
                            keep_idxes[i][j] = keep_idxes[i][j].view(B, len_queue, int(BN / B / len_queue), *keep_idxes[i][j].shape[1:])
                else:
                    for i in range(len(keep_idxes)):
                        keep_idxes[i] = keep_idxes[i].view(B, len_queue, int(BN / B / len_queue), *keep_idxes[i].shape[1:])
            if drop_idxes is not None:
                assert isinstance(drop_idxes, list), f'drop_idxes is not list but {type(drop_idxes)}'
                if len(drop_idxes) and isinstance(drop_idxes[0], list):
                    for i in range(len(drop_idxes)):
                        for j in range(len(drop_idxes[i])):
                            drop_idxes[i][j] = drop_idxes[i][j].view(B, len_queue, int(BN / B / len_queue), *drop_idxes[i][j].shape[1:])
                else:
                    for i in range(len(drop_idxes)):
                        drop_idxes[i] = drop_idxes[i].view(B, len_queue, int(BN / B / len_queue), *drop_idxes[i].shape[1:])
        else:
            img_feats_reshaped = img_feats[self.position_level].view(B, int(BN/B/len_queue), C, H, W)

        GLOBAL_TIMER.event_end('StreamPETR/img_neck')

        return img_feats_reshaped, token_masks, attn_scores, keep_idxes, drop_idxes


    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(
        self, 
        img, 
        T, 
        training_mode=False,
        gt_bboxes=None,
        centers2d=None,
        depths=None,
        prev_exists=None,
        ego_pose_inv=None,
    ):
        """Extract features from images and points."""
        img_feats, token_masks, attn_scores, keep_idxes, drop_idxes = self.extract_img_feat(img, T, training_mode, gt_bboxes, centers2d, depths, prev_exists, ego_pose_inv)
        return img_feats, token_masks, attn_scores, keep_idxes, drop_idxes

    def obtain_history_memory(self,
                            gt_bboxes_3d=None,
                            gt_labels_3d=None,
                            gt_bboxes=None,
                            gt_labels=None,
                            img_metas=None,
                            centers2d=None,
                            depths=None,
                            gt_bboxes_ignore=None,
                            **data):
        losses = dict()
        T = data['img'].size(1)
        num_nograd_frames = T - self.num_frame_head_grads
        num_grad_losses = T - self.num_frame_losses
        for i in range(T):
            requires_grad = False
            return_losses = False
            data_t = dict()
            for key in data:
                if key == 'rec_token_masks' or key == 'rec_attn_scores':
                    if isinstance(data[key], list):
                        dst_list = []
                        for element in data[key]:
                            if len(element):
                                dst_list.append(element[:, :, i])
                            else:
                                dst_list.append([])
                        data_t[key] = dst_list
                    elif isinstance(data[key], torch.Tensor):
                        data_t[key] = data[key][:, :, i]  # num_pruning_layer * B * (BN / B / len_queue) * H * W
                    else:
                        raise NotImplementedError
                else:
                    data_t[key] = data[key][:, i]

            data_t['img_feats'] = data_t['img_feats']
            if i >= num_nograd_frames:
                requires_grad = True
            if i >= num_grad_losses:
                return_losses = True
            loss = self.forward_pts_train(gt_bboxes_3d[i],
                                        gt_labels_3d[i], gt_bboxes[i],
                                        gt_labels[i], img_metas[i], centers2d[i], depths[i], requires_grad=requires_grad,return_losses=return_losses,**data_t)
            if loss is not None:
                for key, value in loss.items():
                    losses['frame_'+str(i)+"_"+key] = value
        return losses


    def prepare_location(self, img_metas, **data):
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        bs, n = data['img_feats'].shape[:2]
        x = data['img_feats'].flatten(0, 1)
        location = locations(x, self.stride, pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)
        return location

    def forward_roi_head(self, location, **data):
        if (self.aux_2d_only and not self.training) or not self.with_img_roi_head:
            return {'topk_indexes':None}
        else:
            outs_roi = self.img_roi_head(location, **data)
            return outs_roi


    def forward_pts_train(self,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_bboxes,
                          gt_labels,
                          img_metas,
                          centers2d,
                          depths,
                          requires_grad=True,
                          return_losses=False,
                          **data):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        location = self.prepare_location(img_metas, **data)

        if not requires_grad:
            self.eval()
            with torch.no_grad():
                outs = self.pts_bbox_head(location, img_metas, None, **data)
            self.train()

        else:
            outs_roi = self.forward_roi_head(location, **data)
            topk_indexes = outs_roi['topk_indexes']
            # Note: must clone first here, location will be modified inplace in pts_bbox_head
            outs = self.pts_bbox_head(location, img_metas, topk_indexes, **data)

        if return_losses:
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
            losses = self.pts_bbox_head.loss(
                *loss_inputs, 
            )

            # calculate the ratio loss for ToC3D
            if 'rec_token_masks' in data:
                assert hasattr(self.img_backbone, 'loss')
                rec_token_masks = data['rec_token_masks']  # num_pruning_layer * B * (BN / B / len_queue) * H * W
                token_selection_loss = self.img_backbone.loss(
                    pred_masks = rec_token_masks,
                    gt_bboxes = gt_bboxes,
                    gt_labels = gt_labels,
                    gt_centers2d = centers2d,
                    gt_depths = depths,
                )
                losses.update(token_selection_loss)

            if self.with_img_roi_head:
                # import ipdb; ipdb.set_trace()
                # gt_bboxes: tuple, len == batch_size,
                #       each entry is a list, len == num_views
                #           each entry is a Tensor of shape num_gt x 4 (tl_x, tl_y, br_x, br_y, unnormalized)
                # centers2d: the centers of each gt_bboxes, tuple, len == batch_size
                #       each entry is a list, len == nun_views
                #           each entry is a Tensor of shape num_gt x 2 (x, y, unnormalized)
                loss2d_inputs = [gt_bboxes, gt_labels, centers2d, depths, outs_roi, img_metas]
                losses2d = self.img_roi_head.loss(*loss2d_inputs)
                losses.update(losses2d) 

            return losses
        else:
            return None

    @force_fp32(apply_to=('img'))
    def forward(self, return_loss=True, **data):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            for key in ['gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'img_metas']:
                data[key] = list(zip(*data[key]))
            return self.forward_train(**data)
        else:
            for key in ['gt_bboxes', 'centers2d', 'depths']:
                if key in data:
                    data[key] = list(zip(*data[key]))
            return self.forward_test(**data)

    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      depths=None,
                      centers2d=None,
                      **data):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        GLOBAL_TIMER.set_activate(activate=False)  # DO NOT mearsure time consumption when training

        T = data['img'].size(1)

        prev_img = data['img'][:, :-self.num_frame_backbone_grads]
        rec_img = data['img'][:, -self.num_frame_backbone_grads:]
        # rec_img_feats = self.extract_feat(rec_img, self.num_frame_backbone_grads)[0]
        prev_gt_bboxes = gt_bboxes[:-self.num_frame_backbone_grads]
        prev_centers2d = centers2d[:-self.num_frame_backbone_grads]
        prev_depths = depths[:-self.num_frame_backbone_grads]
        rec_gt_bboxes = gt_bboxes[-self.num_frame_backbone_grads:]
        rec_centers2d = centers2d[-self.num_frame_backbone_grads:]
        rec_depths = depths[-self.num_frame_backbone_grads:]

        backbone_res = self.extract_feat(
            rec_img, 
            self.num_frame_backbone_grads,
            gt_bboxes=rec_gt_bboxes,
            centers2d=rec_centers2d,
            depths=rec_depths,
            prev_exists=data['prev_exists'],
            ego_pose_inv=data['ego_pose_inv']
        )
        rec_img_feats = backbone_res[0]
        rec_token_masks = backbone_res[1]

        if T-self.num_frame_backbone_grads > 0:
            self.eval()
            with torch.no_grad():
                prev_img_feats = self.extract_feat(
                    prev_img, 
                    T-self.num_frame_backbone_grads, 
                    True,
                    gt_bboxes=prev_gt_bboxes,
                    centers2d=prev_centers2d,
                    depths=prev_depths,
                    prev_exists=data['prev_exists'],
                    ego_pose_inv=data['ego_pose_inv']
                )[0]
            self.train()
            data['img_feats'] = torch.cat([prev_img_feats, rec_img_feats], dim=1)
        else:
            data['img_feats'] = rec_img_feats

        if rec_token_masks is not None:
            data['rec_token_masks'] = rec_token_masks

        losses = self.obtain_history_memory(gt_bboxes_3d,
                        gt_labels_3d, gt_bboxes,
                        gt_labels, img_metas, centers2d, depths, gt_bboxes_ignore, **data)

        return losses
  
  
    def forward_test(self, img_metas, rescale, gt_bboxes=None, centers2d=None, depths=None, **data):
        GLOBAL_TIMER.set_activate(activate=self.test_time_print)

        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        for key in data:
            if key != 'img':
                data[key] = data[key][0][0].unsqueeze(0)
            else:
                data[key] = data[key][0]
        return self.simple_test(img_metas[0], gt_bboxes, centers2d, depths, **data)

    def simple_test_pts(self, img_metas, **data):
        """Test function of point cloud branch."""
        location = self.prepare_location(img_metas, **data)
        outs_roi = self.forward_roi_head(location, **data)
        topk_indexes = outs_roi['topk_indexes']

        if img_metas[0]['scene_token'] != self.prev_scene_token:
            self.prev_scene_token = img_metas[0]['scene_token']
            data['prev_exists'] = data['img'].new_zeros(1)
            self.pts_bbox_head.reset_memory()
        else:
            data['prev_exists'] = data['img'].new_ones(1)

        outs = self.pts_bbox_head(location, img_metas, topk_indexes, **data)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results
    
    def simple_test(self, img_metas, gt_bboxes=None, centers2d=None, depths=None, **data):
        """Test function without augmentaiton."""

        if img_metas[0]['scene_token'] != self.prev_scene_token:
            prev_exists = data['img'].new_zeros(1)
        else:
            prev_exists = data['img'].new_ones(1)
 
        img_feats, token_masks, _, keep_idxes, drop_idxes = self.extract_img_feat(
            data['img'], 
            1,
            gt_bboxes=gt_bboxes,
            centers2d=centers2d,
            depths=depths,
            prev_exists=prev_exists,
            ego_pose_inv=data['ego_pose_inv']
        )
        data['img_feats'] = img_feats

        if self.token_select_vis and self.vis_num_sample > 0:
            # import ipdb; ipdb.set_trace()
            # data['img']: with shape num_views x 3 x H x W
            # token_masks: list with len=num_selection_layers, each entry of shape num_views x H x W x 1
            # img_metas[0]: dict
            #   'img_norm_cfg': dict
            if self.cur_id >= self.vis_start_id:
                print(f'Now vis id: {self.cur_id}')
                token_selection_vis(
                    input_imgs=data['img'],
                    masks=token_masks,
                    keep_idxes=keep_idxes,
                    drop_idxes=drop_idxes,
                    img_norm_cfg=img_metas[0]['img_norm_cfg'],
                    output_path=f'./{self.vis_out_path}/{self.cur_id}/'
                )
                self.vis_num_sample -= 1
            self.cur_id += 1

        GLOBAL_TIMER.event_start('StreamPETR/3D Transformer')

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            img_metas, **data)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        GLOBAL_TIMER.event_end('StreamPETR/3D Transformer')

        GLOBAL_TIMER.update_time_count()
        GLOBAL_TIMER.log()

        return bbox_list

    