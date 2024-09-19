from typing import List
import numpy as np
import os.path as osp
import torch
import mmcv


def token_selection_vis(
    input_imgs: torch.Tensor,
    masks: List[torch.Tensor],
    keep_idxes: List[torch.Tensor],
    drop_idxes: List[torch.Tensor],
    img_norm_cfg: dict,
    output_path: str,
    patch_size: int = 16,
    min_alpha: float = 0.3,
    max_alpha: float = 1.0
):
    '''visualization the token selection

    Args:
        input_imgs (torch.Tensor): with shape num_views x 3 x H x W
        masks (List[Tensor]): the token masks, 1 for important tokens, with shape num_views x H x W x 1
        img_norm_cfg (dict): the norm_cfg used to denorm the input_imgs
        output_path (str): the output_path of vis
    '''
    input_imgs = input_imgs.cpu().numpy()
    masks = [mask.cpu().numpy() for mask in masks]
    if keep_idxes is not None and drop_idxes is not None:
        keep_idxes = [idxes.cpu().numpy() for idxes in keep_idxes]
        drop_idxes = [idxes.cpu().numpy() for idxes in drop_idxes]
    for view_id in range(input_imgs.shape[0]):
        img = input_imgs[view_id]  
        img = np.transpose(img, [1, 2, 0])  # H x W x 3

        # denorm the img
        if img_norm_cfg is not None:
            mean = img_norm_cfg['mean']
            std = img_norm_cfg['std']

            # import ipdb; ipdb.set_trace()
            img = mmcv.imdenormalize(img, mean=mean, std=std, to_bgr=False)  # np.float32, 0~255.0

        alpha = np.ones([*img.shape[:2], 1], dtype=img.dtype) * 255
        for layer_id, layer_mask in enumerate(masks):
            layer_mask = layer_mask[view_id]  # Hp x Wp x 1
            assert img.shape[0] // layer_mask.shape[0] == patch_size
            assert img.shape[1] // layer_mask.shape[1] == patch_size

            pixel_mask = np.repeat(layer_mask, patch_size, axis=1)
            pixel_mask = np.repeat(pixel_mask, patch_size, axis=0)  # H x W x 1
            
            pixel_mask = pixel_mask * (max_alpha - min_alpha) + min_alpha

            out_img = np.concatenate([img, alpha * pixel_mask], axis=-1)
            out_filename = osp.join(output_path, f'view{view_id}_layer{layer_id}.png')
            ori_filename = osp.join(output_path, f'view{view_id}_layer{layer_id}_ori.png')
            mmcv.imwrite(out_img, out_filename)
            mmcv.imwrite(img, ori_filename)

        if keep_idxes is not None and drop_idxes is not None:
            num_patch_h = img.shape[0] // patch_size
            num_patch_w = img.shape[1] // patch_size
            alpha = np.ones([*img.shape[:2], 1], dtype=img.dtype) * 255
            for layer_id, layer_keep_idx in enumerate(keep_idxes):
                keep_idx = layer_keep_idx[view_id]
                assert keep_idx.max() < num_patch_h * num_patch_w

                keep_mask = np.zeros([num_patch_h * num_patch_w], dtype=img.dtype)
                keep_mask[keep_idx] = 1
                keep_mask = keep_mask.reshape(num_patch_h, num_patch_w, 1)

                keep_mask = np.repeat(keep_mask, patch_size, axis=1)
                keep_mask = np.repeat(keep_mask, patch_size, axis=0)

                keep_mask = keep_mask * (max_alpha - min_alpha) + min_alpha

                out_img = np.concatenate([img, alpha * keep_mask], axis=-1)
                out_filename = osp.join(output_path, f'view{view_id}_layer{layer_id}_keepidx.png')
                mmcv.imwrite(out_img, out_filename)