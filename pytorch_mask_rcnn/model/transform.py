import math

import torch
import torch.nn.functional as F


class Transformer:
    def __init__(self, min_size, max_size, image_mean, image_std):
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std
        
    def __call__(self, image, target):
        image = self.normalize(image)
        image, target = self.resize(image, target)
        image = self.batched_image(image)
        
        return image, target

    def normalize(self, image):
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        
        dtype, device = image.dtype, image.device
        mean = torch.tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def resize(self, image, target):
        ori_image_shape = image.shape[-2:]
        min_size = float(min(image.shape[-2:]))
        max_size = float(max(image.shape[-2:]))
        
        scale_factor = min(self.min_size / min_size, self.max_size / max_size)
        size = [round(s * scale_factor) for s in ori_image_shape]
        image = F.interpolate(image[None], size=size, mode='bilinear', align_corners=False)[0]

        if target is None:
            return image, target

        if isinstance(target, list):
            for num in range(len(target)):
                box = target[num]['boxes']
                box[:, [0, 2]] = box[:, [0, 2]] * image.shape[-1] / ori_image_shape[1]
                box[:, [1, 3]] = box[:, [1, 3]] * image.shape[-2] / ori_image_shape[0]
                target[num]['boxes'] = box
            
                if 'masks' in target[num]:
                    mask = target[num]['masks']
                    mask = F.interpolate(mask[None].float(), size=size)[0].byte()
                    target[num]['masks'] = mask
        else:
            box = target['boxes']
            box[:, [0, 2]] = box[:, [0, 2]] * image.shape[-1] / ori_image_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] * image.shape[-2] / ori_image_shape[0]
            target[num]['boxes'] = box
            
            if 'masks' in target:
                mask = target['masks']
                mask = F.interpolate(mask[None].float(), size=size)[0].byte()
                target[num]['masks'] = mask
            
        return image, target
    
    def batched_image(self, image, stride=32):
        size = image.shape[-2:]
        max_size = tuple(math.ceil(s / stride) * stride for s in size)

        batch_shape = (image.shape[-3],) + max_size
        batched_img = image.new_full(batch_shape, 0)
        batched_img[:, :image.shape[-2], :image.shape[-1]] = image

        return batched_img[None]
    
    def postprocess(self, result, image_shape, ori_image_shape):
        if isinstance(result, list):
            for i, res in enumerate(result):
                box = res['boxes']
                future_box = res['future_boxes']
                box[:, [0, 2]] = box[:, [0, 2]] * ori_image_shape[1] / image_shape[1]
                box[:, [1, 3]] = box[:, [1, 3]] * ori_image_shape[0] / image_shape[0]
                future_box[:, [0, 2]] = future_box[:, [0, 2]] * ori_image_shape[1] / image_shape[1]
                future_box[:, [1, 3]] = future_box[:, [1, 3]] * ori_image_shape[0] / image_shape[0]
                result[i]['boxes'] = box
                result[i]['future_boxes'] = future_box
                
                if 'masks' in res:
                    mask = res['masks']
                    mask = paste_masks_in_image(mask, box, 1, ori_image_shape)
                    result[i]['masks'] = mask
        else:
            box = result['boxes']
            future_box = result['future_boxes']
            box[:, [0, 2]] = box[:, [0, 2]] * ori_image_shape[1] / image_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] * ori_image_shape[0] / image_shape[0]
            future_box[:, [0, 2]] = future_box[:, [0, 2]] * ori_image_shape[1] / image_shape[1]
            future_box[:, [1, 3]] = future_box[:, [1, 3]] * ori_image_shape[0] / image_shape[0]
            result['boxes'] = box
            result['future_boxes'] = future_box
            
            if 'masks' in result:
                mask = result['masks']
                mask = paste_masks_in_image(mask, box, 1, ori_image_shape)
                result['masks'] = mask
            
        return result


def expand_detection(mask, box, padding):
    M = mask.shape[-1]
    scale = (M + 2 * padding) / M
    padded_mask = torch.nn.functional.pad(mask, (padding,) * 4)
    
    w_half = (box[:, 2] - box[:, 0]) * 0.5
    h_half = (box[:, 3] - box[:, 1]) * 0.5
    x_c = (box[:, 2] + box[:, 0]) * 0.5
    y_c = (box[:, 3] + box[:, 1]) * 0.5

    w_half = w_half * scale
    h_half = h_half * scale

    box_exp = torch.zeros_like(box)
    box_exp[:, 0] = x_c - w_half
    box_exp[:, 2] = x_c + w_half
    box_exp[:, 1] = y_c - h_half
    box_exp[:, 3] = y_c + h_half
    return padded_mask, box_exp.to(torch.int64)


def paste_masks_in_image(mask, box, padding, image_shape):
    mask, box = expand_detection(mask, box, padding)
    
    N = mask.shape[0]
    size = (N,) + tuple(image_shape)
    im_mask = torch.zeros(size, dtype=mask.dtype, device=mask.device)
    for m, b, im in zip(mask, box, im_mask):
        b = b.tolist()
        w = max(b[2] - b[0], 1)
        h = max(b[3] - b[1], 1)
        
        m = F.interpolate(m[None, None], size=(h, w), mode='bilinear', align_corners=False)[0][0]

        x1 = max(b[0], 0)
        y1 = max(b[1], 0)
        x2 = min(b[2], image_shape[1])
        y2 = min(b[3], image_shape[0])

        im[y1:y2, x1:x2] = m[(y1 - b[1]):(y2 - b[1]), (x1 - b[0]):(x2 - b[0])]
    return im_mask