import random

import torch
import torch.nn.functional as F
from torch import nn

from .pooler import RoIAlign
from .utils import Matcher, BalancedPositiveNegativeSampler, roi_align
from .box_ops import BoxCoder, box_iou, process_box, nms


def fastrcnn_loss(class_logit, box_regression, label, regression_target, trajectory_target, trajectory_regression, proposal=None, intermidiate_targets=None, box_coder=None):
    classifier_loss = F.cross_entropy(class_logit, label)

    N, num_pos = class_logit.shape[0], regression_target.shape[0]
    
    box_regression = box_regression.reshape(N, -1, 4)
    box_regression, label = box_regression[:num_pos], label[:num_pos]
    box_idx = torch.arange(num_pos, device=box_regression.device)

    box_reg_loss = F.smooth_l1_loss(box_regression[box_idx, label], regression_target, reduction='sum') / N

    intermidiate_loss = 0


    # print('trajectory_regression', len(trajectory_regression))

    for i, regression in enumerate(trajectory_regression):

        regression = regression.reshape(N, -1, 4)

        # print('regression', regression.shape)

        # print('regression', regression[box_idx, label].shape)
        # print('intermidiate_targets', intermidiate_targets[-1].shape)

        intermidiate_loss += F.smooth_l1_loss(regression[box_idx, label], intermidiate_targets[i], reduction='sum') / N

        if i == 0:
            current_proposal = box_coder.decode(regression[box_idx, label], proposal[box_idx])
        #     # print('current_proposal inloop', current_proposal.shape)
        else:
            current_proposal = box_coder.decode(regression[box_idx, label], current_proposal)
    

    # print('current_proposal', current_proposal.shape)
    # print('proposal', proposal.shape)
    final_target = box_coder.encode(current_proposal, proposal[box_idx])
    # print('final_target', final_target[box_idx, label].shape)
    # print('intermidiate_targets', intermidiate_targets[-1].shape)
    final_loss = F.smooth_l1_loss(final_target, intermidiate_targets[-1], reduction='sum') / N

    total_loss = box_reg_loss + intermidiate_loss + final_loss

    return classifier_loss, total_loss


def maskrcnn_loss(mask_logit, proposal, matched_idx, label, gt_mask):
    matched_idx = matched_idx[:, None].to(proposal)
    roi = torch.cat((matched_idx, proposal), dim=1)
            
    M = mask_logit.shape[-1]
    gt_mask = gt_mask[:, None].to(roi)
    mask_target = roi_align(gt_mask, roi, 1., M, M, -1)[:, 0]

    idx = torch.arange(label.shape[0], device=label.device)
    mask_loss = F.binary_cross_entropy_with_logits(mask_logit[idx, label], mask_target)
    return mask_loss
    

class RoIHeads(nn.Module):
    def __init__(self, box_roi_pool, box_predictor, trajectory_predictor,
                 fg_iou_thresh, bg_iou_thresh,
                 num_samples, positive_fraction,
                 reg_weights,
                 score_thresh, nms_thresh, num_detections):
        super().__init__()
        self.box_roi_pool = box_roi_pool
        self.box_predictor = box_predictor
        self.trajectory_predictor = trajectory_predictor
        
        self.mask_roi_pool = None
        self.mask_predictor = None
        
        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(num_samples, positive_fraction)
        self.box_coder = BoxCoder(reg_weights)
        
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.num_detections = num_detections
        self.min_size = 1
        
    def has_mask(self):
        if self.mask_roi_pool is None:
            return False
        if self.mask_predictor is None:
            return False
        return True
        
    def select_training_samples(self, proposal, target, future_target):
        gt_box = target['boxes']
        gt_future_box = []
        for ft in future_target:
            gt_future_box.append(ft['boxes'])
        # gt_future_box_two = future_target_two['boxes']
        gt_label = target['labels']
        proposal = torch.cat((proposal, gt_box))
        raw_proposal = proposal
        
        iou = box_iou(gt_box, proposal)
        pos_neg_label, matched_idx = self.proposal_matcher(iou)
        raw_matched_idx = matched_idx
        pos_idx, neg_idx = self.fg_bg_sampler(pos_neg_label)
        idx = torch.cat((pos_idx, neg_idx))

        trajectory_target = []
        
        regression_target = self.box_coder.encode(gt_box[matched_idx[pos_idx]], proposal[pos_idx])
        for ft in gt_future_box:
            trajectory_target.append(self.box_coder.encode(ft[matched_idx[pos_idx]], proposal[pos_idx]))
        # trajectory_target.append(self.box_coder.encode(gt_future_box_two[matched_idx[pos_idx]], proposal[pos_idx]))

        proposal = proposal[idx]
        matched_idx = matched_idx[idx]
        label = gt_label[matched_idx]
        num_pos = pos_idx.shape[0]
        label[num_pos:] = 0

        
        return proposal, matched_idx, label, regression_target, trajectory_target, pos_idx, raw_proposal, raw_matched_idx
    
    def fastrcnn_inference(self, class_logit, box_regression, proposal, image_shape, trajectory_regression):
        final_results = []

        N, num_classes = class_logit.shape
        
        device = class_logit.device
        pred_score = F.softmax(class_logit, dim=-1)
        box_regression = box_regression.reshape(N, -1, 4)
        regression = trajectory_regression[0].reshape(N, -1, 4)
        
        boxes = []
        future_boxes = [[] for _ in range(len(trajectory_regression))]
        labels = []
        scores = []
        for l in range(1, num_classes):
            score, box_delta, trajectory_delta = pred_score[:, l], box_regression[:, l], regression[:, l]

            keep = score >= self.score_thresh
            box, score, box_delta, trajectory_delta = proposal[keep], score[keep], box_delta[keep], trajectory_delta[keep]
            
            future_box = self.box_coder.decode(trajectory_delta, box)
            future_box_collection = [future_box]
            current_future_box = future_box
            for i in range(1, len(trajectory_regression)):
                tmp_regression = trajectory_regression[i].reshape(N, -1, 4)
                tmp_regression = tmp_regression[:, l][keep]
                current_future_box = self.box_coder.decode(tmp_regression, current_future_box)
                # current_future_box = self.box_coder.decode(tmp_regression, box)
                future_box_collection.append(current_future_box)

            box = self.box_coder.decode(box_delta, box)


            original_box, original_score = box, score

            for i in range(len(future_box_collection)):
                
                box, score, future_box = process_box(original_box, original_score, image_shape, self.min_size, future_box=future_box_collection[i])
                future_box_collection[i] = future_box
            
            keep = nms(box, score, self.nms_thresh)[:self.num_detections]
            box, score = box[keep], score[keep]
            for i in range(len(future_box_collection)):
                future_box_collection[i] = future_box_collection[i][keep]
            label = torch.full((len(keep),), l, dtype=keep.dtype, device=device)

            
            boxes.append(box)
            for i in range(len(future_box_collection)):
                future_boxes[i].append(future_box_collection[i])
            labels.append(label)
            scores.append(score)

        # print('boxes', len(boxes))
        # print('future_boxes item', len(future_boxes[0]))

        for i in range(len(trajectory_regression)):

            results = dict(boxes=torch.cat(boxes), labels=torch.cat(labels), scores=torch.cat(scores), 
                        future_boxes=torch.cat(future_boxes[i]))
            final_results.append(results)
        return final_results
    
    def forward(self, feature, proposal, image_shape, target, time=None):
        device_ = feature.get_device()
        frame_rate = 60./32
        if target is not None:
            time_list = list(range(1, len(target), 1))
            time_list = [random.choice(time_list)]
        else:
            time_list = time
        # random_future_time = [random.randint(1, len(target)-1)] if (target is not None) else time
        random_future_time = time_list
        time_step = torch.tensor([tm*frame_rate for tm in random_future_time]).to(device_)
        if self.training:
            # proposal, matched_idx, label, regression_target, trajectory_target, pos_idx, raw_proposal, raw_matched_idx = self.select_training_samples(proposal, target[0], 
                                                                                        # target[1:random_future_time[0]+1])

            proposal, matched_idx, label, regression_target, trajectory_target, pos_idx, raw_proposal, raw_matched_idx = self.select_training_samples(proposal, target[0], 
                                                                                        [target[1], target[random_future_time[0]]])

        box_feature = self.box_roi_pool(feature, proposal, image_shape)
        class_logit, box_regression = self.box_predictor(box_feature)

        # print('box_regression shape', box_regression)

        # new trajectory_target
        # if self.training:
        #     N = class_logit.shape[0]
        #     tm_box_regression = box_regression.reshape(N, -1, 4)
        #     num_pos = regression_target.shape[0]
        #     tm_box_regression, tmp_label = tm_box_regression[:num_pos], label[:num_pos]
        #     num_pos = torch.arange(num_pos, device=label.device)
        #     gt_future_box = target[random_future_time[0]]["boxes"]
        #     predicted_key_box = self.box_coder.decode(tm_box_regression[num_pos, tmp_label], raw_proposal[pos_idx])
        #     trajectory_target = self.box_coder.encode(gt_future_box[raw_matched_idx[pos_idx]], predicted_key_box)

        trajectory_regression = []
        if self.training:
            intermidiate_targets = [trajectory_target[0]]
        for tm in time_step:
            # tmp = self.trajectory_predictor(box_feature, box_regression, tm)
            # trajectory_regression.append(tmp)

            # sum distance method

            for cum_time in range(1, random_future_time[0]+1):
                tmp = self.trajectory_predictor(box_feature, box_regression, torch.tensor(cum_time*frame_rate).to(device_))
                if self.training:
                    if cum_time > 1:
                        tmp_gt_prev = target[cum_time-1]['boxes']
                        tmp_gt_curr = target[cum_time]['boxes']
                        tmp_target = self.box_coder.encode(tmp_gt_curr, tmp_gt_prev)
                        intermidiate_targets.append(tmp_target)
                trajectory_regression.append(tmp)
        if self.training:
            intermidiate_targets.append(trajectory_target[1])
        
        result, losses = {}, {}
        if self.training:
            # classifier_loss, box_reg_loss = fastrcnn_loss(class_logit, box_regression, label, 
            #                                                 regression_target, trajectory_target, trajectory_regression)
            classifier_loss, box_reg_loss = fastrcnn_loss(class_logit, box_regression, label, 
                                                            regression_target, trajectory_target, 
                                                            trajectory_regression, proposal=proposal, 
                                                            intermidiate_targets=intermidiate_targets, box_coder=self.box_coder)
            losses = dict(roi_classifier_loss=classifier_loss, roi_box_loss=box_reg_loss)
        else:
            result = self.fastrcnn_inference(class_logit, box_regression, proposal, image_shape, trajectory_regression)
            
        if self.has_mask():
            if self.training:
                num_pos = regression_target.shape[0]
                
                mask_proposal = proposal[:num_pos]
                pos_matched_idx = matched_idx[:num_pos]
                mask_label = label[:num_pos]
                
                '''
                # -------------- critial ----------------
                box_regression = box_regression[:num_pos].reshape(num_pos, -1, 4)
                idx = torch.arange(num_pos, device=mask_label.device)
                mask_proposal = self.box_coder.decode(box_regression[idx, mask_label], mask_proposal)
                # ---------------------------------------
                '''
                
                if mask_proposal.shape[0] == 0:
                    losses.update(dict(roi_mask_loss=torch.tensor(0)))
                    return result, losses
            else:
                mask_proposal = result['boxes']
                
                if mask_proposal.shape[0] == 0:
                    result.update(dict(masks=torch.empty((0, 28, 28))))
                    return result, losses
                
            mask_feature = self.mask_roi_pool(feature, mask_proposal, image_shape)
            mask_logit = self.mask_predictor(mask_feature)
            
            if self.training:
                gt_mask = target['masks']
                mask_loss = maskrcnn_loss(mask_logit, mask_proposal, pos_matched_idx, mask_label, gt_mask)
                losses.update(dict(roi_mask_loss=mask_loss))
            else:
                label = result['labels']
                idx = torch.arange(label.shape[0], device=label.device)
                mask_logit = mask_logit[idx, label]

                mask_prob = mask_logit.sigmoid()
                result.update(dict(masks=mask_prob))
                
        return result, losses
