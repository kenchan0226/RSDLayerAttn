# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoTokenizer

from volta.datasets import DatasetMapTrain, DatasetMapEval
from volta.datasets._image_features_reader import ImageFeaturesH5Reader
from volta.losses import InfoNCELoss, ListNetLoss

logger = logging.getLogger(__name__)

LossMap = {
    "BCEWithLogitLoss": nn.BCEWithLogitsLoss(reduction="mean"),
    "CrossEntropyLoss": nn.CrossEntropyLoss(),
    "InfoNCELoss": InfoNCELoss(),
    "ListNetLoss": ListNetLoss(),
    "InfoNCESequenceLabelLoss": {"region_classification": InfoNCELoss(), "sequence_labeling": nn.BCEWithLogitsLoss(reduction="mean")},
    "BCESequenceLabelLoss": {"region_classification": nn.BCEWithLogitsLoss(reduction="mean"), "sequence_labeling": nn.BCEWithLogitsLoss(reduction="mean")},
    "BCEInfoNCELoss": {"region_classification": nn.BCEWithLogitsLoss(reduction="mean"), "contrastive": InfoNCELoss()},
    "ListNetInfoNCELoss": {"region_classification": ListNetLoss(), "contrastive": InfoNCELoss()},
    "BCEListNet": {"region_classification": nn.BCEWithLogitsLoss(reduction="mean"), "contrastive": ListNetLoss()},
    "BCEInfoNCECELoss": {"region_classification": nn.BCEWithLogitsLoss(reduction="mean"), "contrastive": InfoNCELoss(), "object_categorization": nn.CrossEntropyLoss()}
}


def ForwardModelsVal(config, task_cfg, device, task_id, batch, model, criterion):
    batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

    if task_cfg[task_id]["type"] == "V-logit-mc":
        features, spatials, image_mask, question, target, input_mask, segment_ids, multi_choice_ids, question_id = batch
    elif task_cfg[task_id]["type"].startswith("VL-contrast") or task_cfg[task_id]["type"] == "V-logit" or task_cfg[task_id]["type"] == "VL-keywordmlp":
        features, spatials, spatials_ori, image_mask, question, target, input_mask, segment_ids, question_id = batch
    elif task_cfg[task_id]["type"].startswith("VL-seq-label"):
        features, spatials, spatials_ori, image_mask, question, target, input_mask, segment_ids, question_id, sequence_labels_target = batch
    elif task_cfg[task_id]["type"].startswith("VL-obj-categorize"):
        features, spatials, spatials_ori, image_mask, question, target, input_mask, segment_ids, question_id, ref_category_id = batch
    else:
        features, spatials, image_mask, question, target, input_mask, segment_ids, question_id = batch

    batch_size = features.size(0)
    if task_cfg[task_id]["process"] in ["dialog"]:
        raise NotImplementedError("dialog process for validation")

    elif task_cfg[task_id]["process"] in ["expand"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = (
            features.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, config.v_feature_size)
            .contiguous()
            .view(-1, max_num_bbox, config.v_feature_size)
        )
        spatials = (
            spatials.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, config.num_locs)
            .contiguous()
            .view(-1, max_num_bbox, config.num_locs)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))

    elif task_cfg[task_id]["process"] in ["retrieval"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(-1, features.size(2), features.size(3))
        spatials = spatials.view(-1, spatials.size(2), spatials.size(3))
        image_mask = image_mask.view(-1, image_mask.size(2))
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))

    elif task_cfg[task_id]["process"] in ["nlvr"]:
        batch_size = features.size(0)
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(batch_size * 2, int(features.size(1) / 2), features.size(2))
        spatials = spatials.view(batch_size * 2, int(spatials.size(1) / 2), spatials.size(2))
        image_mask = image_mask.view(batch_size * 2, int(image_mask.size(1) / 2))
        question = question.repeat(1, 2)
        question = question.view(batch_size * 2, int(question.size(1) / 2))
        input_mask = input_mask.repeat(1, 2)
        input_mask = input_mask.view(batch_size * 2, int(input_mask.size(1) / 2))
        segment_ids = segment_ids.repeat(1, 2)
        segment_ids = segment_ids.view(batch_size * 2, int(segment_ids.size(1) / 2))

    vil_prediction, vision_prediction, linguisic_prediction, _ = model(question, features, spatials, task_id,
                                                                       segment_ids, input_mask, image_mask)

    if task_cfg[task_id]["type"] == "VL-classifier":
        loss = criterion(vil_prediction, target)
        loss = loss.mean() * target.size(1)
        batch_score = compute_score_with_logits(vil_prediction, target).sum()

    elif task_cfg[task_id]["type"] == "VL-classifier-GQA":
        loss = criterion(vil_prediction, target)
        loss = loss.mean() * target.size(1)
        batch_score = compute_score_with_logits(vil_prediction, target).sum()

    elif task_cfg[task_id]["type"] == "VL-logit":
        vil_logit = vil_prediction.view(batch_size, num_options)
        loss = criterion(vil_logit, target)
        _, preds = torch.max(vil_logit, 1)
        batch_score = (preds == target).sum()

    elif task_cfg[task_id]["type"] == "V-logit" or task_cfg[task_id]["type"] == "VL-keywordmlp":
        if task_cfg[task_id]["loss"] == "BCEWithLogitLoss":
            loss = criterion(vil_prediction, target)
            loss = loss.mean() * target.size(1)
        elif task_cfg[task_id]["loss"] == "ListNetLoss":
            loss = criterion(vil_prediction, target, image_mask, task_cfg[task_id]["temperature"])
        _, select_idx = torch.max(vil_prediction, dim=1)
        #print(select_idx)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

    elif task_cfg[task_id]["type"] == "VL-seq-label":
        region_prediction, sequence_prediction = vil_prediction
        region_classification_loss = criterion["region_classification"](region_prediction, target)
        region_classification_loss = region_classification_loss.mean() * target.size(1)
        sequence_labeling_loss = criterion["sequence_labeling"](sequence_prediction, sequence_labels_target)
        loss = task_cfg[task_id]["region_loss_weight"] * region_classification_loss + task_cfg[task_id][
            "sequence_loss_weight"] * sequence_labeling_loss
        #loss = loss.mean() * target.size(1)

        _, select_idx = torch.max(region_prediction, dim=1)
        #print(select_idx)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

    elif task_cfg[task_id]["type"] == "VL-contrast":
        if task_cfg[task_id]["loss"] == "InfoNCELoss" or task_cfg[task_id]["loss"] == "ListNetLoss":
            loss = criterion(vil_prediction, target, image_mask, task_cfg[task_id]["temperature"])
        elif task_cfg[task_id]["loss"] == "BCEWithLogitLoss":
            loss = criterion(vil_prediction, target)
            loss = loss.mean() * target.size(1)
        _, select_idx = torch.max(vil_prediction, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

    elif task_cfg[task_id]["type"] == "VL-contrast-separated":
        pred_scores, sim_scores, attn_scores = vil_prediction
        contrastive_loss = criterion["contrastive"](sim_scores, target, image_mask, task_cfg[task_id]["temperature"])
        if task_cfg[task_id]["loss"] == "BCEInfoNCELoss" or task_cfg[task_id]["loss"] == "BCEListNet":
            region_classification_loss = criterion["region_classification"](pred_scores, target)
            region_classification_loss = region_classification_loss.mean() * target.size(1)
        elif task_cfg[task_id]["loss"] == "ListNetInfoNCELoss":
            region_classification_loss = criterion["region_classification"](pred_scores, target, image_mask, task_cfg[task_id]["listnet_temperature"])
        loss = task_cfg[task_id]["region_loss_weight"] * region_classification_loss + task_cfg[task_id][
            "contrast_loss_weight"] * contrastive_loss

        _, select_idx = torch.max(pred_scores, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

    elif task_cfg[task_id]["type"] == "VL-obj-categorize-contrast":
        pred_scores, sim_scores, tgt_obj_class_scores, attn_scores = vil_prediction
        # contrastive loss
        contrastive_loss = criterion["contrastive"](sim_scores, target, image_mask, task_cfg[task_id]["temperature"])
        # region classification loss
        if task_cfg[task_id]["loss"] == "BCEInfoNCELoss" or task_cfg[task_id]["loss"] == "BCEListNet":
            region_classification_loss = criterion["region_classification"](pred_scores, target)
            region_classification_loss = region_classification_loss.mean() * target.size(1)
        elif task_cfg[task_id]["loss"] == "ListNetInfoNCELoss":
            region_classification_loss = criterion["region_classification"](pred_scores, target, image_mask, task_cfg[task_id]["listnet_temperature"])
        # tgt object categorization loss
        tgt_object_categorization_loss = criterion["object_categorization"](tgt_obj_class_scores, ref_category_id)

        loss = task_cfg[task_id]["region_loss_weight"] * region_classification_loss + task_cfg[task_id][
            "contrast_loss_weight"] * contrastive_loss + task_cfg[task_id]["categorization_loss_weight"] * tgt_object_categorization_loss

        _, select_idx = torch.max(pred_scores, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

    elif task_cfg[task_id]["type"] == "VL-seq-label-contrast":
        region_prediction, sequence_prediction = vil_prediction
        region_classification_loss = criterion["region_classification"](region_prediction, target, image_mask, task_cfg[task_id]["temperature"])
        sequence_labeling_loss = criterion["sequence_labeling"](sequence_prediction, sequence_labels_target)
        loss = task_cfg[task_id]["region_loss_weight"] * region_classification_loss + task_cfg[task_id]["sequence_loss_weight"] * sequence_labeling_loss

        #loss = criterion(vil_prediction, target, image_mask, task_cfg[task_id]["temperature"])
        #print(loss)
        #exit()
        #loss = loss.mean() * target.size(1)
        _, select_idx = torch.max(region_prediction, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

    elif task_cfg[task_id]["type"] == "V-logit-mc":
        vision_logit = vil_prediction[:, 101:]  # FIXME from ViLBERT
        vision_logit = vision_logit.squeeze(2).gather(1, multi_choice_ids)
        vision_logit = vision_logit.unsqueeze(2)
        loss = criterion(vision_logit, target)
        loss = loss.mean() * target.size(1)
        _, preds = torch.max(vision_logit, dim=1)
        _, target = torch.max(target, dim=1)
        batch_score = (preds == target).sum()

    elif task_cfg[task_id]["type"] == "VL-binary-classifier":
        loss = criterion(vil_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_prediction, target).sum()

    elif task_cfg[task_id]["type"] == "VL-tri-classifier":
        loss = criterion(vil_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_prediction, target).sum()

    return float(loss), float(batch_score), batch_size


def ForwardModelsTrain(config, task_cfg, device, task_id, batch, model, criterion):
    batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

    if task_cfg[task_id]["type"] == "V-logit-mc":
        features, spatials, image_mask, question, target, input_mask, segment_ids, multi_choice_ids, question_id = batch
    elif task_cfg[task_id]["type"].startswith("VL-contrast") or task_cfg[task_id]["type"] == "V-logit" or task_cfg[task_id][
            "type"] == "VL-keywordmlp":
        features, spatials, spatials_ori, image_mask, question, target, input_mask, segment_ids, question_id = batch
    elif task_cfg[task_id]["type"].startswith("VL-seq-label"):
        features, spatials, spatials_ori, image_mask, question, target, input_mask, segment_ids, question_id, sequence_labels_target = batch
    elif task_cfg[task_id]["type"].startswith("VL-obj-categorize"):
        features, spatials, spatials_ori, image_mask, question, target, input_mask, segment_ids, question_id, ref_category_id = batch
        print("ref_category_id")
        print(ref_category_id.size())
        print(ref_category_id.detach().cpu().numpy())
    else:
        features, spatials, image_mask, question, target, input_mask, segment_ids, question_id = batch

    batch_size = features.size(0)
    if task_cfg[task_id]["process"] in ["dialog"]:
        max_num_bbox = features.size(1)
        nround = question.size(1)
        num_options = question.size(2)
        rbatch_size = batch_size * nround
        question = question.view(rbatch_size, question.size(2), question.size(3))
        target = target.view(-1)
        input_mask = input_mask.view(rbatch_size, input_mask.size(2), input_mask.size(3))
        segment_ids = segment_ids.view(rbatch_size, segment_ids.size(2), segment_ids.size(3))

        features = (
            features.unsqueeze(1)
            .unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox, config.v_feature_size)
            .contiguous()
            .view(-1, max_num_bbox, config.v_feature_size)
        )
        spatials = (
            spatials.unsqueeze(1)
            .unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox, config.num_locs)
            .contiguous()
            .view(-1, max_num_bbox, config.num_locs)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )

        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))
        batch_size = rbatch_size

    elif task_cfg[task_id]["process"] in ["expand"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = (
            features.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, config.v_feature_size)
            .contiguous()
            .view(-1, max_num_bbox, config.v_feature_size)
        )
        spatials = (
            spatials.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, config.num_locs)
            .contiguous()
            .view(-1, max_num_bbox, config.num_locs)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))

    elif task_cfg[task_id]["process"] in ["retrieval"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(-1, features.size(2), features.size(3))
        spatials = spatials.view(-1, spatials.size(2), spatials.size(3))
        image_mask = image_mask.view(-1, image_mask.size(2))
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))

    elif task_cfg[task_id]["process"] in ["nlvr"]:
        batch_size = features.size(0)
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(batch_size * 2, int(features.size(1) / 2), features.size(2))
        spatials = spatials.view(batch_size * 2, int(spatials.size(1) / 2), spatials.size(2))
        image_mask = image_mask.view(batch_size * 2, int(image_mask.size(1) / 2))
        question = question.repeat(1, 2)
        question = question.view(batch_size * 2, int(question.size(1) / 2))
        input_mask = input_mask.repeat(1, 2)
        input_mask = input_mask.view(batch_size * 2, int(input_mask.size(1) / 2))
        segment_ids = segment_ids.repeat(1, 2)
        segment_ids = segment_ids.view(batch_size * 2, int(segment_ids.size(1) / 2))

    vil_prediction, vision_prediction, linguisic_prediction, _ = model(question, features, spatials, task_id,
                                                                       segment_ids, input_mask, image_mask)
    # for different task, we use different output to calculate the loss.
    if task_cfg[task_id]["type"] == "VL-classifier":
        loss = criterion(vil_prediction, target)
        loss = loss.mean() * target.size(1)
        batch_score = compute_score_with_logits(vil_prediction, target).sum() / float(batch_size)

    elif task_cfg[task_id]["type"] == "VL-classifier-GQA":
        loss = criterion(vil_prediction, target)
        loss = loss.mean() * target.size(1)
        batch_score = compute_score_with_logits(vil_prediction, target).sum() / float(batch_size)

    elif task_cfg[task_id]["type"] == "VL-logit":
        vil_logit = vil_prediction.view(batch_size, num_options)
        loss = criterion(vil_logit, target)
        _, preds = torch.max(vil_logit, 1)
        batch_score = float((preds == target).sum()) / float(batch_size)

    elif task_cfg[task_id]["type"] == "V-logit" or task_cfg[task_id]["type"] == "VL-keywordmlp":
        if task_cfg[task_id]["loss"] == "BCEWithLogitLoss":
            loss = criterion(vil_prediction, target)
            loss = loss.mean() * target.size(1)
        elif task_cfg[task_id]["loss"] == "ListNetLoss":
            loss = criterion(vil_prediction, target, image_mask, task_cfg[task_id]["temperature"])
        _, select_idx = torch.max(vil_prediction, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = float(torch.sum(select_target > 0.5)) / batch_size

    elif task_cfg[task_id]["type"] == "VL-seq-label":
        region_prediction, sequence_prediction = vil_prediction
        region_classification_loss = criterion["region_classification"](region_prediction, target)
        #print("region_classification_loss")
        #print(region_classification_loss)
        region_classification_loss = region_classification_loss.mean() * target.size(1)
        #print("region_classification_loss")
        #print(region_classification_loss)
        sequence_labeling_loss = criterion["sequence_labeling"](sequence_prediction, sequence_labels_target)
        #print("sequence_labeling_loss")
        #print(sequence_labeling_loss)
        #exit()
        loss = task_cfg[task_id]["region_loss_weight"] * region_classification_loss + task_cfg[task_id][
            "sequence_loss_weight"] * sequence_labeling_loss

        _, select_idx = torch.max(region_prediction, dim=1)
        #print(select_idx)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

    elif task_cfg[task_id]["type"] == "VL-contrast":
        if task_cfg[task_id]["loss"] == "InfoNCELoss" or task_cfg[task_id]["loss"] == "ListNetLoss":
            loss = criterion(vil_prediction, target, image_mask, task_cfg[task_id]["temperature"])
        elif task_cfg[task_id]["loss"] == "BCEWithLogitLoss":
            loss = criterion(vil_prediction, target)
        #loss = loss.mean() * target.size(1)
        #print(loss)
        #exit()
        #loss = loss.mean() * target.size(1)
        _, select_idx = torch.max(vil_prediction, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()
    elif task_cfg[task_id]["type"] == "VL-contrast-separated":
        pred_scores, sim_scores, attn_scores = vil_prediction
        contrastive_loss = criterion["contrastive"](sim_scores, target, image_mask, task_cfg[task_id]["temperature"])
        if task_cfg[task_id]["loss"] == "BCEInfoNCELoss" or task_cfg[task_id]["loss"] == "BCEListNet":
            region_classification_loss = criterion["region_classification"](pred_scores, target)
            region_classification_loss = region_classification_loss.mean() * target.size(1)
        elif task_cfg[task_id]["loss"] == "ListNetInfoNCELoss":
            region_classification_loss = criterion["region_classification"](pred_scores, target, image_mask,
                                                   task_cfg[task_id]["listnet_temperature"])
        #logger.info("InfoNCE loss")
        #logger.info(contrastive_loss.item())
        #logger.info("BCE loss")
        #logger.info(region_classification_loss.item())
        #logger.info("")
        #sys.stdout.flush()
        loss = task_cfg[task_id]["region_loss_weight"] * region_classification_loss + task_cfg[task_id][
            "contrast_loss_weight"] * contrastive_loss

        _, select_idx = torch.max(pred_scores, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

    elif task_cfg[task_id]["type"] == "VL-obj-categorize-contrast":
        pred_scores, sim_scores, tgt_obj_class_scores, attn_scores = vil_prediction
        # contrastive loss
        contrastive_loss = criterion["contrastive"](sim_scores, target, image_mask, task_cfg[task_id]["temperature"])
        # region classification loss
        if task_cfg[task_id]["loss"] == "BCEInfoNCELoss" or task_cfg[task_id]["loss"] == "BCEListNet":
            region_classification_loss = criterion["region_classification"](pred_scores, target)
            region_classification_loss = region_classification_loss.mean() * target.size(1)
        elif task_cfg[task_id]["loss"] == "ListNetInfoNCELoss":
            region_classification_loss = criterion["region_classification"](pred_scores, target, image_mask, task_cfg[task_id]["listnet_temperature"])
        # tgt object categorization loss
        tgt_object_categorization_loss = criterion["object_categorization"](tgt_obj_class_scores, ref_category_id)

        print("region_classification_loss")
        print(region_classification_loss.item())
        print("contrast loss")
        print(contrastive_loss.item())
        print("tgt_object_categorization_loss")
        print(tgt_object_categorization_loss.item())
        exit()

        loss = task_cfg[task_id]["region_loss_weight"] * region_classification_loss + task_cfg[task_id][
            "contrast_loss_weight"] * contrastive_loss + task_cfg[task_id]["categorization_loss_weight"] * tgt_object_categorization_loss

        _, select_idx = torch.max(pred_scores, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

    elif task_cfg[task_id]["type"] == "VL-seq-label-contrast":
        region_prediction, sequence_prediction = vil_prediction
        region_classification_loss = criterion["region_classification"](region_prediction, target, image_mask, task_cfg[task_id]["temperature"])
        #print("region_classification_loss")
        #print(region_classification_loss)
        #print("sequence_prediction")
        #print(sequence_prediction.size())
        #print("sequence_labels_target")
        #print(sequence_labels_target.size())
        #print(sequence_labels_target[0].detach().cpu().numpy())
        #print(sequence_labels_target[1].detach().cpu().numpy())
        #print(sequence_labels_target[2].detach().cpu().numpy())
        sequence_labeling_loss = criterion["sequence_labeling"](sequence_prediction, sequence_labels_target)
        #print("sequence_labeling_loss")
        #print(sequence_labeling_loss)


        loss = task_cfg[task_id]["region_loss_weight"] * region_classification_loss + task_cfg[task_id]["sequence_loss_weight"] * sequence_labeling_loss
        #print("loss")
        #print(loss)
        #exit()
        #loss = criterion(vil_prediction, target, image_mask, task_cfg[task_id]["temperature"])
        #print(loss)
        #exit()
        #loss = loss.mean() * target.size(1)
        _, select_idx = torch.max(region_prediction, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

        # compute score for sequence_labeling:


    elif task_cfg[task_id]["type"] == "V-logit-mc":
        vision_logit = vil_prediction[:, 101:]  # FIXME from ViLBERT
        vision_logit = vision_logit.squeeze(2).gather(1, multi_choice_ids)
        vision_logit = vision_logit.unsqueeze(2)
        loss = criterion(vision_logit, target)
        loss = loss.mean() * target.size(1)
        _, preds = torch.max(vision_logit, dim=1)
        _, target = torch.max(target, dim=1)
        batch_score = float((preds == target).sum()) / float(batch_size)

    elif task_cfg[task_id]["type"] == "VL-binary-classifier":
        loss = criterion(vil_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_prediction, target).sum() / float(batch_size)

    elif task_cfg[task_id]["type"] == "VL-tri-classifier":
        loss = criterion(vil_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_prediction, target).sum() / float(batch_size)

    return loss, batch_score


def LoadLoss(task_cfg, task_id):
    task = "TASK" + task_id
    loss = LossMap[task_cfg[task]["loss"]]
    return loss


def LoadDataset(args, config, task_cfg, task_id, split="trainval"):
    tokenizer = AutoTokenizer.from_pretrained(config.bert_model, do_lower_case=config.do_lower_case)

    task = "TASK" + task_id
    task_name = task_cfg[task]["name"]

    # initialize the feature reader
    feats_h5path1 = task_cfg[task]["features_h5path1"]
    feats_h5path2 = task_cfg[task]["features_h5path2"]
    features_reader1 = ImageFeaturesH5Reader(feats_h5path1, config, args.in_memory) if feats_h5path1 != "" else None
    features_reader2 = ImageFeaturesH5Reader(feats_h5path2, config, args.in_memory) if feats_h5path2 != "" else None

    batch_size = task_cfg[task]["batch_size"] // args.grad_acc_steps
    num_workers = args.num_workers
    if args.local_rank != -1:
        batch_size = int(batch_size / dist.get_world_size())
        num_workers = int(num_workers / dist.get_world_size())

    logger.info("Loading %s Dataset with batch size %d" % (task_name, batch_size))
    dset_train, dset_train, task2num_iters = None, None, {}
    if "train" in split:
        dset_train = DatasetMapTrain[task_name](
            task=task_cfg[task]["name"],
            dataroot=task_cfg[task]["dataroot"],
            annotations_jsonpath=task_cfg[task]["train_annotations_jsonpath"],
            split=task_cfg[task]["train_split"],
            image_features_reader=features_reader1,
            gt_image_features_reader=features_reader2,
            tokenizer=tokenizer,
            bert_model=config.bert_model,
            padding_index=0,
            max_seq_length=task_cfg[task]["max_seq_length"],
            max_region_num=task_cfg[task]["max_region_num"],
            num_locs=config.num_locs,
            add_global_imgfeat=config.add_global_imgfeat,
            append_mask_sep=(config.fusion_method == 'vl-bert_vqa'),
        )
        if args.local_rank == -1:
            train_sampler = RandomSampler(dset_train)
        else:
            train_sampler = DistributedSampler(dset_train)
        dl_train = DataLoader(
            dset_train,
            sampler=train_sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=args.drop_last,
        )
        task2num_iters = {task: len(dl_train)}

    dset_val, dl_val = None, None
    if "val" in split:
        dset_val = DatasetMapTrain[task_name](
            task=task_cfg[task]["name"],
            dataroot=task_cfg[task]["dataroot"],
            annotations_jsonpath=task_cfg[task]["val_annotations_jsonpath"],
            split=task_cfg[task]["val_split"],
            image_features_reader=features_reader1,
            gt_image_features_reader=features_reader2,
            tokenizer=tokenizer,
            bert_model=config.bert_model,
            padding_index=0,
            max_seq_length=task_cfg[task]["max_seq_length"],
            max_region_num=task_cfg[task]["max_region_num"],
            num_locs=config.num_locs,
            add_global_imgfeat=config.add_global_imgfeat,
            append_mask_sep=(config.fusion_method == 'vl-bert_vqa'),
        )
        dl_val = DataLoader(
            dset_val,
            shuffle=False,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            drop_last=args.drop_last,
        )

    return batch_size, task2num_iters, dset_train, dset_val, dl_train, dl_val


def LoadDatasetEval(args, config, task_cfg, task_id):
    tokenizer = AutoTokenizer.from_pretrained(config.bert_model, do_lower_case=config.do_lower_case)

    task = "TASK" + task_id
    task_name = task_cfg[task]["name"]

    # initialize the feature reader
    feats_h5path1 = task_cfg[task]["features_h5path1"]
    feats_h5path2 = task_cfg[task]["features_h5path2"]
    features_reader1 = ImageFeaturesH5Reader(feats_h5path1, config, args.in_memory) if feats_h5path1 != "" else None
    features_reader2 = ImageFeaturesH5Reader(feats_h5path2, config, args.in_memory) if feats_h5path2 != "" else None

    batch_size = task_cfg[task].get("eval_batch_size", args.batch_size)
    if args.local_rank != -1:
        batch_size = int(batch_size / dist.get_world_size())

    logger.info("Loading %s Dataset with batch size %d" % (task_name, batch_size))
    if args.split:
        eval_split = args.split
    else:
        eval_split = task_cfg[task]["val_split"]

    if task_name.startswith("Retrieval"):
        dset_val = DatasetMapEval[task_name](
            task=task_cfg[task]["name"],
            dataroot=task_cfg[task]["dataroot"],
            annotations_jsonpath=task_cfg[task]["val_annotations_jsonpath"],
            split=eval_split,
            image_features_reader=features_reader1,
            gt_image_features_reader=features_reader2,
            tokenizer=tokenizer,
            bert_model=config.bert_model,
            padding_index=0,
            max_seq_length=task_cfg[task]["max_seq_length"],
            max_region_num=task_cfg[task]["max_region_num"],
            num_locs=config.num_locs,
            add_global_imgfeat=config.add_global_imgfeat,
            append_mask_sep=(config.fusion_method == 'vl-bert_vqa'),
            num_subiters=args.num_subiters,
        )
    else:
        dset_val = DatasetMapEval[task_name](
            task=task_cfg[task]["name"],
            dataroot=task_cfg[task]["dataroot"],
            annotations_jsonpath=task_cfg[task]["val_annotations_jsonpath"],
            split=eval_split,
            image_features_reader=features_reader1,
            gt_image_features_reader=features_reader2,
            tokenizer=tokenizer,
            bert_model=config.bert_model,
            padding_index=0,
            max_seq_length=task_cfg[task]["max_seq_length"],
            max_region_num=task_cfg[task]["max_region_num"],
            num_locs=config.num_locs,
            add_global_imgfeat=config.add_global_imgfeat,
            append_mask_sep=(config.fusion_method == 'vl-bert_vqa'),
        )

    dl_val = DataLoader(
        dset_val,
        shuffle=False,
        batch_size=batch_size,
        num_workers=10,
        pin_memory=True,
        drop_last=args.drop_last,
    )
    task2num_iters = {task: len(dl_val)}

    return batch_size, task2num_iters, dset_val, dl_val


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = one_hots * labels
    return scores

def compute_binary_sequence_label_score_with_logits(logits, labels):
    """
    :param logits: [batch, seq_len, 1]
    :param labels: [batch, seq_len, 1]
    :return:
    """
    predicted_prob = torch.sigmoid(logits.squeeze(2))
    prediction = (predicted_prob > 0.5).float()
    print("prediction")
    print(prediction.size())
    print(prediction[0])
    print("labels")
    print(labels[0,:,0])
    match_tensor = prediction == labels.squeeze(2)
    print("match_tensor")
    print(match_tensor.size())
    print(match_tensor[0])
    batch_score = match_tensor.sum(1).sum(0).item()
    return batch_score

def EvaluatingModel(config, task_cfg, device, task_id, batch, model, dataloader, criterion, results, others, bbox):
    batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

    if task_cfg[task_id]["type"] == "V-logit-mc":
        features, spatials, image_mask, question, target, input_mask, segment_ids, multi_choice_ids, question_id = batch
    elif task_cfg[task_id]["type"].startswith("VL-contrast") or task_cfg[task_id]["type"] == "V-logit" or task_cfg[task_id][
            "type"] == "VL-keywordmlp":
        features, spatials, spatials_ori, image_mask, question, target, input_mask, segment_ids, question_id = batch
    elif task_cfg[task_id]["type"].startswith("VL-seq-label"):
        features, spatials, spatials_ori, image_mask, question, target, input_mask, segment_ids, question_id, sequence_labels_target = batch
    elif task_cfg[task_id]["type"].startswith("VL-obj-categorize"):
        features, spatials, spatials_ori, image_mask, question, target, input_mask, segment_ids, question_id, ref_category_id = batch
    else:
        features, spatials, image_mask, question, target, input_mask, segment_ids, question_id = batch

    batch_size = features.size(0)

    if task_cfg[task_id]["process"] in ["dialog"]:
        max_num_bbox = features.size(1)
        nround = question.size(1)
        num_options = question.size(2)
        rbatch_size = batch_size * nround
        question = question.view(rbatch_size, question.size(2), question.size(3))
        target = target.view(-1)
        input_mask = input_mask.view(
            rbatch_size, input_mask.size(2), input_mask.size(3)
        )
        segment_ids = segment_ids.view(
            rbatch_size, segment_ids.size(2), segment_ids.size(3)
        )

        features = (
            features.unsqueeze(1)
            .unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox, config.v_feature_size)
            .contiguous()
            .view(-1, max_num_bbox, config.v_feature_size)
        )
        spatials = (
            spatials.unsqueeze(1)
            .unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox, config.num_locs)
            .contiguous()
            .view(-1, max_num_bbox, config.num_locs)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )

        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))
        batch_size = rbatch_size

    elif task_cfg[task_id]["process"] in ["expand"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = (
            features.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, config.v_feature_size)
            .contiguous()
            .view(-1, max_num_bbox, config.v_feature_size)
        )
        spatials = (
            spatials.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, config.num_locs)
            .contiguous()
            .view(-1, max_num_bbox, config.num_locs)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))

    elif task_cfg[task_id]["process"] in ["retrieval"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(-1, features.size(2), features.size(3))
        spatials = spatials.view(-1, spatials.size(2), spatials.size(3))
        image_mask = image_mask.view(-1, image_mask.size(2))
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))

    elif task_cfg[task_id]["process"] in ["nlvr"]:
        batch_size = features.size(0)
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(batch_size * 2, int(features.size(1) / 2), features.size(2))
        spatials = spatials.view(batch_size * 2, int(spatials.size(1) / 2), spatials.size(2))
        image_mask = image_mask.view(batch_size * 2, int(image_mask.size(1) / 2))
        question = question.repeat(1, 2)
        question = question.view(batch_size * 2, int(question.size(1) / 2))
        input_mask = input_mask.repeat(1, 2)
        input_mask = input_mask.view(batch_size * 2, int(input_mask.size(1) / 2))
        segment_ids = segment_ids.repeat(1, 2)
        segment_ids = segment_ids.view(batch_size * 2, int(segment_ids.size(1) / 2))

    with torch.no_grad():
        vil_prediction, vision_prediction, linguisic_prediction, _ = model(question, features, spatials, task_id,
                                                                           segment_ids, input_mask, image_mask)

    if task_cfg[task_id]["type"] == "VL-classifier":
        logits = torch.max(vil_prediction, 1)[1].data  # argmax
        loss = 0
        batch_score = 0
        for i in range(logits.size(0)):
            results.append(
                {
                    "question_id": question_id[i].item(),
                    "answer": dataloader.dataset.label2ans[logits[i].item()],
                }
            )

    elif task_cfg[task_id]["type"] == "VL-classifier-GQA":
        logits = torch.max(vil_prediction, 1)[1].data
        loss = 0
        batch_score = 0
        for i in range(logits.size(0)):
            results.append(
                {
                    "questionId": str(question_id[i].item()),
                    "prediction": dataloader.dataset.label2ans[logits[i].item()],
                }
            )

    elif task_cfg[task_id]["type"] == "VL-logit":
        vil_logit = vil_prediction.view(batch_size, num_options)
        loss = criterion(vil_logit, target)
        _, preds = torch.max(vil_logit, 1)
        batch_score = (preds == target).sum()

        probs = torch.softmax(vil_logit, dim=1)
        for i in range(vil_logit.size(0)):
            results.append(
                {
                    "question_id": question_id[i].item(),
                    "answer": [prob.item() for prob in probs[i]],
                }
            )

    elif task_cfg[task_id]["type"] == "VL-seq-label":
        region_prediction, sequence_prediction = vil_prediction
        region_classification_loss = criterion["region_classification"](region_prediction, target)
        region_classification_loss = region_classification_loss.mean() * target.size(1)
        sequence_labeling_loss = criterion["sequence_labeling"](sequence_prediction, sequence_labels_target)
        loss = task_cfg[task_id]["region_loss_weight"] * region_classification_loss + task_cfg[task_id][
            "sequence_loss_weight"] * sequence_labeling_loss

        _, select_idx = torch.max(region_prediction, dim=1)
        #print(select_idx)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

        for i in range(select_idx.size(0)):
            bbox_item = spatials_ori[i, select_idx[i],:4].cpu().detach().tolist()
            bbox_item = bbox_item[0]
            bbox.append(
              {
                question_id[i].item(): [bbox_item[0], bbox_item[1], bbox_item[2]-bbox_item[0], bbox_item[3]-bbox_item[1]]
                #question_id[i].item(): spatials_ori[i, select_idx[i],:4].cpu().detach().tolist()
              }
            )
            results.append(
                {
                    "id": question_id[i].item(),
                    "target": select_idx[i].item(),
                    "IOU": select_target[i].item(),
                }
            )

    elif task_cfg[task_id]["type"] == "V-logit" or task_cfg[task_id]["type"] == "VL-keywordmlp":
        if task_cfg[task_id]["loss"] == "BCEWithLogitLoss":
            loss = criterion(vil_prediction, target)
            loss = loss.mean() * target.size(1)
        elif task_cfg[task_id]["loss"] == "ListNetLoss":
            loss = criterion(vil_prediction, target, image_mask, task_cfg[task_id]["temperature"])
        #print()
        #print(vil_prediction.size())
        #print("vil_prediction")
        #print(vil_prediction[0,:,0])
        # vli_predition: [batch, num_regions, 1]
        _, select_idx = torch.max(vil_prediction, dim=1)
        #print(select_idx.size())
        #print("idx")
        #print(select_idx[0])
        #print("target")
        #print(target.size())
        #print("spatials_ori")
        #print(spatials_ori[select_idx[0],:4])
        #exit()
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

        for i in range(select_idx.size(0)):
            bbox_item = spatials_ori[i, select_idx[i],:4].cpu().detach().tolist()
            bbox_item = bbox_item[0]
            bbox.append(
              {
                question_id[i].item(): [bbox_item[0], bbox_item[1], bbox_item[2]-bbox_item[0], bbox_item[3]-bbox_item[1]]
                #question_id[i].item(): spatials_ori[i, select_idx[i],:4].cpu().detach().tolist()
              }
            )
            results.append(
                {
                    "id": question_id[i].item(),
                    "target": select_idx[i].item(),
                    "IOU": select_target[i].item(),
                }
            )

    elif task_cfg[task_id]["type"] == "VL-seq-label-contrast":
        #loss = criterion(vil_prediction, target)

        region_prediction, sequence_prediction = vil_prediction
        region_classification_loss = criterion["region_classification"](region_prediction, target, image_mask,
                                                                        task_cfg[task_id]["temperature"])
        sequence_labeling_loss = criterion["sequence_labeling"](sequence_prediction, sequence_labels_target)
        loss = task_cfg[task_id]["region_loss_weight"] * region_classification_loss + task_cfg[task_id][
            "sequence_loss_weight"] * sequence_labeling_loss

        loss = loss.mean() * target.size(1)
        _, select_idx = torch.max(region_prediction, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

        for i in range(select_idx.size(0)):
            bbox_item = spatials_ori[i, select_idx[i],:4].cpu().detach().tolist()
            bbox_item = bbox_item[0]
            bbox.append(
              {
                question_id[i].item(): [bbox_item[0], bbox_item[1], bbox_item[2]-bbox_item[0], bbox_item[3]-bbox_item[1]]
                #question_id[i].item(): spatials_ori[i, select_idx[i],:4].cpu().detach().tolist()
              }
            )
            results.append(
                {
                    "id": question_id[i].item(),
                    "target": select_idx[i].item(),
                    "IOU": select_target[i].item(),
                }
            )

    elif task_cfg[task_id]["type"] == "VL-contrast":
        if task_cfg[task_id]["loss"] == "InfoNCELoss" or task_cfg[task_id]["loss"] == "ListNetLoss":
            loss = criterion(vil_prediction, target, image_mask, task_cfg[task_id]["temperature"])
        elif task_cfg[task_id]["loss"] == "BCEWithLogitLoss":
            loss = criterion(vil_prediction, target)
        #loss = loss.mean() * target.size(1)
        _, select_idx = torch.max(vil_prediction, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

        for i in range(select_idx.size(0)):
            bbox_item = spatials_ori[i, select_idx[i],:4].cpu().detach().tolist()
            bbox_item = bbox_item[0]
            bbox.append(
              {
                question_id[i].item(): [bbox_item[0], bbox_item[1], bbox_item[2]-bbox_item[0], bbox_item[3]-bbox_item[1]]
              }
            )
            results.append(
                {
                    "id": question_id[i].item(),
                    "target": select_idx[i].item(),
                    "IOU": select_target[i].item(),
                }
            )
    elif task_cfg[task_id]["type"] == "VL-contrast-separated":
        pred_scores, sim_scores, attn_scores = vil_prediction
        contrastive_loss = criterion["contrastive"](sim_scores, target, image_mask, task_cfg[task_id]["temperature"])
        if task_cfg[task_id]["loss"] == "BCEInfoNCELoss" or task_cfg[task_id]["loss"] == "BCEListNet":
            region_classification_loss = criterion["region_classification"](pred_scores, target)
            region_classification_loss = region_classification_loss.mean() * target.size(1)
        elif task_cfg[task_id]["loss"] == "ListNetInfoNCELoss":
            region_classification_loss = criterion["region_classification"](pred_scores, target, image_mask,
                                                   task_cfg[task_id]["listnet_temperature"])
        loss = task_cfg[task_id]["region_loss_weight"] * region_classification_loss + task_cfg[task_id][
            "contrast_loss_weight"] * contrastive_loss

        _, select_idx = torch.max(pred_scores, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

        attn_scores = attn_scores.detach().cpu().tolist()

        # debug
        """
        for i in range(pred_scores.size(0)):
            print("target")
            print(target[i].detach().squeeze(1).cpu().numpy())
            print("pred_scores")
            print(pred_scores[i].detach().squeeze(1).cpu().numpy())
            print("sim_scores")
            print(sim_scores[i].detach().squeeze(1).cpu().numpy())
            print("region_classification_loss")
            print(region_classification_loss.item())
            print("contrastive_loss")
            print(contrastive_loss.item())
            print()
        exit()
        """

        for i in range(select_idx.size(0)):
            bbox_item = spatials_ori[i, select_idx[i],:4].cpu().detach().tolist()
            bbox_item = bbox_item[0]
            bbox.append(
              {
                question_id[i].item(): [bbox_item[0], bbox_item[1], bbox_item[2]-bbox_item[0], bbox_item[3]-bbox_item[1]]
              }
            )
            results.append(
                {
                    "id": question_id[i].item(),
                    "target": select_idx[i].item(),
                    "IOU": select_target[i].item(),
                    "attention_score": attn_scores[i]
                }
            )

    elif task_cfg[task_id]["type"] == "VL-obj-categorize-contrast":
        pred_scores, sim_scores, tgt_obj_class_scores, attn_scores = vil_prediction
        # contrastive loss
        contrastive_loss = criterion["contrastive"](sim_scores, target, image_mask, task_cfg[task_id]["temperature"])
        # region classification loss
        if task_cfg[task_id]["loss"] == "BCEInfoNCELoss" or task_cfg[task_id]["loss"] == "BCEListNet":
            region_classification_loss = criterion["region_classification"](pred_scores, target)
            region_classification_loss = region_classification_loss.mean() * target.size(1)
        elif task_cfg[task_id]["loss"] == "ListNetInfoNCELoss":
            region_classification_loss = criterion["region_classification"](pred_scores, target, image_mask, task_cfg[task_id]["listnet_temperature"])
        # tgt object categorization loss
        tgt_object_categorization_loss = criterion["object_categorization"](tgt_obj_class_scores, ref_category_id)

        loss = task_cfg[task_id]["region_loss_weight"] * region_classification_loss + task_cfg[task_id][
            "contrast_loss_weight"] * contrastive_loss + task_cfg[task_id]["categorization_loss_weight"] * tgt_object_categorization_loss

        _, select_idx = torch.max(pred_scores, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

        attn_scores = attn_scores.detach().cpu().tolist()

        # debug
        """
        for i in range(pred_scores.size(0)):
            print("target")
            print(target[i].detach().squeeze(1).cpu().numpy())
            print("pred_scores")
            print(pred_scores[i].detach().squeeze(1).cpu().numpy())
            print("sim_scores")
            print(sim_scores[i].detach().squeeze(1).cpu().numpy())
            print("region_classification_loss")
            print(region_classification_loss.item())
            print("contrastive_loss")
            print(contrastive_loss.item())
            print()
        exit()
        """

        for i in range(select_idx.size(0)):
            bbox_item = spatials_ori[i, select_idx[i], :4].cpu().detach().tolist()
            bbox_item = bbox_item[0]
            bbox.append(
                {
                    question_id[i].item(): [bbox_item[0], bbox_item[1], bbox_item[2] - bbox_item[0],
                                            bbox_item[3] - bbox_item[1]]
                }
            )
            results.append(
                {
                    "id": question_id[i].item(),
                    "target": select_idx[i].item(),
                    "IOU": select_target[i].item(),
                    "attention_score": attn_scores[i]
                }
            )

    elif task_cfg[task_id]["type"] == "V-logit-mc":
        vision_logit = vil_prediction[:, 101:]  # FIXME from ViLBERT
        vision_logit = vision_logit.squeeze(2).gather(1, multi_choice_ids)
        vision_logit = vision_logit.unsqueeze(2)
        loss = criterion(vision_logit, target)
        loss = loss.mean() * target.size(1)
        _, preds = torch.max(vision_logit, dim=1)
        _, target = torch.max(target, dim=1)
        batch_score = float((preds == target).sum())

        for i in range(preds.size(0)):
            results.append({"id": question_id[i].item(), "target": preds[i].item()})

    elif task_cfg[task_id]["type"] == "VL-binary-classifier":
        loss = criterion(vil_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_prediction, target).sum()

    elif task_cfg[task_id]["type"] == "VL-tri-classifier":
        loss = criterion(vil_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_prediction, target).sum()

    return float(loss), float(batch_score), batch_size, results, others, bbox
