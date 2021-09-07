# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import _pickle as cPickle

import numpy as np

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer
from ._image_features_reader import ImageFeaturesH5Reader

from tools.refer.refer import REFER, REFERTALK2CAR
from collections import defaultdict


def iou(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)
    K = gt_boxes.size(0)

    gt_boxes_area = (
        (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
    ).view(1, K)

    anchors_area = (
        (anchors[:, 2] - anchors[:, 0] + 1) * (anchors[:, 3] - anchors[:, 1] + 1)
    ).view(N, 1)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)
    #print(boxes)
    iw = (
        torch.min(boxes[:, :, 2], query_boxes[:, :, 2])
        - torch.max(boxes[:, :, 0], query_boxes[:, :, 0])
        + 1
    )
    iw[iw < 0] = 0

    ih = (
        torch.min(boxes[:, :, 3], query_boxes[:, :, 3])
        - torch.max(boxes[:, :, 1], query_boxes[:, :, 1])
        + 1
    )
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


class ReferExpressionDataset(Dataset):
    def __init__(
        self,
        task: str,
        dataroot: str,
        annotations_jsonpath: str,
        split: str,
        image_features_reader: ImageFeaturesH5Reader,
        gt_image_features_reader: ImageFeaturesH5Reader,
        tokenizer: AutoTokenizer,
        bert_model,
        padding_index: int = 0,
        max_seq_length: int = 20,
        max_region_num: int = 60,
        num_locs=5,
        add_global_imgfeat=None,
        append_mask_sep=False,
    ):
        self.split = split

        if task == "refcocog":
            self.refer = REFER(dataroot, dataset=task, splitBy="umd")
        elif task.startswith("talk2car"):
            print("using talk2car REFER data loader")
            self.refer = REFERTALK2CAR(dataroot)
        else:
            self.refer = REFER(dataroot, dataset=task, splitBy="unc")

        if self.split == "mteval":
            self.ref_ids = self.refer.getRefIds(split="train")
        else:
            self.ref_ids = self.refer.getRefIds(split=split)

        print("%s refs are in split [%s]." % (len(self.ref_ids), split))

        self.num_labels = 1
        self._image_features_reader = image_features_reader
        self._gt_image_features_reader = gt_image_features_reader
        self._tokenizer = tokenizer

        self._padding_index = padding_index
        self._max_seq_length = max_seq_length
        self.dataroot = dataroot
        self.entries = self._load_annotations()

        self._max_region_num = max_region_num + int(add_global_imgfeat is not None)
        self._num_locs = num_locs
        self._add_global_imgfeat = add_global_imgfeat

        if "roberta" in bert_model:
            cache_path = os.path.join(
                dataroot,
                "cache",
                task
                + "_"
                + split
                + "_"
                + "roberta"
                + "_"
                + str(max_seq_length)
                + "_"
                + str(max_region_num)
                + ".pkl",
            )
        else:
            cache_path = os.path.join(
                dataroot,
                "cache",
                task
                + "_"
                + split
                + "_"
                + str(max_seq_length)
                + "_"
                + str(max_region_num)
                + ".pkl",
            )

        if not os.path.exists(cache_path):
            self.tokenize()
            self.tensorize()
            cPickle.dump(self.entries, open(cache_path, "wb"))
        else:
            print("loading entries from %s" % (cache_path))
            self.entries = cPickle.load(open(cache_path, "rb"))

    def _load_annotations(self):
        # Build an index which maps image id with a list of caption annotations.
        entries = []
        remove_ids = []
        if self.split == "mteval":
            remove_ids = np.load(
                os.path.join(self.dataroot, "cache", "coco_test_ids.npy")
            )
            remove_ids = [int(x) for x in remove_ids]

        for ref_id in self.ref_ids:
            ref = self.refer.Refs[ref_id]
            image_id = ref["image_id"]
            if self.split == "train" and int(image_id) in remove_ids:
                continue
            elif self.split == "mteval" and int(image_id) not in remove_ids:
                continue
            ref_id = ref["ref_id"]
            refBox = self.refer.getRefBox(ref_id)
            for sent, sent_id in zip(ref["sentences"], ref["sent_ids"]):
                caption = sent["raw"]
                entries.append(
                    {
                        "caption": caption,
                        "sent_id": sent_id,
                        "image_id": image_id,
                        "refBox": refBox,
                        "ref_id": ref_id,
                    }
                )

        return entries

    def tokenize(self):
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.
        """
        for entry in self.entries:
            tokens = self._tokenizer.encode(entry["caption"])
            tokens = [tokens[0]] + tokens[1:-1][: self._max_seq_length - 2] + [tokens[-1]]

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < self._max_seq_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (self._max_seq_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), self._max_seq_length)
            entry["token"] = tokens
            entry["input_mask"] = input_mask
            entry["segment_ids"] = segment_ids

    def tensorize(self):
        for entry in self.entries:
            token = torch.from_numpy(np.array(entry["token"]))
            entry["token"] = token

            input_mask = torch.from_numpy(np.array(entry["input_mask"]))
            entry["input_mask"] = input_mask

            segment_ids = torch.from_numpy(np.array(entry["segment_ids"]))
            entry["segment_ids"] = segment_ids

    def __getitem__(self, index):
        entry = self.entries[index]

        image_id = entry["image_id"]
        ref_box = entry["refBox"]

        ref_box = [
            ref_box[0],
            ref_box[1],
            ref_box[0] + ref_box[2],
            ref_box[1] + ref_box[3],
        ]
        features, num_boxes, boxes, boxes_ori = self._image_features_reader[image_id]

        boxes_ori = boxes_ori[:num_boxes]
        boxes = boxes[:num_boxes]
        features = features[:num_boxes]

        mix_boxes_ori = boxes_ori
        mix_boxes = boxes
        mix_features = features
        mix_num_boxes = min(int(num_boxes), self._max_region_num)
        
        #print("index:{}, ref_box:{}, predict:{}".format(index, ref_box, mix_boxes_ori[:, :1]))
        
        mix_target = iou(
            torch.tensor(mix_boxes_ori[:, :4]).float(),
            torch.tensor([ref_box]).float(),
        )
        
        """
        bbox1 = mix_boxes_ori[torch.argmax(mix_target),:4].tolist()
        int_bbox1=[]
        for k in bbox1:
            k =int(k)
            int_bbox1.append(k)
        #print("index:{}, predict_box:{}".format(index,int_bbox1))
        print(index, int_bbox1)
        """
        
        image_mask = [1] * (mix_num_boxes)
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)
        mix_boxes_pad = np.zeros((self._max_region_num, self._num_locs))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        mix_boxes_pad[:mix_num_boxes] = mix_boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = mix_features[:mix_num_boxes]
                
        # pad mix box ori
        mix_boxes_ori_pad = np.zeros((self._max_region_num, self._num_locs))
        mix_boxes_ori_pad[:mix_num_boxes]  = mix_boxes_ori[:mix_num_boxes]
        
        # appending the target feature.
        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()
        
        
        target = torch.zeros((self._max_region_num, 1)).float()
        target[:mix_num_boxes] = mix_target[:mix_num_boxes]
        
        #bbox2 = mix_boxes_ori[torch.argmax(target[:mix_num_boxes]),:4].tolist()
        
        spatials_ori = torch.tensor(mix_boxes_ori_pad).float()
        #print(spatials_ori)
        caption = entry["token"]
        input_mask = entry["input_mask"]
        segment_ids = entry["segment_ids"]
        
        #print("spatials")
        #print(spatials[1,:4])
        #print("mix_boxes")
        #print(mix_boxes[1,:4])
        #print("spatials_ori")
        #print(spatials_ori[1,:4])
        #print("mix_boxes_ori")
        #print(mix_boxes_ori[1,:4])
        #exit()

        return features, spatials, spatials_ori, image_mask, caption, target, input_mask, segment_ids, image_id

    def __len__(self):
        return len(self.entries)


class ReferExpressionTargetObjCategorizationDataset(ReferExpressionDataset):
    def __init__(
            self,
            task: str,
            dataroot: str,
            annotations_jsonpath: str,
            split: str,
            image_features_reader: ImageFeaturesH5Reader,
            gt_image_features_reader: ImageFeaturesH5Reader,
            tokenizer: AutoTokenizer,
            bert_model,
            padding_index: int = 0,
            max_seq_length: int = 20,
            max_region_num: int = 60,
            num_locs=5,
            add_global_imgfeat=None,
            append_mask_sep=False,
    ):
        super(ReferExpressionTargetObjCategorizationDataset, self).__init__(task, dataroot, annotations_jsonpath, split,
                                                                  image_features_reader, gt_image_features_reader,
                                                                  tokenizer,
                                                                  bert_model, padding_index, max_seq_length,
                                                                  max_region_num,
                                                                  num_locs, add_global_imgfeat, append_mask_sep)

        print("ReferExpressionObjClassificationDataset built")

    def _load_annotations(self):
        # Build an index which maps image id with a list of caption annotations.
        entries = []
        remove_ids = []
        if self.split == "mteval":
            remove_ids = np.load(
                os.path.join(self.dataroot, "cache", "coco_test_ids.npy")
            )
            remove_ids = [int(x) for x in remove_ids]

        for ref_id in self.ref_ids:
            ref = self.refer.Refs[ref_id]
            image_id = ref["image_id"]
            if self.split == "train" and int(image_id) in remove_ids:
                continue
            elif self.split == "mteval" and int(image_id) not in remove_ids:
                continue
            ref_id = ref["ref_id"]
            refBox = self.refer.getRefBox(ref_id)
            ref_ann = self.refer.refToAnn[ref_id]
            for sent, sent_id in zip(ref["sentences"], ref["sent_ids"]):
                caption = sent["raw"]
                entries.append(
                    {
                        "caption": caption,
                        "sent_id": sent_id,
                        "image_id": image_id,
                        "refBox": refBox,
                        #"ref_category_name": ref_ann["category_name"],
                        "ref_category_id": [ref_ann["category_id"]],
                        "ref_id": ref_id,
                    }
                )

        return entries

    def tensorize(self):
        for entry in self.entries:
            token = torch.from_numpy(np.array(entry["token"]))
            entry["token"] = token

            input_mask = torch.from_numpy(np.array(entry["input_mask"]))
            entry["input_mask"] = input_mask

            segment_ids = torch.from_numpy(np.array(entry["segment_ids"]))
            entry["segment_ids"] = segment_ids

            # tensorize sequence label
            ref_category_id = torch.from_numpy(np.array(entry["ref_category_id"]))
            entry["ref_category_id"] = ref_category_id


    def __getitem__(self, index):
        entry = self.entries[index]

        image_id = entry["image_id"]
        ref_box = entry["refBox"]

        ref_box = [
            ref_box[0],
            ref_box[1],
            ref_box[0] + ref_box[2],
            ref_box[1] + ref_box[3],
        ]
        features, num_boxes, boxes, boxes_ori = self._image_features_reader[image_id]

        boxes_ori = boxes_ori[:num_boxes]
        boxes = boxes[:num_boxes]
        features = features[:num_boxes]

        mix_boxes_ori = boxes_ori
        mix_boxes = boxes
        mix_features = features
        mix_num_boxes = min(int(num_boxes), self._max_region_num)

        # print("index:{}, ref_box:{}, predict:{}".format(index, ref_box, mix_boxes_ori[:, :1]))

        mix_target = iou(
            torch.tensor(mix_boxes_ori[:, :4]).float(),
            torch.tensor([ref_box]).float(),
        )

        """
        bbox1 = mix_boxes_ori[torch.argmax(mix_target),:4].tolist()
        int_bbox1=[]
        for k in bbox1:
            k =int(k)
            int_bbox1.append(k)
        #print("index:{}, predict_box:{}".format(index,int_bbox1))
        print(index, int_bbox1)
        """

        image_mask = [1] * (mix_num_boxes)
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)
        mix_boxes_pad = np.zeros((self._max_region_num, self._num_locs))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        mix_boxes_pad[:mix_num_boxes] = mix_boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = mix_features[:mix_num_boxes]

        # pad mix box ori
        mix_boxes_ori_pad = np.zeros((self._max_region_num, self._num_locs))
        mix_boxes_ori_pad[:mix_num_boxes] = mix_boxes_ori[:mix_num_boxes]

        # appending the target feature.
        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        target = torch.zeros((self._max_region_num, 1)).float()
        target[:mix_num_boxes] = mix_target[:mix_num_boxes]

        # bbox2 = mix_boxes_ori[torch.argmax(target[:mix_num_boxes]),:4].tolist()

        spatials_ori = torch.tensor(mix_boxes_ori_pad).float()
        # print(spatials_ori)
        caption = entry["token"]
        input_mask = entry["input_mask"]
        segment_ids = entry["segment_ids"]
        ref_category_id = entry["ref_category_id"]

        # print("spatials")
        # print(spatials[1,:4])
        # print("mix_boxes")
        # print(mix_boxes[1,:4])
        # print("spatials_ori")
        # print(spatials_ori[1,:4])
        # print("mix_boxes_ori")
        # print(mix_boxes_ori[1,:4])
        # exit()

        return features, spatials, spatials_ori, image_mask, caption, target, input_mask, segment_ids, image_id, ref_category_id


class ReferExpressionSequenceLabelDataset(ReferExpressionDataset):
    def __init__(
        self,
        task: str,
        dataroot: str,
        annotations_jsonpath: str,
        split: str,
        image_features_reader: ImageFeaturesH5Reader,
        gt_image_features_reader: ImageFeaturesH5Reader,
        tokenizer: AutoTokenizer,
        bert_model,
        padding_index: int = 0,
        max_seq_length: int = 20,
        max_region_num: int = 60,
        num_locs=5,
        add_global_imgfeat=None,
        append_mask_sep=False,
    ):

        self.sequence_label_to_id = defaultdict(int)  # the default value is 0
        self.sequence_label_to_id["PROPN"] = 1
        self.sequence_label_to_id["NOUN"] = 1
        self.sequence_label_to_id["ADJ"] = 1
        #self.sequence_label_to_id["DET"] = 1
        #self.sequence_label_to_id["ADP"] = 1
        # PROPN, NOUN, ADJ, ADV, DET, ADP

        super(ReferExpressionSequenceLabelDataset, self).__init__(task, dataroot, annotations_jsonpath, split,
                                                            image_features_reader, gt_image_features_reader, tokenizer,
                                                            bert_model, padding_index, max_seq_length, max_region_num,
                                                            num_locs, add_global_imgfeat, append_mask_sep)

        print("ReferExpressionSequenceLabelDataset built")

    def _load_annotations(self):
        # Build an index which maps image id with a list of caption annotations.
        entries = []
        remove_ids = []
        if self.split == "mteval":
            remove_ids = np.load(
                os.path.join(self.dataroot, "cache", "coco_test_ids.npy")
            )
            remove_ids = [int(x) for x in remove_ids]

        for ref_id in self.ref_ids:
            ref = self.refer.Refs[ref_id]
            image_id = ref["image_id"]
            if self.split == "train" and int(image_id) in remove_ids:
                continue
            elif self.split == "mteval" and int(image_id) not in remove_ids:
                continue
            ref_id = ref["ref_id"]
            refBox = self.refer.getRefBox(ref_id)
            for sent, sent_id in zip(ref["sentences"], ref["sent_ids"]):
                caption = sent["raw"]
                sequence_labels = sent["pos_labels_simple"]
                tokenized_sent = sent["spacy_tokenized_sent"]
                entries.append(
                    {
                        "caption": caption,
                        "tokenized_sent": tokenized_sent,
                        "sequence_labels": sequence_labels,
                        "sent_id": sent_id,
                        "image_id": image_id,
                        "refBox": refBox,
                        "ref_id": ref_id,
                    }
                )

        return entries

    # Tokenize all texts and align the labels with them.
    def tokenize(self):
        for entry in self.entries:
            # We use is_split_into_words because the texts in our dataset are lists of words (with a label for each word).
            transformers_tokenized_sent = self._tokenizer(entry["tokenized_sent"], padding=False, is_split_into_words=True)
            label = entry["sequence_labels"]
            assert len(entry["tokenized_sent"]) == len(entry["sequence_labels"])
            #print("raw")
            #print(entry["caption"])
            #print("tokenized_sent")
            #print(entry["tokenized_sent"])
            #print("label")
            #print(label)
            # construct label ids
            # Adapted from Huggingface Transformers
            word_ids = transformers_tokenized_sent.word_ids()
            #print("word_ids")
            #print(word_ids)
            #print("tokens")
            #print(transformers_tokenized_sent["input_ids"])
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(self.sequence_label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(self.sequence_label_to_id[label[word_idx]])
                    #label_ids.append(label_to_id[label[word_idx]] if data_args.label_all_tokens else -100)
                previous_word_idx = word_idx
            #print("label_ids")
            #print(label_ids)
            tokens = transformers_tokenized_sent["input_ids"]
            #print("tokens")
            #print(tokens)
            #exit()

            # truncate to max len
            tokens = [tokens[0]] + tokens[1:-1][: self._max_seq_length - 2] + [tokens[-1]]
            label_ids = [label_ids[0]] + label_ids[1:-1][: self._max_seq_length - 2] + [label_ids[-1]]

            assert len(tokens) == len(label_ids)

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            # padding
            if len(tokens) < self._max_seq_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (self._max_seq_length - len(tokens))
                tokens = tokens + padding
                # Pad label
                label_padding = [-100] * (self._max_seq_length - len(label_ids))
                label_ids = label_ids + label_padding

                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), self._max_seq_length)
            entry["token"] = tokens
            entry["input_mask"] = input_mask
            entry["segment_ids"] = segment_ids
            entry["sequence_label_ids"] = label_ids

    def tensorize(self):
        for entry in self.entries:
            token = torch.from_numpy(np.array(entry["token"]))
            entry["token"] = token

            input_mask = torch.from_numpy(np.array(entry["input_mask"]))
            entry["input_mask"] = input_mask

            segment_ids = torch.from_numpy(np.array(entry["segment_ids"]))
            entry["segment_ids"] = segment_ids

            # tensorize sequence label
            sequence_label_ids = torch.from_numpy(np.array(entry["sequence_label_ids"]))
            entry["sequence_label_ids"] = sequence_label_ids

    def __getitem__(self, index):
        entry = self.entries[index]

        image_id = entry["image_id"]
        ref_box = entry["refBox"]

        ref_box = [
            ref_box[0],
            ref_box[1],
            ref_box[0] + ref_box[2],
            ref_box[1] + ref_box[3],
        ]
        features, num_boxes, boxes, boxes_ori = self._image_features_reader[image_id]

        boxes_ori = boxes_ori[:num_boxes]
        boxes = boxes[:num_boxes]
        features = features[:num_boxes]

        mix_boxes_ori = boxes_ori
        mix_boxes = boxes
        mix_features = features
        mix_num_boxes = min(int(num_boxes), self._max_region_num)

        # print("index:{}, ref_box:{}, predict:{}".format(index, ref_box, mix_boxes_ori[:, :1]))

        mix_target = iou(
            torch.tensor(mix_boxes_ori[:, :4]).float(),
            torch.tensor([ref_box]).float(),
        )

        """
        bbox1 = mix_boxes_ori[torch.argmax(mix_target),:4].tolist()
        int_bbox1=[]
        for k in bbox1:
            k =int(k)
            int_bbox1.append(k)
        #print("index:{}, predict_box:{}".format(index,int_bbox1))
        print(index, int_bbox1)
        """

        image_mask = [1] * (mix_num_boxes)
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)
        mix_boxes_pad = np.zeros((self._max_region_num, self._num_locs))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        mix_boxes_pad[:mix_num_boxes] = mix_boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = mix_features[:mix_num_boxes]

        # pad mix box ori
        mix_boxes_ori_pad = np.zeros((self._max_region_num, self._num_locs))
        mix_boxes_ori_pad[:mix_num_boxes] = mix_boxes_ori[:mix_num_boxes]

        # appending the target feature.
        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        target = torch.zeros((self._max_region_num, 1)).float()
        target[:mix_num_boxes] = mix_target[:mix_num_boxes]

        # bbox2 = mix_boxes_ori[torch.argmax(target[:mix_num_boxes]),:4].tolist()

        spatials_ori = torch.tensor(mix_boxes_ori_pad).float()
        # print(spatials_ori)
        caption = entry["token"]
        input_mask = entry["input_mask"]
        segment_ids = entry["segment_ids"]
        sequence_labels = entry["sequence_label_ids"].float().unsqueeze(1)

        # print("spatials")
        # print(spatials[1,:4])
        # print("mix_boxes")
        # print(mix_boxes[1,:4])
        # print("spatials_ori")
        # print(spatials_ori[1,:4])
        # print("mix_boxes_ori")
        # print(mix_boxes_ori[1,:4])
        # exit()

        return features, spatials, spatials_ori, image_mask, caption, target, input_mask, segment_ids, image_id, sequence_labels
