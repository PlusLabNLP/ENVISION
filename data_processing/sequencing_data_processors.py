import os
import re
import sys
import csv
import json
import glob
import string
from tqdm import tqdm
import random
import numpy as np
import glob
import shutil

from transformers import AutoTokenizer

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

import cv2
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

import logging
from enum import Enum
from typing import List, Optional, Union
from dataclasses import dataclass


WIKIHOW_DATA_ROOT = "data/wikihow"
IMAGE_FIELD_NAMES = [
    "image-large",
    "image-src-1",
]


class DataProcessor:
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


@dataclass
class MultimodalWikiHowExample:
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        text_seq: list of strings. The untokenized text of the story.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        pairID: (Optional) string. Unique identifier for the pair of sentences.
        img_path_seq: list of strings. List of image paths corresponding to
            each text item in text_seq.
        task_id: (Optional) int. The integer id to each dataset (task).
    """

    guid: str
    text_seq: list
    label: Optional[str] = None
    pairID: Optional[str] = None
    img_path_seq: Optional[list] = None
    task_id: Optional[int] = None
    multiref_gt: Optional[list] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


class WikiHowPairWiseProcessor(DataProcessor):
    """Processor for WikiHow Steps Dataset, pair-wise data.
    Args:
        data_dir: string. Root directory for the dataset.
        order_criteria: The criteria of determining if a pair is ordered or not.
            "tight" means only strictly consecutive pairs are considered as
            ordered, "loose" means ancestors also count.
        paired_with_image: will only consider sequence that have perfect image
            pairings.
        min_story_length: minimum length of sequence for each.
        max_story_length: maximum length of sequence for each.
    """

    def __init__(self, data_dir=None, order_criteria="tight",
                 paired_with_image=True,
                 min_story_length=5, max_story_length=5,
                 caption_transforms=None, **kwargs):
        """Init."""
        self.data_dir = data_dir
        if self.data_dir is None:
            self.data_dir = WIKIHOW_DATA_ROOT
        assert order_criteria in ["tight", "loose"]
        assert os.path.exists(self.data_dir)
        self.images_dir = self.data_dir
        if "images_dir" in kwargs:
            self.images_dir = kwargs["images_dir"]
        self.order_criteria = order_criteria
        self.paired_with_image = paired_with_image

        min_story_length = max(1, min_story_length)
        max_story_length = max(1, max_story_length)
        min_story_length = min(min_story_length, max_story_length)
        self.min_story_length = min_story_length
        self.max_story_length = max_story_length

        self.caption_transforms = caption_transforms

        if "version_text" in kwargs:
            self.version_text = kwargs["version_text"]
        else:
            self.version_text = None

        self.save_missing_images_info = None
        if "save_missing_images_info" in kwargs:
            self.save_missing_images_info = kwargs["save_missing_images_info"]

        self.multiref_gt = False

    def get_labels(self):
        """See base class."""
        return ["unordered", "ordered"]  # 0: unordered, 1: ordered.

    def _read_json(self, data_dir=None, split="train"):
        """Reads in json lines to create the dataset."""
        if data_dir is None:
            data_dir = self.data_dir

        if self.version_text is not None:
            json_path = os.path.join(data_dir, "wikihow-{}-".format(self.version_text)+split+".json")
            if not os.path.exists(json_path):
                raise ValueError("File: {} not found!".format(json_path))
        else:
            json_path = os.path.join(data_dir, "wikihow-"+split+".json")
        print("Using {}".format(json_path))

        line_cnt = 0
        json_file = open(json_path)
        data = []
        for line in json_file:
            d = json.loads(line.strip())
            line_cnt += 1
            data.append(d)

        story_seqs = []
        missing_images = []

        # Each element in a story seq is (text, image) tuple.
        for data_raw in tqdm(data, total=len(data)):

            # Form the data id.
            wikihow_url = data_raw["url"]
            title_text = data_raw["title"]
            summary_text = data_raw["summary"]

            wikihow_check_id = "###".join([wikihow_url, title_text])
            wikihow_check_id = wikihow_url

            # Multi-reference GTs.
            if "multiref_gt" in data_raw:
                if not self.multiref_gt: self.multiref_gt = True

            for section_id in range(len(data_raw["sections"])):

                section_curr = data_raw["sections"][section_id]
                wikihow_page_id = "###".join([wikihow_url, title_text, str(section_id)])
                wikihow_page_id = "###".join([wikihow_url, str(section_id)])
                story_seq = [wikihow_page_id]

                # TODO: consistency of human test sets.
                include_data = True
                if self.version_text is not None and self.version_text == "human_annot_only_filtered":
                    include_data = False

                for step_id in range(len(section_curr["steps"])):
                    step_curr = section_curr["steps"][step_id]
                    step_headline = step_curr["step_headline"]
                    step_text = step_curr["step_text"]["text"]
                    bullet_points = step_curr["step_text"]["bullet_points"]
                    combined_text = " ".join([step_text] + bullet_points)

                    if self.version_text is not None and self.version_text == "human_annot_only_filtered":
                        check_str = combined_text.split(".")[0]
                        if check_str in human_check_dict:
                            include_data = True

                    if self.caption_transforms is not None:
                        combined_text = self.caption_transforms.transform(combined_text)

                    element = None
                    if self.paired_with_image:
                        # We take the first image for each step.
                        image_path_curr = None
                        for image_field_key in IMAGE_FIELD_NAMES:
                            if image_field_key in step_curr["step_assets"]:
                                image_path_curr = step_curr["step_assets"][
                                    image_field_key]
                                image_path_curr_new = None
                                if image_path_curr is not None and len(image_path_curr) > 0:
                                    image_path_curr = os.path.join(self.images_dir, image_path_curr)

                                    if "wikihow.com" not in image_path_curr:
                                        image_path_curr_new = image_path_curr.replace(
                                            "/images/",
                                            "/www.wikihow.com/images/")
                                    else:
                                        image_path_curr_new = image_path_curr
                                    if not os.path.exists(image_path_curr_new):
                                        image_path_curr_new = image_path_curr.replace(
                                            "/images/",
                                            "/wikihow.com/images/")
                                        if not os.path.exists(image_path_curr_new):
                                            missing_images.append(wikihow_page_id+"###"+str(step_id))
                                            element = None
                                        else:
                                            element = (combined_text, image_path_curr_new)
                                    else:
                                        element = (combined_text, image_path_curr_new)
                                else:
                                    missing_images.append(wikihow_page_id+"###"+str(step_id))
                                    element = None
                                if image_path_curr_new is not None and os.path.exists(image_path_curr_new):
                                    break
                    else:
                        element = (combined_text, None)

                    if element is not None:
                        story_seq.append(element)

                # TODO: Currently different sections are in different
                # sequences for sorting.
                if len(story_seq) < self.min_story_length + 1:
                    pass
                elif not include_data:
                    pass
                else:
                    story_seq = story_seq[:self.max_story_length+1]

                    curr_story_seq_len = len(story_seq)
                    if self.multiref_gt:
                        story_seq = {
                            "story_seq": story_seq,
                            "multiref_gt": data_raw["multiref_gt"]
                        }

                    # TODO: maybe relax this?
                    if (curr_story_seq_len >= self.min_story_length + 1
                        and curr_story_seq_len <= self.max_story_length + 1):
                        story_seqs.append(story_seq)

        print("[WARNING] Number of missing images in {}: {}".format(
            split, len(missing_images)))
        missing_image_paths_f = (
                os.path.join(self.data_dir,
                "missing_images_{}.txt".format(split)
            )
        )

        if self.save_missing_images_info:
            missing_image_paths_file = open(missing_image_paths_f, "w")
            for missing_image_path in missing_images:
                missing_image_paths_file.write(missing_image_path+"\n")
            missing_image_paths_file.close()
            print("          Saves at: {}".format(missing_image_paths_f))

        print("There are {} valid story sequences in {}".format(
              len(story_seqs), json_path))

        return story_seqs

    def _create_examples(self, lines):
        """Creates examples for the training, dev and test sets."""
        paired_examples = []
        for story_seq in lines:
            if self.multiref_gt:
                multiref_gt = story_seq["multiref_gt"]
                story_seq = story_seq["story_seq"]
            else:
                multiref_gt = None
            story_id = story_seq.pop(0)
            len_seq = len(story_seq)
            for i in range(0, len_seq):
                for j in range(0, len_seq):
                    if i == j:
                        continue
                    if self.order_criteria == "tight":
                        if j == i + 1:
                            label = "ordered"
                        else:
                            label = "unordered"
                    elif self.order_criteria == "loose":
                        if j > i:
                            label = "ordered"
                        else:
                            label = "unordered"
                    guid = "{}_{}{}".format(story_id, i+1, j+1)
                    text_a = story_seq[i][0]
                    text_b = story_seq[j][0]
                    img_path_a = story_seq[i][1]
                    img_path_b = story_seq[j][1]
                    distance = abs(j - i)
                    example = InputPairWiseExample(guid=guid, text_a=text_a,
                                                   text_b=text_b, label=label,
                                                   img_path_a=img_path_a,
                                                   img_path_b=img_path_b,
                                                   distance=distance,
                                                   multiref_gt=multiref_gt)
                    paired_examples.append(example)
        return paired_examples

    def get_train_examples(self, data_dir=None):
        """See base class."""
        lines = self._read_json(data_dir=data_dir, split="train")
        return self._create_examples(lines)

    def get_dev_examples(self, data_dir=None):
        """See base class."""
        lines = self._read_json(data_dir=data_dir, split="dev")
        return self._create_examples(lines)

    def get_test_examples(self, data_dir=None):
        """See base class."""
        lines = self._read_json(data_dir=data_dir, split="test")
        return self._create_examples(lines)


class WikiHowGeneralProcessor(WikiHowPairWiseProcessor):
    """Processor for WikiHow Steps Dataset, general sorting prediction.
    Args:
        data_dir: string. Root directory for the dataset.
        paired_with_image: will only consider sequence that have perfect image
            pairings.
        min_story_length: minimum length of sequence for each.
        max_story_length: maximum length of sequence for each.
    """

    def __init__(self, data_dir=None, max_story_length=5, pure_class=False,
                 paired_with_image=True, min_story_length=5,
                 caption_transforms=None, version_text=None, **kwargs):
        """Init."""
        self.data_dir = data_dir
        if self.data_dir is None:
            self.data_dir = WIKIHOW_DATA_ROOT
        assert os.path.exists(self.data_dir)
        self.images_dir = self.data_dir
        if "images_dir" in kwargs:
            self.images_dir = kwargs["images_dir"]
        self.max_story_length = max_story_length
        self.pure_class = pure_class
        self.paired_with_image = paired_with_image

        min_story_length = max(1, min_story_length)
        max_story_length = max(1, max_story_length)
        min_story_length = min(min_story_length, max_story_length)
        self.min_story_length = min_story_length
        self.max_story_length = max_story_length

        self.caption_transforms = caption_transforms
        self.version_text = version_text

        self.save_missing_images_info = None
        if "save_missing_images_info" in kwargs:
            self.save_missing_images_info = kwargs["save_missing_images_info"]

        self.multiref_gt = False

    def get_labels(self):
        """See base class."""
        if self.pure_class:
            n = self.max_story_length
            fact = 1
            for i in range(1, n+1):
                fact = fact * i
            labels = [0 for i in range(fact)]
            return labels

        return list(range(self.max_story_length))

    def _create_examples(self, lines):
        """Creates examples for the training, dev and test sets."""
        head_examples = []
        for story_seq in lines:
            if self.multiref_gt:
                multiref_gt = story_seq["multiref_gt"]
                story_seq = story_seq["story_seq"]
            else:
                multiref_gt = None
            story_id = story_seq.pop(0)
            len_seq = len(story_seq)
            guid = story_id
            text_seq = [x[0] for x in story_seq]
            img_path_seq = [x[1] for x in story_seq]
            example = MultimodalWikiHowExample(
                guid=guid, text_seq=text_seq,
                img_path_seq=img_path_seq,
                multiref_gt=multiref_gt
            )
            head_examples.append(example)
        return head_examples

    def get_train_examples(self, data_dir=None):
        """See base class."""
        lines = self._read_json(data_dir=data_dir, split="train")
        return self._create_examples(lines)

    def get_dev_examples(self, data_dir=None):
        """See base class."""
        lines = self._read_json(data_dir=data_dir, split="dev")
        return self._create_examples(lines)

    def get_test_examples(self, data_dir=None):
        """See base class."""
        lines = self._read_json(data_dir=data_dir, split="test")
        return self._create_examples(lines)
