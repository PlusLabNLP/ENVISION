import os
import re
import sys
import csv
import json
import glob
import string
from tqdm import tqdm
import xml.etree.ElementTree as ET
from IPython.display import Image
import random
import numpy as np
import glob
import shutil
from termcolor import colored

from IPython.display import HTML as html_print
from IPython.display import Markdown

from transformers import AutoTokenizer

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

import cv2
import pandas as pd
from textwrap import wrap

import warnings
warnings.filterwarnings('ignore')

import logging
from enum import Enum
from typing import List, Optional, Union
from dataclasses import dataclass

from matplotlib import pyplot as plt
import matplotlib.patches as patches


def show_image_with_path(img_path, title=None, title_max_len=None, img_size=6, font_size=12):
    img = cv2.imread(img_path)
    plt.figure(figsize=(img_size, img_size))
    plt.imshow(img[:, :, ::-1])
    plt.axis('off')
    if title is not None:
        if title_max_len is not None:
            title = title[:title_max_len] + "... ..."
        title = "\n".join(wrap(title, 60))
        plt.title(title, fontsize=font_size)
    pass ####


def show_one_sampled_data(data, scrambled=False, resized_version=False,
                          show_label=False, unimodal=None, img_size=6, font_size=12,
                          title_max_len=None, show_url=False, id_mappings=None,
                          step_id=None, order=None):
    if step_id is not None:
        step_id -= 1

    if unimodal is not None:
        assert unimodal in ["image", "text"]

    text_seq = data.text_seq[:]
    img_path_seq = data.img_path_seq[:]
    idx_seq = np.arange(len(text_seq))

    if show_url:
        data_guid = data.guid
        if len(data_guid.split("###")) > 1:
            url, title = data_guid.split("###")[0], data_guid.split("###")[1]
        else:
            url = data_guid
        print("Article URL: {}".format(url))

    if scrambled:
        np.random.shuffle(idx_seq)
        idx_seq_to_sort = idx_seq[:]
        arg_sort_idx_seq = np.argsort(idx_seq_to_sort)
        label = list(arg_sort_idx_seq + 1)
        if show_label:
            print("Label: {}".format(arg_sort_idx_seq + 1))

    if order is not None:
        order = [x-min(order) for x in order]
        idx_seq = order

    for seq_idx in idx_seq:
        if step_id is not None:
            if seq_idx != step_id:
                continue
        if show_label:
            seq_idx_in_title = str(seq_idx + 1) + ". "
        else:
            seq_idx_in_title = ""
        text = text_seq[seq_idx]
        text = seq_idx_in_title + text
        img_path = img_path_seq[seq_idx]
        if not resized_version and img_path is not None:
            img_path = img_path.replace("jpg_resized_256", "jpg")
        if unimodal == "image":
            show_image_with_path(img_path, title=None,
                                 title_max_len=title_max_len,
                                 img_size=img_size, font_size=font_size)
        elif unimodal == "text":
            plt.figure(figsize=(img_size, img_size))
            plt.imshow(np.zeros((256, 256)))
            plt.axis('off')
            title = "\n".join(wrap(text, 60))
            plt.title(title, fontsize=font_size)
        else:
            show_image_with_path(img_path, title=text,
                                 title_max_len=title_max_len,
                                 img_size=img_size, font_size=font_size)

    if scrambled:
        return label

    return list(range(len(text_seq)))
