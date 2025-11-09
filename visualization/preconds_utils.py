import csv
import os
import sys
import re
import string
import json
import xml.etree.ElementTree as ET
from IPython.display import Image
import random
import numpy as np
import glob
from termcolor import colored
import pickle
import copy
import pprint
from collections import OrderedDict

from transformers import AutoTokenizer

import cv2
import pandas as pd
from textwrap import wrap
from tqdm import tqdm

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score

# Add-ons.
from visualization_utils import *


id_to_label_dict = {
    0: "null",
    1: "pre-condition",
    2: "post-condition",
}


def make_dict_wrt_id(data, id_keys):
    data_dict = OrderedDict()
    for datum in data:
        doc_id = []
        for id_key in id_keys:
            doc_id.append("{}".format(datum[id_key]))
        doc_id = "###".join(doc_id)
        data_dict[doc_id] = datum
    return data_dict


def compare_golden_and_heuristics(golden_data, golden_heus=None, temp_data=None,
                                  doc_id=None, model_name_and_preds=None,
                                  show_version="v1", highlight_spans=False):
    assert show_version in ["v1", "v2"]
    if doc_id is None:
        doc_ids = list(golden_data.keys())
        doc_id = np.random.choice(doc_ids, 1, replace=False)[0]
    url = doc_id.split("###")[0]
    section_id = doc_id.split("###")[1]
    print("URL: {}  Section: {}".format(url, section_id))
    
    temp_datum = None
    if temp_data is not None:
        if url in temp_data:
            temp_datum = temp_data[url]["sections"][section_id]
        else:
            print("Temporal Data does not exist!")
    
    step_texts = golden_data[doc_id]["step_texts"]
    step_texts = [x["text"] for x in step_texts]
    golden_trace = golden_data[doc_id]["annotations"]["condition_edges"]
    golden_spans = golden_data[doc_id]["annotations"]["text_spans"]
    
    start_ends_to_highlight = [[] for i in range(len(step_texts))]
    for i in range(len(step_texts)):
        len_step_text = np.zeros(len(step_texts[i]), dtype=np.int32)
        for golden_span_id in golden_spans:
            golden_span = golden_spans[golden_span_id]
            if golden_span["step_id"] != i: continue
            start, end = golden_span["step_start"], golden_span["step_end"]
            len_step_text[start:end] = 1
        pass
        start_ends_to_highlight[i] = len_step_text
    pass
    
    heus_trace = None
    if golden_heus is not None:
        heus_trace = golden_heus[doc_id]["trace"]
        span_id_org2new_mapping = golden_heus[doc_id]["span_id_org2new_mapping"]
        span_id_new2org_mapping = golden_heus[doc_id]["span_id_new2org_mapping"]
        heus_entity_trace_labels = heus_trace["entity_trace_labels"]
        heus_id_actionable_map = heus_trace["id_actionable_map"]
    
    step_compr = []
    for i in range(len(step_texts)):
        step_compr_curr = {
            "step_text": step_texts[i],
            "golden_trace": {},
            "heus_trace": {},
        }
        
        # Model predictions.
        if model_name_and_preds is not None:
            for model_name in model_name_and_preds:
                step_compr_curr[model_name] = {}
                model_preds = model_name_and_preds[model_name][doc_id]
                for link in sorted(model_preds):
                    head_id = "span_{}".format(link.split("=>")[0].split("-")[1].split("_")[0])
                    tail_id = "span_{}".format(link.split("=>")[1].split("-")[1].split("_")[0])
                    new_link = "{}=>{}".format(head_id, tail_id)
                    label = id_to_label_dict[model_preds[link]]
                    head_span = golden_spans[head_id]
                    tail_span = golden_spans[tail_id]
                    head_text = head_span["text"]
                    tail_text = tail_span["text"]
                    head_step = head_span["step_id"]
                    tail_step = tail_span["step_id"]
                    if tail_step == i:
                        model_trace_curr = {
                            "head_text": head_text,
                            "tail_text": tail_text,
                            "head_step": head_step,
                            "tail_step": tail_step,
                            "label": label
                        }
                        step_compr_curr[model_name][new_link] = model_trace_curr
                pass
            pass
        
        # Golden data.
        for link in golden_trace:
            link_id = link.split(":")[0]
            head_id = golden_trace[link]["head_id"]
            tail_id = golden_trace[link]["tail_id"]
            label = golden_trace[link]["label"]
            head_span = golden_spans[head_id]
            tail_span = golden_spans[tail_id]
            
            head_text = head_span["text"]
            tail_text = tail_span["text"]
            head_step = head_span["step_id"]
            tail_step = tail_span["step_id"]
            if tail_step == i:
                golden_trace_curr = {
                    "head_text": head_text,
                    "tail_text": tail_text,
                    "head_step": head_step,
                    "tail_step": tail_step,
                    "label": label
                }
                step_compr_curr["golden_trace"][link_id] = golden_trace_curr
        
        # Heuristics data.
        if golden_heus is not None:
            for link in heus_entity_trace_labels:
                label = heus_entity_trace_labels[link].replace("pre", "pre-").replace("post", "post-")
                head_id, tail_id = link.split("=>")
                head_step = int(head_id.split("_")[2])
                tail_step = int(tail_id.split("_")[2])
                head_text = heus_id_actionable_map[head_id]
                tail_text = heus_id_actionable_map[tail_id]

                if tail_step == i:
                    heus_trace_curr = {
                        "head_text": head_text,
                        "tail_text": tail_text,
                        "head_step": head_step,
                        "tail_step": tail_step,
                        "label": label
                    }
                    mapped_head_id = span_id_new2org_mapping[head_id]
                    mapped_tail_id = span_id_new2org_mapping[tail_id]
                    link_id = "{}=>{}".format(mapped_head_id, mapped_tail_id)
                    step_compr_curr["heus_trace"][link_id] = heus_trace_curr
        
        # Add data.
        step_compr.append(step_compr_curr)
        

    for i in range(len(step_compr)):
        step_compr_curr = step_compr[i]
        print("\n"+"="*100)
        print("{} {}".format(red_text(bold_text("Step:")), red_text(bold_text(str(i+1)))))
        
        if temp_datum is not None:
            # pprint.pprint(temp_datum["steps"][str(i)])
            temp_text = temp_datum["steps"][str(i)]["temp_text"]["temporal_predictions"]
            temp_text.append("NULL")
            sent_has_events_text = temp_datum["steps"][str(i)]["temp_text"]["sent_has_events"]
            sorted_sent_ids_text = temp_datum["steps"][str(i)]["temp_text"]["sorted_sent_ids"]
            len_text = len(sorted_sent_ids_text)
            temp_dict_text = {u: v for u, v in zip(sent_has_events_text, temp_text)}
            
            if len(temp_datum["steps"][str(i)]["temp_bullet_points"]) > 0:
                temp_bull = temp_datum["steps"][str(i)]["temp_bullet_points"]["temporal_predictions"]
                temp_bull.append("NULL")
                sent_has_events_bull = temp_datum["steps"][str(i)]["temp_bullet_points"]["sent_has_events"]
                sorted_sent_ids_bull = temp_datum["steps"][str(i)]["temp_bullet_points"]["sorted_sent_ids"]
                len_bull = len(sorted_sent_ids_bull)
                temp_dict_bull = {u: v for u, v in zip(sent_has_events_bull, temp_bull)}
            
            step_sents = sent_tokenize(step_compr_curr["step_text"])
            # print(temp_text)
            # print(temp_bull)
            for sent_id in range(len(step_sents)):
                sent_to_print = step_sents[sent_id]
                if sent_id < len_text:
                    temp_sent_id = sent_id
                    if temp_sent_id in temp_dict_text:
                        sent_to_print += " {}".format(yellow_text(bold_text(temp_dict_text[temp_sent_id])))
                    else:
                        sent_to_print += " {}".format(yellow_text(bold_text("NULL")))
                else:
                    temp_sent_id = sent_id - len_text
                    if temp_sent_id in temp_dict_bull:
                        sent_to_print += " {}".format(yellow_text(bold_text(temp_dict_bull[temp_sent_id])))
                    else:
                        sent_to_print += " {}".format(yellow_text(bold_text("NULL")))
                print(sent_to_print)
        else:
            step_text_to_print = ""
            for c in range(len(step_compr_curr["step_text"])):
                char = step_compr_curr["step_text"][c]
                if start_ends_to_highlight[i][c] == 1:
                    step_text_to_print += bg_yellow_text(char)
                else:
                    step_text_to_print += char
            if highlight_spans:
                print(step_text_to_print)
            else:
                print(step_compr_curr["step_text"])

        golden_trace = step_compr_curr["golden_trace"]
        
        if golden_heus is not None:
            heus_trace = step_compr_curr["heus_trace"]
        
        if show_version == "v1":
            print(bold_text("-"*30 + "-- Golden Traces --" + "-"*30))
            show_links_v1(golden_trace, heus_trace)

            if golden_heus is not None:
                print(bold_text("-"*30 + " Heuristics Traces " + "-"*30))
                show_links_v1(heus_trace, golden_trace)
            
            if model_name_and_preds is not None:
                for model_name in model_name_and_preds:
                    print(bold_text("-"*30 + " {} ".format(model_name) + "-"*30))
                    model_trace = step_compr_curr[model_name]
                    show_links_v1(model_trace, golden_trace)
            pass
        elif show_version == "v2":
            all_other_traces = OrderedDict()
            if golden_heus is not None:
                all_other_traces["Heuristics"] = heus_trace
            if model_name_and_preds is not None:
                for model_name in model_name_and_preds:
                    all_other_traces[model_name] = step_compr_curr[model_name]
            show_links_v2(golden_trace, all_other_traces)
    
    return None


def show_links_v2(golden_trace, all_other_traces=None):
    assert type(all_other_traces) == OrderedDict
    for link in sorted(golden_trace):
        head_text = golden_trace[link]["head_text"]
        tail_text = golden_trace[link]["tail_text"]
        head_step = golden_trace[link]["head_step"]
        label = golden_trace[link]["label"]
        head_text = "(Step {}) {}".format(head_step+1, head_text)
        if "pre" in label:
            head_text = blue_text(head_text)
            tail_text = magenta_text(tail_text)
        else:
            head_text = magenta_text(head_text)
            tail_text = green_text(tail_text)
        to_print = "{} => {}".format(head_text, tail_text)
        print(to_print)
        print("{} {} {}".format(yellow_text(bold_text("(-)")),
                                yellow_text(bold_text("Golden:")),
                                yellow_text(label)))
        # Other traces.
        for trace_name in all_other_traces:
            other_trace = all_other_traces[trace_name]
            if link not in other_trace:
                print("{} {} {}".format(bold_text(red_text("(X)")),
                                        yellow_text(bold_text(trace_name+":")),
                                        yellow_text("null")))
            else:
                trace_label = other_trace[link]["label"]
                if trace_label == golden_trace[link]["label"]:
                    print("{} {} {}".format(bold_text(green_text("(V)")),
                                            yellow_text(bold_text(trace_name+":")),
                                            yellow_text(trace_label)))
                else:
                    print("{} {} {}".format(bold_text(magenta_text("(X)")),
                                            yellow_text(bold_text(trace_name+":")),
                                            yellow_text(trace_label)))

    return None


def show_links_v1(trace, ref_trace=None):
    for link in sorted(trace):
        head_text = trace[link]["head_text"]
        tail_text = trace[link]["tail_text"]
        head_step = trace[link]["head_step"]
        label = trace[link]["label"]
        head_text = "(Step {}) {}".format(head_step+1, head_text)
        if "pre" in label:
            head_text = blue_text(head_text)
            tail_text = magenta_text(tail_text)
        else:
            head_text = magenta_text(head_text)
            tail_text = green_text(tail_text)
        if ref_trace is not None and link in ref_trace:
            to_print = "{}: {} {} {}".format(bold_text(yellow_text(label)),
                                             bold_text(head_text),
                                             bold_text("=>"),
                                             bold_text(tail_text))
        else:
            to_print = "{}: {} => {}".format(yellow_text(label), head_text, tail_text)
        print(to_print)

    return None
