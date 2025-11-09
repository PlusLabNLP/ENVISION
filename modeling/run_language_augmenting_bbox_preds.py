import pickle
import argparse
import csv, os, sys, re, string, json, glob, shutil, random, datetime, math, copy
import pprint
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from collections import OrderedDict

import cv2
import openai
OPENAI_KEY = None
import pandas as pd
import numpy as np

import torch
from transformers import AutoTokenizer
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from scipy.special import softmax

# Evals.
from coco_eval import get_avg_precision_at_iou
from coco_eval import get_single_image_results
from coco_eval import calc_precision_recall, calc_iou_individual
from run_glip_baseline_on_ego4d import coco_eval_results
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
sys.path.insert(0, "../cocoapi")
from PythonAPI.pycocotools.coco import COCO
from PythonAPI.pycocotools.cocoeval import COCOeval

# Multimodal toolboxes.
from lavis.models import model_zoo
from lavis.models import load_model_and_preprocess
from transformers import CLIPProcessor, CLIPModel

# NLP.
from nltk.tokenize import word_tokenize


def construct_coco_results_dict(
    coco_results_file,
    by_label_category=False,
    ooc_category_id=None,
    dis_category_id=None,
):
    assert os.path.exists(coco_results_file)
    coco_results = json.load(open(coco_results_file))
    coco_results_dict = {}
    if by_label_category:
        coco_results_dict = {
            ooc_category_id: {},
            dis_category_id: {},
        }
    for coco_result in coco_results:
        image_id = coco_result["image_id"]
        if by_label_category:
            category_id = coco_result["category_id"]
            if image_id not in coco_results_dict[category_id]:
                coco_results_dict[category_id][image_id] = []
        else:
            if image_id not in coco_results_dict:
                coco_results_dict[image_id] = []
        to_append = {
            "bbox": coco_result["bbox"],
            "score": coco_result["score"],
            "category_id": coco_result["category_id"],
            "image_id": image_id,
        }
        if by_label_category:
            coco_results_dict[category_id][image_id].append(to_append)
        else:
            coco_results_dict[image_id].append(to_append)
    if by_label_category:
        for category_id in coco_results_dict:
            for image_id in coco_results_dict[category_id]:
                coco_results_dict[category_id][image_id] = sorted(
                    coco_results_dict[category_id][image_id],
                    key=lambda x: x["score"],
                    reverse=True
                )
    else:
        for image_id in coco_results_dict:
            coco_results_dict[image_id] = sorted(
                coco_results_dict[image_id],
                key=lambda x: x["score"],
                reverse=True
            )
    return coco_results_dict


def distractor_reranking(
    ooc_result_file,       # object-of-change.
    output_file_name,
    dis_result_file=None,  # distractor-objects
    use_top_k_dis_bboxes=None,
    iou_thres=0.5,
    ooc_category_id=None,
    dis_category_id=None,
):
    import PythonAPI.pycocotools._mask as _mask
    iou_func = _mask.iou

    if dis_result_file is not None:
        ooc_results_dict = construct_coco_results_dict(ooc_result_file)
        dis_results_dict = construct_coco_results_dict(dis_result_file)
    # Single result json file can contain distractor objects as well.
    else:
        multi_results_dict = construct_coco_results_dict(
            ooc_result_file,
            by_label_category=True,
            ooc_category_id=ooc_category_id,
            dis_category_id=dis_category_id,
        )
        ooc_results_dict = multi_results_dict[ooc_category_id]
        dis_results_dict = multi_results_dict[dis_category_id]
    if use_top_k_dis_bboxes is None:
        use_top_k_dis_bboxes = 100

    def find_matched_bbox_and_alter_scores(
        query,
        others,
        iou_thres=0.5,
    ):
        query_bbox = query["bbox"]
        other_bboxes = [o["bbox"] for o in others]
        iscrowd = [0] * len(other_bboxes)
        ious = iou_func(other_bboxes, [query_bbox], iscrowd)
        over_iou_indices = []
        for i in range(len(ious)):
            iou = ious[i]
            # XXX: For now let's just make it 0 (re-rank to last).
            # if iou >= iou_thres:
            if iou >= iou_thres and iou < 0.5:
                # others[i]["score"] = 0
                # others[i]["score"] += 0.2
                # others[i]["score"] += query["score"]
                # others[i]["score"] += query["score"] * 0.2
                # others[i]["score"] = (others[i]["score"] + query["score"]) / 2
                pass
            elif iou >= 0.8:
                # others[i]["score"] -= query["score"]
                # if others[i]["score"] > 0.25:
                #     others[i]["score"] -= 0.25
                over_iou_indices.append((iou[0], i))
                pass
            pass

        over_iou_indices = [v for u, v in over_iou_indices]
        for i in range(len(ious)):
            if i not in over_iou_indices:
                others[i]["score"] += 0.05
        # over_iou_indices = sorted(over_iou_indices, reverse=True)
        # x = 0.5
        # for iou, i in over_iou_indices:
        #     if others[i]["score"] >= x:
        #         others[i]["score"] -= x
        #     x = max(x-0.05, 0)
        return others

    new_ooc_results = []
    for image_id in ooc_results_dict:
        ooc_res = ooc_results_dict[image_id]
        if image_id not in dis_results_dict:
            new_ooc_results += ooc_res
            continue
        dis_res = dis_results_dict[image_id]
        for dis_bbox in dis_res[:use_top_k_dis_bboxes]:
            ooc_res = find_matched_bbox_and_alter_scores(
                query=dis_bbox,
                others=ooc_res,
                iou_thres=iou_thres,
            )
        new_ooc_results += ooc_res
        pass
    json.dump(
        new_ooc_results,
        open(output_file_name, "w"),
    )
    print("Saving re-ranked file to: {}".format(output_file_name))

    return None
    

def get_scod_clipped_narrations(
    scod_clip,
    all_narrations,
    curr_videos_root,
    narration_pass="narration_pass_1",
    anchor_frame="pre",
    top_k=5,
    verbose=False,
    get_video=True,
):    
    video_uid = scod_clip["video_uid"]
    if get_video:
        ego4d_video_path = os.path.join(
            curr_videos_root,
            "{}.mp4".format(video_uid),
        )

        cap = cv2.VideoCapture(ego4d_video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if verbose:
            print(ego4d_video_path, fps)
        duration = float(frame_cnt) / float(fps)
        cap.release()
    
    clip_uid = scod_clip["clip_uid"]
    pre_frame = scod_clip["pre_frame"]
    pnr_frame = scod_clip["pnr_frame"]
    pos_frame = scod_clip["post_frame"]
    if narration_pass not in all_narrations[video_uid]:
        return [], []
    curr_narrations = all_narrations[video_uid][narration_pass]["narrations"]
    
    if verbose: 
        obj_of_changes = []
        for frame in ["pre", "pnr", "post"]:
            if "bbox" not in scod_clip["{}_frame".format(frame)]:
                continue
            obj_info = scod_clip["{}_frame".format(frame)]["bbox"]
            for o in obj_info:
                if o["object_type"] == "object_of_change":
                    obj_of_changes.append(o["structured_noun"])
        obj_of_changes = list(set(obj_of_changes))
    
    pre_frame_num = pre_frame["frame_number"]
    pnr_frame_num = pnr_frame["frame_number"]
    pos_frame_num = pos_frame["frame_number"]
    
    if get_video:
        pre_frame_sec = round(pre_frame_num/fps, 2)
        pnr_frame_sec = round(pnr_frame_num/fps, 2)
        pos_frame_sec = round(pos_frame_num/fps, 2)
    
    clip_narrations = []
    closest_narrations = []
    for i in range(len(curr_narrations)-1):
        timestamp_sec = curr_narrations[i]["timestamp_sec"]
        timestamp_frame = curr_narrations[i]["timestamp_frame"]
        next_timestamp_sec = curr_narrations[i+1]["timestamp_sec"]
        next_timestamp_frame = curr_narrations[i+1]["timestamp_frame"]
        narration_text = curr_narrations[i]["narration_text"]
        if (
            timestamp_frame >= pre_frame_num
            and timestamp_frame <= pos_frame_num
        ):
            clip_narrations.append((
                timestamp_sec, timestamp_frame,
                next_timestamp_sec, next_timestamp_frame,
                narration_text,
            ))
        
        anchor_frame_num = scod_clip["{}_frame".format(anchor_frame)]["frame_number"]
        closest_narrations.append((
            abs(timestamp_frame-anchor_frame_num),
            timestamp_sec, timestamp_frame,
            next_timestamp_sec, next_timestamp_frame,
            narration_text,
        ))
        
    closest_narrations = sorted(closest_narrations)[:top_k]
    closest_narrations = [(a, b, c, d, e) for _, a, b, c, d, e in closest_narrations]
    
    if verbose and get_video:
        print("FPS: {}".format(fps))
        print("Object of change: {}".format(obj_of_changes))
        print("Pre/PNR/Post frame = {} / {} / {}".format(pre_frame_num, pnr_frame_num, pos_frame_num))
        print("Pre/PNR/Post sec   = {} / {} / {}".format(
            str(datetime.timedelta(seconds=pre_frame_sec))[:10],
            str(datetime.timedelta(seconds=pnr_frame_sec))[:10],
            str(datetime.timedelta(seconds=pos_frame_sec))[:10])
        )

    tagged_closest_narrations = []
    for a, b, c, d, e in closest_narrations:
        tag, tag_str = False, ""
        if (b > pre_frame_num and b < pos_frame_num) or (d > pre_frame_num and d < pos_frame_num):
            tag, tag_str = True, "(v)"
        if verbose:
            print(
                "[{}-{}]({}-{}) {} {}".format(
                    str(datetime.timedelta(seconds=a))[:10],
                    str(datetime.timedelta(seconds=c))[:10],
                    b, d, e, tag_str,
                )
            )
        tagged_closest_narrations.append((tag, a, b, c, d, e))
    
    return clip_narrations, tagged_closest_narrations


def cropping_bboxes_and_inference(
    image_path,
    pred_bboxes,
    gt_bboxes,
    narration,
    top_k_bboxes=3,
    region_proposal_probings={
        "caption_func": None,
        "clip_func": None,
    },
    bbox_inference_strategy="top-1",
    verbose=False,
    args=None,
):
    assert os.path.exists(image_path)
    frame_image = Image.open(image_path)
    rpn_captions = []
    bbox_metadata = []

    ap_gt_bboxes = [
        [b["bbox"]["x"],
         b["bbox"]["y"],
         b["bbox"]["x"]+b["bbox"]["width"],
         b["bbox"]["y"]+b["bbox"]["height"]]
        for b in gt_bboxes
    ]

    image_width, image_height = Image.open(image_path).size

    bboxes = pred_bboxes
    for bbox_idx in range(len(bboxes[:top_k_bboxes])):
        bbox_info = bboxes[bbox_idx]
        bbox = bbox_info["bbox"]
        if "label" not in bbox_info:
            continue

        ap_pred_bboxes = [
            [bbox[0],
             bbox[1],
             bbox[0]+bbox[2],
             bbox[1]+bbox[3]]
        ]
        ap_res = {"AP50": None, "AP75": None}
        for iou_thr in [0.5, 0.75]:
            res = get_single_image_results(ap_gt_bboxes, ap_pred_bboxes,
                                           iou_thr=iou_thr)
            res = calc_precision_recall({"obj": res})
            prec, rec = res
            ap_res["AP{}".format(int(iou_thr*100))] = prec

        cropped_bboxed_frame_image = None
        if (
            args.output_image_cropped_pickle_file is not None
            or not "top-"  in bbox_inference_strategy
        ):
            # enlarge_bbox_cropping_width = 50
            enlarge_bbox_cropping_width = int(0.2 * bbox[2])
            enlarge_bbox_cropping_height = int(0.2 * bbox[3])
            org_bbox = bbox.copy()
            enlarge_bbox_cropping_width = None
            enlarge_bbox_cropping_height = enlarge_bbox_cropping_width
            if enlarge_bbox_cropping_width is not None:
                bbox[0] = max(0, org_bbox[0] - enlarge_bbox_cropping_width)
                bbox[1] = max(0, org_bbox[1] - enlarge_bbox_cropping_height)
                bbox[2] = min(image_width-org_bbox[0], org_bbox[2]+2*enlarge_bbox_cropping_width)
                bbox[3] = min(image_height-org_bbox[1], org_bbox[3]+2*enlarge_bbox_cropping_height)

            cropped_bboxed_frame_image = Image.open(image_path).crop((
                bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
            ))
            # print(cropped_bboxed_frame_image.size, image_width, image_height, org_bbox, bbox)
            # raise

        # Cropping captions.
        if args.output_image_cropped_pickle_file is not None:
            np_cropped_bboxed_frame_image = np.asarray(cropped_bboxed_frame_image)
            bbox_metadata.append({
                "score": bbox_info["score"],
                "metric": ap_res,
                "narration": narration,
                "cropped_region": np_cropped_bboxed_frame_image,
                "bbox": bbox,
            })
        elif region_proposal_probings is not None:
            if region_proposal_probings["caption_func"] is not None:
                # TODO: can change the generation settings.
                lavis_caption = region_proposal_probings["caption_func"].generate(
                    {
                        "image": vis_processors["eval"](
                            cropped_bboxed_frame_image).unsqueeze(0).to(device)
                    },
                    num_beams=4,
                    repetition_penalty=0.9,
                    num_captions=3,
                    use_nucleus_sampling=True,
                )
                rpn_captions.append(
                    (
                        cropped_bboxed_frame_image,
                        bbox_info["score"],
                        lavis_caption,
                        ap_res,
                        bbox,
                    )
                )
            else:
                rpn_captions.append(
                    (
                        cropped_bboxed_frame_image,
                        bbox_info["score"],
                        "NULL",
                        ap_res,
                        bbox,
                    )
                )

    if len(rpn_captions) > 0:
        metadata = []
        for idx in range(len(rpn_captions)):
            rpn_caption = rpn_captions[idx]
            ap_res = rpn_caption[3]
            if verbose:
                print("Bbox-{}. Score: {}".format(idx+1, rpn_caption[1]))
                print("        AP50 = {}  AP75 = {}".format(
                    ap_res["AP50"], ap_res["AP75"]))
                pprint.pprint(rpn_caption[2])
            metadata.append({
                "score": rpn_caption[1],
                "metric": ap_res,
                "captions": rpn_caption[2],
                "narration": narration,
                "cropped_region": rpn_caption[0],
                "bbox": rpn_caption[4],
            })

        ap_res, selected_metadata = inference_metadata(
            metadata, bbox_inference_strategy,
            region_proposal_probings, args=args)
        # Delete the frame for easier json storage.
        for metadatum in metadata:
            del metadatum["cropped_region"]

        return ap_res, metadata, bbox_metadata, selected_metadata

    ap_res, selected_metadata = inference_metadata(
        bbox_metadata, bbox_inference_strategy,
        region_proposal_probings, args=args)
    return ap_res, None, bbox_metadata, selected_metadata


# TODO(bryanzhou, telinwu): some strategy to dynamically pick the bbox.
def inference_metadata(
    metadata,
    bbox_inference_strategy,
    region_proposal_probings,
    args=None,
):
    ap_res = {"AP50": 0, "AP75": 0}

    if "top-" in bbox_inference_strategy:
        top_k = int(bbox_inference_strategy.split("top-")[-1])
        selected_metadata = top_k_inference(metadata, top_k)
    elif "naive_entity_overlapping" in bbox_inference_strategy:
        selected_metadata = naive_entity_overlapping_inference(metadata)
    elif "clip_similarity":
        selected_metadata = clip_similarity_ranking(
            metadata, region_proposal_probings, args=args)
    elif "new_inference":
        selected_metadata = new_inference(
            metadata, region_proposal_probings, args=args)
    else:
        raise NotImplementedError("Not done with {} yet!".format(bbox_inference_strategy))

    raw_ap_res = [x["metric"] for x in selected_metadata]
    for _ap_res in raw_ap_res:
        for ap_key in ap_res:
            ap_res[ap_key] += _ap_res[ap_key]
    for ap_key in ap_res:
        ap_res[ap_key] /= len(selected_metadata)

    return ap_res, selected_metadata


# TODO: ...
def new_inference(
    metadata,
    region_proposal_probings,
    args=None,
):
    raise


def top_k_inference(
    metadata,
    top_k=3,
):
    return metadata[:top_k]


def naive_entity_overlapping_inference(
    metadata
):
    overlapping_best_score = 0
    overlapping_best_idx = 0
    for idx in range(len(metadata)):
        metadatum = metadata[idx]
        narration = metadatum["narration"]
        captions = metadatum["captions"]
        narration_tokens = word_tokenize(narration)
        caption_tokens = [word_tokenize(c) for c in captions]

        is_noun = lambda pos: pos[:2] == "NN"
        all_nouns_narration = sorted(list(set([w for (w, p)
            in nltk.pos_tag(narration_tokens) if is_noun(p)])))
        if "C" in all_nouns_narration:
            all_nouns_narration.pop(all_nouns_narration.index("C"))
        curr_overlapping_score = 0
        for cap_tkn in caption_tokens:
            curr_nouns_cap = sorted(list(set([w for (w, p)
                in nltk.pos_tag(cap_tkn) if is_noun(p)])))
            for noun_cap in curr_nouns_cap:
                if noun_cap in all_nouns_narration:
                    curr_overlapping_score += 1
                pass
            pass
        if curr_overlapping_score > overlapping_best_score:
            overlapping_best_score = curr_overlapping_score
            overlapping_best_idx = idx
        pass

    return [metadata[overlapping_best_idx]]


def extract_SRL_args(res):
    parsed_args_list = []
    for i in range(len(res["verbs"])):
        curr_parse = res["verbs"][i]
        verb = curr_parse["verb"]
        desc = curr_parse["description"]

        args_dict = OrderedDict()
        token_spans = []
        in_arg = False
        tokens = desc.split(" ")
        for token in tokens:
            if "[V" in token or "[ARG" in token:
                in_arg = True
                curr_arg = token.split("[")[-1].split(":")[0]
            elif in_arg and "]" in token:
                curr_token = token.split("]")[0]
                token_spans.append(curr_token)
                curr_span = " ".join(token_spans)
                args_dict[curr_arg] = curr_span
                in_arg = False
                token_spans = []
            elif in_arg:
                curr_token = token
                token_spans.append(curr_token)
            else:
                pass
        parsed_args_list.append((args_dict, curr_parse))

    # parsed_args_list = sorted(parsed_args_list, key=lambda x: len(x[0]))
    parsed_args_list = sorted(
        parsed_args_list,
        key=lambda x: (
            len(x[0]),
            abs(x[1]["tags"].index("B-V") - x[1]["tags"].index("B-ARG1")) \
                if "B-ARG1" in x[1]["tags"] else 1000,
        ),
        reverse=True,
    )
    # pprint.pprint(parsed_args_list)
    # raise

    return parsed_args_list


def clip_similarity_ranking(
    metadata,
    region_proposal_probings,
    args=None,
):

    clip_processor, clip_model = region_proposal_probings["clip_func"]

    clip_images = []
    scores = [m["score"] for m in metadata]
    ap_ress = [m["metric"] for m in metadata]
    avg_ap50 = np.mean([a["AP50"] for a in ap_ress])
    for idx in range(len(metadata)):
        metadatum = metadata[idx]
        narration = metadatum["narration"]
        cropped_region = metadatum["cropped_region"]
        clip_images.append(cropped_region)

    use_nouns = True
    prefix = "A photo of "
    # prefix = ""

    trimmed_narration = narration.split("#C ")[-1]
    trimmed_narration = trimmed_narration.strip()
    # trimmed_narration = trimmed_narration.replace("C ", "Someone ")
    trimmed_narration = trimmed_narration.replace("C ", "A person ")

    if "srl_func" in region_proposal_probings:
        srl_model = region_proposal_probings["srl_func"]
        srl_res = srl_model.predict(sentence=trimmed_narration)
        # pprint.pprint(srl_res)
        parsed_args_list = extract_SRL_args(srl_res)
        # pprint.pprint(parsed_args_list)
        srl_text = []
        for parsed_args, srl in parsed_args_list:
            for _arg in ["V", "ARG1"]:
                if _arg in parsed_args:
                    use_nouns = False
                    srl_text.append(parsed_args[_arg])
                pass
            pass
        srl_text = " ".join(srl_text)
        pass

    # Some parsing.
    if use_nouns:
        is_noun = lambda pos: pos[:2] == "NN"
        narration_tokens = word_tokenize(narration)
        # all_nouns_narration = sorted(list(set([w for (w, p)
        #     in nltk.pos_tag(narration_tokens) if is_noun(p)])))
        all_nouns_narration = [w for (w, p)
            in nltk.pos_tag(narration_tokens) if is_noun(p)]
        while "C" in all_nouns_narration:
            all_nouns_narration.pop(all_nouns_narration.index("C"))
        while "hand" in all_nouns_narration:
            all_nouns_narration.pop(all_nouns_narration.index("hand"))
        # print(narration)
        # print(all_nouns_narration)
        # text = prefix + " ".join(all_nouns_narration) + "."
        texts = []
        for n in all_nouns_narration:
            t = prefix + n + " being used."
            texts.append(t)
        # print(text)
        if len(texts) <= 0:
            return [metadata[0]]
        # text = narration
        # raise

    texts = [trimmed_narration]
    texts = [srl_text]
    # print(narration)
    # print(texts)
    # raise

    inputs = clip_processor(
        text=texts, images=clip_images, return_tensors="pt", padding=True)
    inputs = inputs.to(args.device)
    outputs = clip_model(**inputs)
    logits_per_text = outputs.logits_per_text
    probs = logits_per_text.softmax(dim=1)
    # selected_idx = torch.argmax(probs).detach().cpu().numpy()

    if False:
        texts2 = [trimmed_narration]
        inputs2 = clip_processor(
            text=texts2, images=clip_images, return_tensors="pt", padding=True)
        inputs2 = inputs2.to(args.device)
        outputs2 = clip_model(**inputs2)
        logits_per_text2 = outputs2.logits_per_text
        probs2 = logits_per_text2.softmax(dim=1)
        probs = (probs + probs2) / 2

    probs = probs.detach().cpu().numpy()
    summed_probs = np.sum(probs, axis=0)
    new_summed_probs = summed_probs
    scores = np.asarray(scores)

    if True:
    # if False:
        if (
            np.max(summed_probs) > 0.5
            and np.max(scores) < 0.9
        ):
            # new_summed_probs = summed_probs + scores
            new_summed_probs = summed_probs
        else:
            new_summed_probs = scores
    else:
        new_summed_probs = summed_probs + scores

    if True:
        inds = np.argsort(-new_summed_probs)
        new_summed_probs = [scores[idx] for idx in inds]
        for idx in range(len(new_summed_probs)):
            curr_score = new_summed_probs[idx]
            metadata[idx]["score"] = curr_score
        return metadata

    max_prob = np.max(new_summed_probs)
    # print(summed_probs)
    # print(scores)
    # print(new_summed_probs)
    # new_summed_probs = summed_probs
    selected_idx = np.argmax(new_summed_probs)
    # raise
    # if max_prob < 0.5:
    #     selected_idx = 0
    # print("Max prob = {}  Selected = {}".format(max_prob, selected_idx))
    # raise

    if (
        selected_idx != 0
        and avg_ap50 > 0
        and ap_ress[0]["AP50"] <= 0
        and False
    ):
        print(scores)
        pprint.pprint(ap_ress)
        print(selected_idx)
        print(probs)
        print(narration)
        print(text)
        raise

    # top_score = scores[0]
    # if top_score > 0.75:
    #     selected_idx = 0

    return [metadata[selected_idx]]


def perform_lang_aug_scod(
    inferences,
    keyed_scod_clips,
    all_narrations,
    narration_pass,
    process_first_k=None,
    top_k_bboxes=10,
    region_proposal_probings=None,
    bbox_inference_strategy="top-1",
    coco_results_dict=None,
    data_keys_to_use=None,
    verbose=True,
    args=None,
):
    # Decide if we need to do early stopping.
    if process_first_k is None:
        process_first_k = len(inferences) + 10e8
    process_cnt = 0

    if verbose:
        print("="*50)

    total_ap_res = {"AP50": [], "AP75": []}
    overall_inference_metadata = []

    for inf in tqdm(inferences, desc="Inferencing"):
        if process_cnt >= process_first_k:
            print("Early stopping due to max count = {}".format(process_first_k))
            break

        no_valid_narrated_frames = True

        for fr in inf:
            frame_inf = inf[fr]
            video_uid = frame_inf["video_uid"]
            frame_cnt = frame_inf["frame_cnt"]
            image_folder = frame_inf["image_folder"]
            image_name = frame_inf["image_name"]
            gt_bboxes = frame_inf["gt_bboxes"]
            pred_bboxes = frame_inf["pred_bboxes"]
            frame_name = fr.split("_")[0]
            if frame_name not in args.frames_to_consider:
                continue

            key = "{}_{}".format(video_uid, frame_cnt)
            scod = keyed_scod_clips[key]

            # Try to get the (closest) narration.
            clip_narrations, closest_narrations = get_scod_clipped_narrations(
                scod,
                all_narrations,
                curr_videos_root,
                narration_pass=narration_pass,
                top_k=5,
                anchor_frame="pnr",
                verbose=False,
            )

            # Skipping criteria.
            if len(closest_narrations) <= 0:
                continue
            if (
                data_keys_to_use is not None and
                key not in data_keys_to_use
            ):
                continue
            no_valid_narrated_frames = False
            valid_narrations = [x for x in closest_narrations if x[0] is True]
            if False:
                if len(valid_narrations) > 0:
                    closest_narration = valid_narrations[-1][-1]
                else:
                    closest_narration = closest_narrations[0][-1]
            else:
                closest_narration = closest_narrations[0][-1]

            if verbose:
                print("ID: {}".format(key))
                print("{} Narration: {}".format(fr, closest_narration))

            # Inference in the cropped regions.
            image_path = os.path.join(image_folder, image_name)
            ap_res, inf_metadata, bbox_metadata, selected_metadata = cropping_bboxes_and_inference(
                image_path,
                pred_bboxes,
                gt_bboxes,
                closest_narration,
                top_k_bboxes=top_k_bboxes,
                region_proposal_probings=region_proposal_probings,
                bbox_inference_strategy=bbox_inference_strategy,
                verbose=verbose,
                args=args,
            )

            add_to_coco_res(
                coco_results_dict,
                gt_bboxes,
                selected_metadata
            )

            for ap_key in total_ap_res:
                total_ap_res[ap_key].append(ap_res[ap_key])

            curr_inference = {
                "video_uid": frame_inf["video_uid"],
                "frame_cnt": frame_inf["frame_cnt"],
                "frame_name": frame_name,
                "image_folder": frame_inf["image_folder"],
                "image_name": frame_inf["image_name"],
                "data_key": "{}_{}".format(video_uid, frame_cnt),
                "narration": closest_narrations[0][-1],
                "bbox_metadata": inf_metadata,
            }
            if args.output_image_cropped_pickle_file is not None:
                curr_inference["bbox_metadata"] = bbox_metadata
                curr_inference["closest_narrations"] = closest_narrations
                curr_inference["gt_bboxes"] = gt_bboxes
            # pprint.pprint(curr_inference)
            # raise
            overall_inference_metadata.append(curr_inference)

        # FIXME: for now, if not narrated, we just skip.
        if no_valid_narrated_frames:
            continue

        if verbose:
            print("="*50)

        process_cnt += 1

    # Overall AP results.
    for ap_key in total_ap_res:
        total_ap_res[ap_key] = np.mean(np.asarray(total_ap_res[ap_key]))

    return total_ap_res, overall_inference_metadata


def add_to_coco_res(
    coco_results_dict,
    gt_bboxes,
    selected_metadata,
):
    curr_image_id = coco_results_dict["image_id"]
    for gt_bbox in gt_bboxes:
        bbox = [
            gt_bbox["bbox"]["x"],     gt_bbox["bbox"]["y"],
            gt_bbox["bbox"]["width"], gt_bbox["bbox"]["height"],
        ]
        ann = {
            "segmentation": [],
            "area": bbox[2] * bbox[3],
            "iscrowd": 0,
            "ignore": 0,
            "image_id": curr_image_id,
            "bbox": bbox,
            "category_id": 1,
            "id": coco_results_dict["id"],
        }
        coco_results_dict["gts"].append(ann)
        coco_results_dict["id"] += 1

    for selected_d in selected_metadata:
        pred = {
            "image_id": curr_image_id,
            "category_id": 1,
            "bbox": selected_d["bbox"],
            "score": selected_d["score"],
        }
        coco_results_dict["dts"].append(pred)

    coco_results_dict["image_id"] += 1
    pass  ####


def get_coco_results(
    original_coco_gt_file,
    coco_gt_file,
    coco_pred_file,
    top_k=1,
    verbose=False,
):
    coco_gt = COCO(annotation_file=original_coco_gt_file, verbose=verbose)
    coco_gt = coco_gt.loadRes(coco_gt_file)
    coco_dt = coco_gt.loadRes(coco_pred_file)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.verbose = verbose
    if top_k is not None:
        coco_eval.params.maxDets = [top_k, top_k, top_k]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return None


def perform_lang_aug_scod_on_file(
    f,
    bbox_inference_strategy,
    region_proposal_probings={
        "caption_func": None,
        "clip_func": None,
    },
    process_first_k=None,
    data_keys_to_use=None,
    coco_results_dict=None,
    verbose=False,
    args=None,
):

    total_ap_res = {"AP50": [], "AP75": []}

    if "json" in f.split("/")[-1].split(".")[-1]:
        data = json.load(open(f))
    elif "pickle" in f.split("/")[-1].split(".")[-1]:
        data = pickle.load(open(f, "rb"))

    print("Num inferences: {}".format(len(data)))
    if process_first_k is None:
        process_first_k = len(inferences) + 10e8
    process_cnt = 0

    for d in tqdm(data, desc="Inferencing..."):
        data_key = d["data_key"]

        if (
            "frame_name" in d and
            d["frame_name"] not in args.frames_to_consider
        ):
            continue

        if process_cnt >= process_first_k:
            print("Early stopping due to max count = {}".format(process_first_k))
            break
        if (
            data_keys_to_use is not None and
            data_key not in data_keys_to_use
        ):
            continue

        metadata = d["bbox_metadata"]
        narration = d["narration"]
        gt_bboxes = d["gt_bboxes"]
        for metadatum in metadata:
            if "narration" not in metadatum:
                metadatum["narration"] = narration
        ap_res, selected_metadata = inference_metadata(
            metadata, bbox_inference_strategy,
            region_proposal_probings, args=args)
        for ap_key in total_ap_res:
            total_ap_res[ap_key].append(ap_res[ap_key])

        add_to_coco_res(
            coco_results_dict,
            gt_bboxes,
            selected_metadata
        )

        process_cnt += 1

    # Overall AP results.
    for ap_key in total_ap_res:
        total_ap_res[ap_key] = np.mean(np.asarray(total_ap_res[ap_key]))
    
    return total_ap_res


""" Usage
"""
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--process_first_k",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--narration_pass",
        default="narration_pass_1",
        type=str,
    )
    parser.add_argument(
        "--top_k_bboxes",
        default=3,
        type=int,
    )
    parser.add_argument(
        "--best_k",
        default=3,
        type=int,
    )
    parser.add_argument(
        "--use_data_criteria",
        action="store_true",
    )
    parser.add_argument(
        "--use_srl",
        action="store_true",
    )
    parser.add_argument(
        "--bbox_inference_strategy",
        default="top-1",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--coco_gt_file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "--results_file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--output_image_cropped_pickle_file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--blip_model_name",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--blip_model_type",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--clip_model_name",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--use_results_file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--frames_to_consider",
       default=["pnr"],
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--use_image_cropped_pickle_file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--ego4d_videos_root",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--ego4d_annotations_root",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--processed_scod_image_folder",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--processed_inference_json_file",
        default=None,
        type=str,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    if args.output_image_cropped_pickle_file is not None:
        args.use_data_criteria = False
        args.use_results_file = False
        args.bbox_inference_strategy = ["top-1"]

    # TODO: can change the below criteria function if necessary.
    AP_KEY = "AP50" # From ["AP", "AP50", "AP75"], standard object detection average precision metrics.
    BEST_AP_KEY = "Best-{}-{}".format(args.best_k, AP_KEY)
    criteria = lambda x: (
        x["pnr_frame"][AP_KEY]["precision"] <= 0 and
        # x["pnr_frame"][AP_KEY]["precision"] <= 0 and
        x["pnr_frame"][BEST_AP_KEY]["precision"] > 0
    )

    region_proposal_probings = {
        "caption_func": None,
        "clip_func": None,
        "srl_func": None,
    }

    # TODO: LAVIS.
    # Note that the `num_captions` may not exist for all types of blip models.
    # CPU device for the try-outs.
    device = torch.device("cpu")
    device = torch.device("cuda")
    args.device = device
    if args.blip_model_name is not None:
        lavis_model, vis_processors, _ = load_model_and_preprocess(
            name=args.blip_model_name,
            model_type=args.blip_model_type,

            # name="blip2_t5",
            # model_type="pretrain_flant5xxl",
            # model_type="caption_coco_flant5xl",

            # name="blip2",
            # model_type="coco",
            is_eval=True,
            device=device
        )
        region_proposal_probings = {
            "caption_func": lavis_model,
        }
        region_proposal_probings["caption_func"] = None

    # CLIP.
    if args.clip_model_name is not None:
        clip_model = CLIPModel.from_pretrained(
            # args.clip_model_name
            # "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge"
            # "/baselines/clip_scod_train_finetuned/checkpoint-500"
            # "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge"
            # "/baselines/clip_scod_train_enlarged_bbox0p2_finetuned/checkpoint-1500"

            "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge"
            # "/baselines/clip_scod_val_finetuned/checkpoint-5000/"
            # "/baselines/clip_large_scod_train_finetuned/checkpoint-1000/"
            "/baselines/clip_large_scod_train_srl_v_arg1_finetuned/checkpoint-1000/"
        )
        clip_processor = CLIPProcessor.from_pretrained(args.clip_model_name)
        clip_model = clip_model.to(device)
        region_proposal_probings["clip_func"] = (
            clip_processor, clip_model
        )
    
    # AllenNLP.
    if args.use_srl:
        import allennlp_models.tagging
        from allennlp_models import pretrained
        from allennlp.predictors.predictor import Predictor
        predictor_srl = pretrained.load_predictor(
            "structured-prediction-srl-bert",
            # cuda_device=0,
        )
        region_proposal_probings["srl_func"] = predictor_srl

    # Ego4D args.
    curr_videos_root = args.ego4d_videos_root
    annots_root = args.ego4d_annotations_root
    scod_train_file = "fho_scod_train.json"
    scod_val_file = "fho_scod_val.json"
    processed_scod_image_folder = args.processed_scod_image_folder

    assert args.narration_pass in [
        "narration_pass_1",
        "narration_pass_2",
    ]

    all_narrations_file = os.path.join(annots_root, "narration.json")
    all_narrations = json.load(open(all_narrations_file))

    scod_train_file = os.path.join(annots_root, scod_train_file)
    scod_val_file = os.path.join(annots_root, scod_val_file)
    scod_train_data = json.load(open(scod_train_file))
    scod_val_data = json.load(open(scod_val_file))
    scod_clips = scod_val_data["clips"]

    downloaded_video_id_and_paths = {
        x.split(".mp4")[0]: os.path.join(curr_videos_root, x)
        for x in os.listdir(curr_videos_root) if ".mp4" in x
    }
    print("Currently we downloaded {} ego4d videos.".format(len(downloaded_video_id_and_paths)))

    downloaded_scod_clips = []

    video_wo_narrations = {}
    for scod_clip in scod_clips:
        video_uid = scod_clip["video_uid"]
        if video_uid in downloaded_video_id_and_paths:
            downloaded_scod_clips.append(scod_clip)
        if args.narration_pass not in all_narrations[video_uid]:
            video_wo_narrations[video_uid] = True

    print("And there are {} clips available for scod.".format(len(downloaded_scod_clips)))

    # print("The following videos do not have narrations at {}".format(args.narration_pass))
    # for video_uid in sorted(video_wo_narrations):
    #     print(video_uid)
    print("In total, {} videos do not have narration: {}".format(
        len(video_wo_narrations), args.narration_pass))

    # Results.
    keyed_scod_clips = {}
    for scod_clip in tqdm(downloaded_scod_clips, desc="SCOD Clips"):
        video_uid = scod_clip["video_uid"]
        for fr in ["pre", "pnr", "post"]:
            frame_cnt = scod_clip["{}_frame".format(fr)]["frame_number"]
            key = "{}_{}".format(video_uid, frame_cnt)   
            keyed_scod_clips[key] = scod_clip
        pass

    data_keys_to_use = None
    if args.use_data_criteria:
        _, _, keyed_results = coco_eval_results(

            args.processed_inference_json_file,
            top_k=1,
            verbose=False,
            first_k=None,
            best_k=args.best_k,
            frame_keys=["pnr"],
        )
        data_keys_to_use = []
        for data_key, data_res in keyed_results.items():
            if criteria(data_res):
                data_keys_to_use.append(data_key)
        print("Valid results fulfilling the criteria: {}".format(len(data_keys_to_use)))
    # raise
    
    # NOTE: We can perform direct experiments on pre-saved files.
    if (
        args.use_results_file or
        args.use_image_cropped_pickle_file
    ):
        assert not (args.use_results_file and args.use_image_cropped_pickle_file), (
            "`--use_results_file` and `--use_image_cropped_pickle_file` "
            "cannot be both there!"
        )
        if args.use_results_file and not os.path.exists(args.use_results_file):
            print("File {} not found, will generate it.".format(args.results_file))
        elif (
            args.use_image_cropped_pickle_file and
            not os.path.exists(args.use_image_cropped_pickle_file)
        ):
            raise ValueError("Please generate the cropped image pickle file"
                             " first for {}".format(args.use_image_cropped_pickle_file))
        else:
            if args.use_results_file:
                pre_stored_file = args.use_results_file
            elif args.use_image_cropped_pickle_file:
                pre_stored_file = args.use_image_cropped_pickle_file

            all_coco_results_dict = {}
            all_total_ap = {}
            for bbox_inference_strategy in args.bbox_inference_strategy:
                # For COCO results.
                all_coco_results_dict[bbox_inference_strategy] = {
                    "gts": [],
                    "dts": [],
                    "image_id": 0,
                    "id": 0,
                }
                total_ap = perform_lang_aug_scod_on_file(
                    pre_stored_file,
                    bbox_inference_strategy,
                    region_proposal_probings,
                    process_first_k=args.process_first_k,
                    data_keys_to_use=data_keys_to_use,
                    coco_results_dict=all_coco_results_dict[bbox_inference_strategy],
                    verbose=args.verbose,
                    args=args,
                )
                all_total_ap[bbox_inference_strategy] = total_ap
            for bbox_inference_strategy in args.bbox_inference_strategy:
                print("----- {} -----".format(bbox_inference_strategy))
                for ap_key in sorted(all_total_ap[bbox_inference_strategy]):
                    print("{}: {}".format(ap_key, all_total_ap[bbox_inference_strategy][ap_key]))
                get_coco_results(
                    original_coco_gt_file=args.coco_gt_file,
                    coco_gt_file=all_coco_results_dict[bbox_inference_strategy]["gts"],
                    coco_pred_file=all_coco_results_dict[bbox_inference_strategy]["dts"],
                    top_k=args.top_k_bboxes,
                )
            print("Done!")
            exit(-1)

    #  (Adjustable) inference function.
    all_coco_results_dict = {}
    all_total_ap = {}
    for bbox_inference_strategy in args.bbox_inference_strategy:
        # For COCO results.
        all_coco_results_dict[bbox_inference_strategy] = {
            "gts": [],
            "dts": [],
            "image_id": 0,
            "id": 0,
        }
        pred_bbox_res = json.load(open(args.processed_inference_json_file))
        total_ap, overall_inference_metadata = perform_lang_aug_scod(
            inferences=pred_bbox_res,
            keyed_scod_clips=keyed_scod_clips,
            all_narrations=all_narrations,
            narration_pass=args.narration_pass,
            process_first_k=args.process_first_k,
            top_k_bboxes=args.top_k_bboxes,
            region_proposal_probings=region_proposal_probings,
            bbox_inference_strategy=bbox_inference_strategy,
            data_keys_to_use=data_keys_to_use,
            coco_results_dict=all_coco_results_dict[bbox_inference_strategy],
            verbose=args.verbose,
            args=args,
        )
        all_total_ap[bbox_inference_strategy] = total_ap
    for bbox_inference_strategy in args.bbox_inference_strategy:
        print("----- {} -----".format(bbox_inference_strategy))
        for ap_key in sorted(all_total_ap[bbox_inference_strategy]):
            print("{}: {}".format(ap_key, all_total_ap[bbox_inference_strategy][ap_key]))
        get_coco_results(
            original_coco_gt_file=args.coco_gt_file,
            coco_gt_file=all_coco_results_dict[bbox_inference_strategy]["gts"],
            coco_pred_file=all_coco_results_dict[bbox_inference_strategy]["dts"],
            top_k=args.top_k_bboxes,
        )

    if args.output_image_cropped_pickle_file is not None:
        with open(args.output_image_cropped_pickle_file, "wb") as f:
            pickle.dump(overall_inference_metadata, f)
        print("Saving pickle file to: {}".format(
            args.output_image_cropped_pickle_file))
        print("Done!")
        exit(-1)

    json.dump(
        overall_inference_metadata,
        open(args.results_file, "w"),
        indent=4,
    )
    print("Saving results file to: {}".format(args.results_file))

    print("Done!")
