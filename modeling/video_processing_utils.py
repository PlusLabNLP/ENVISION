import re, pprint
from datetime import datetime
import csv, os, sys, re, string, json, glob, shutil, random, math
from collections import OrderedDict

import cv2
import pandas as pd
import numpy as np

from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

import torch
from transformers import AutoTokenizer
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

from run_language_augmenting_bbox_preds import get_scod_clipped_narrations
from run_language_augmenting_bbox_preds import extract_SRL_args


def bg_magenta_text(text):
    return "\x1b[45m" + text + "\x1b[0m"


def extract_ego4d_scod_frames_via_ffmpeg(
    video_uid,
    curr_scod_video_to_frame_cnt,
    curr_videos_root,
    frame_duplicated_video_ids,
    frame_save_folder=None,
    extension="png",
    verbose=False,
):
    frame_cnts = curr_scod_video_to_frame_cnt["frame_cnts"]
    frame_cnts = sorted(list(set(frame_cnts)))
    if len(frame_cnts) == 0:
        if verbose:
            print("Video {} is done!".format(video_uid))
        return None
    if verbose:
        print("To extract {} frames from video: {}".format(
            len(frame_cnts), video_uid))

    ego4d_video_path = os.path.join(
        curr_videos_root,
        "{}.mp4".format(video_uid),
    )

    # NOTE: First we check if the video is corrupted due to duplicated
    # frames, will deal with these later.
    ffmpeg_frame_check_cmd = (
        "/home/telinwu/ffmpeg-git-20220910-amd64-static/ffmpeg "
        "-i {} "
        "-vf select='eq(n\,1005)' "
        "temp.jpg 2>&1 | tee temp.txt > /dev/null".format(
            ego4d_video_path,
        )
    )
    os.system(ffmpeg_frame_check_cmd)
    ffmpeg_log = open("temp.txt").readlines()
    ffmpeg_log = [x.strip() for x in ffmpeg_log]
    frame_duplicated = False
    for line in ffmpeg_log:
        if "More than" in line and "frames duplicated" in line:
            frame_duplicated = True
    os.remove("temp.jpg")
    os.remove("temp.txt")
    if frame_duplicated:
        frame_duplicated_video_ids.append(video_uid)
        dedup_folder = "/local1/telinwu/research/resources/Ego4D/v1/full_scale"
        dedup_ego4d_video_path = os.path.join(
            dedup_folder,
            "{}.mp4".format(video_uid),
        )
        ffmpeg_frame_dedup_cmd = (
            "/home/telinwu/ffmpeg-git-20220910-amd64-static/ffmpeg "
            "-i {} "
            "-vf mpdecimate,setpts=N/FRAME_RATE/TB "
            "-loglevel panic {}".format(
                ego4d_video_path,
                dedup_ego4d_video_path,
            )
        )
        if verbose:
            print("Executing command: {}".format(ffmpeg_frame_dedup_cmd))
        # os.system(ffmpeg_frame_dedup_cmd)
        ego4d_video_path = dedup_ego4d_video_path
        return None

    curr_frame_save_folder = curr_scod_video_to_frame_cnt["folder"]
    if not os.path.exists(curr_frame_save_folder):
        os.makedirs(curr_frame_save_folder)

    # NOTE: If no duplicated frames, extract the frames.
    eq_string = []
    for frame_cnt in frame_cnts:
        eq_string.append("eq(n\,{})".format(frame_cnt))
    eq_string = "\'" + "+".join(eq_string) + "\'"

    ffmpeg_frame_extract_cmd = (
        "/home/telinwu/ffmpeg-git-20220910-amd64-static/ffmpeg "
        "-i {} "
        "-vf select={} -frame_pts true -vsync 0 "
        "-loglevel panic {}/%d.{}".format(
            ego4d_video_path,
            eq_string,
            curr_frame_save_folder,
            extension,
        )
    )
    if verbose:
        print("Executing command: {}".format(ffmpeg_frame_extract_cmd))
    raise
    os.system(ffmpeg_frame_extract_cmd)

    return None


def extract_ego4d_scod_frames_via_cv2(
    video_uid,
    curr_scod_video_to_frame_cnt,
    curr_videos_root,
    frame_duplicated_video_ids,
    frame_save_folder=None,
    extension="png",
    verbose=False,
):
    frame_cnts = curr_scod_video_to_frame_cnt["frame_cnts"]
    frame_cnts = sorted(list(set(frame_cnts)))
    if len(frame_cnts) == 0:
        if verbose:
            print("Video {} is done!".format(video_uid))
        return None
    if verbose:
        print("To extract {} frames from video: {}".format(
            len(frame_cnts), video_uid))

    ego4d_video_path = os.path.join(
        curr_videos_root,
        "{}.mp4".format(video_uid),
    )
    
    video = cv2.VideoCapture(ego4d_video_path)

    curr_frame_save_folder = curr_scod_video_to_frame_cnt["folder"]
    if not os.path.exists(curr_frame_save_folder):
        os.makedirs(curr_frame_save_folder)

    for frame_id in tqdm(frame_cnts, desc="processing: {}".format(video_uid)):
        ret = video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = video.read()
        cv2.imwrite("{}/{}.{}".format(
            curr_frame_save_folder, frame_id, extension), frame)

    return None


def bulk_extract_ego4d_scod_frames(
    scod_data_jsons,
    curr_videos_root,
    frame_save_folder=None,
    extension="png",
    verbose=False,
    videos_list_file=None,
):
    all_scod_clips = []
    for scod_data_json in scod_data_jsons:
        all_scod_clips += json.load(open(scod_data_json))["clips"]

    frame_duplicated_video_ids = []

    scod_video_to_frame_cnt = OrderedDict()
    total_img_cnt = 0
    exist_img_cnt = 0
    for scod_clip in tqdm(all_scod_clips, desc="Getting Video Info"):

        video_uid = scod_clip["video_uid"]
        curr_frame_save_folder = os.path.join(
            frame_save_folder, video_uid)
        if video_uid not in scod_video_to_frame_cnt:
            scod_video_to_frame_cnt[video_uid] = {
                "folder": curr_frame_save_folder,
                "frame_cnts": [],
            }

        frames_to_extract = []
        for frame_name in ["pre_frame", "pnr_frame", "post_frame"]:
            frame_cnt = scod_clip[frame_name]["frame_number"]
            frame_save_name = os.path.join(
                curr_frame_save_folder,
                "{}.{}".format(frame_cnt, extension)
            )
            if not os.path.exists(frame_save_name):
                frames_to_extract.append(frame_cnt)
            else:
                exist_img_cnt += 1
            total_img_cnt += 1
        scod_video_to_frame_cnt[video_uid]["frame_cnts"] += frames_to_extract

    print("{} / {} ({:.3f}%) extracted.".format(
        exist_img_cnt, total_img_cnt, exist_img_cnt/total_img_cnt*100.))
    # videos_to_be_extract = 0
    # for video_uid in tqdm(scod_video_to_frame_cnt, desc="SCOD Frame Extracting"):
    #     if len(scod_video_to_frame_cnt[video_uid]["frame_cnts"]) > 0:
    #         print(video_uid)
    #         videos_to_be_extract += 1
    # print(videos_to_be_extract)
    # raise
    if videos_list_file is not None:
        allowed_video_uids = open(videos_list_file).readlines()
        allowed_video_uids = [l.strip() for l in allowed_video_uids]
        for video_uid in sorted(list(scod_video_to_frame_cnt.keys())):
            if video_uid not in allowed_video_uids:
                del scod_video_to_frame_cnt[video_uid]
    raise

    for video_uid in tqdm(scod_video_to_frame_cnt, desc="SCOD Frame Extracting"):
        curr_scod_video_to_frame_cnt = scod_video_to_frame_cnt[video_uid]
        # extract_ego4d_scod_frames_via_ffmpeg(
        extract_ego4d_scod_frames_via_cv2(
            video_uid,
            curr_scod_video_to_frame_cnt,
            curr_videos_root,
            frame_duplicated_video_ids,
            frame_save_folder=frame_save_folder,
            extension=extension,
            verbose=verbose,
        )

    if len(frame_duplicated_video_ids) > 0:
        frame_duplicated_out_file = "scod_video_frames_duplicated.txt"
        frame_duplicated_out_file = open(frame_duplicated_out_file, "w")
        for video_uid in frame_duplicated_video_ids:
            frame_duplicated_out_file.write(video_uid+"\n")
        frame_duplicated_out_file.close()

    return None


def convert_png_to_jpg(
    root_folder,
    src_scod_images_folder,
    dsc_scod_images_folder,
):
    video_uids = os.listdir(os.path.join(root_folder, src_scod_images_folder))
    for video_uid in tqdm(video_uids, desc="Converting PNG to JPG"):
        images = glob.glob(os.path.join(root_folder,
            src_scod_images_folder, video_uid, "*.png"))
        for src_img_file in images:
            src_img_name = src_img_file.split("/")[-1]
            dst_img_folder = os.path.join(root_folder,
                dsc_scod_images_folder, video_uid)
            if not os.path.exists(dst_img_folder):
                os.makedirs(dst_img_folder)
            dst_img_file = os.path.join(dst_img_folder, src_img_name.replace(
                "png", "jpg"))
            src_img = Image.open(src_img_file)
            dst_img = src_img.save(dst_img_file)
        pass
    pass

    return None


def stats_on_frames_of_data(
    scod_data_jsons_dict,
    scod_coco_jsons,
    all_narrations=None,
    curr_videos_root=None,
    narration_pass="narration_pass_1",
):
    assert type(scod_data_jsons_dict) is dict

    scod_data = {}
    scod_data_info = {}
    narration_info = {}
    ooc_info = {}

    for split in scod_data_jsons_dict:
        scod_data_json = scod_data_jsons_dict[split]
        if scod_data_json is None:
            continue
        scod_data[split] = json.load(open(scod_data_json))["clips"]
        scod_data_info[split] = {}

        if all_narrations is not None:
            narration_info[split] = {
                "no_narration": 0,
                "with_narration": 0,
                "gt_noun_in_narration_pre": 0,
                "gt_noun_in_narration_pnr": 0,
                "gt_noun_in_narration_post": 0,
                "gt_noun_not_in_narration_pre": 0,
                "gt_noun_not_in_narration_pnr": 0,
                "gt_noun_not_in_narration_post": 0,
            }

    for split in scod_data:
        split_scod_data = scod_data[split]
        ooc_info[split] = {
            "nope_ooc": 0,
            "with_ooc": 0,
            "nope_tool": 0,
            "with_tool": 0,
        }
        for scod in tqdm(split_scod_data, desc="Stats-for-{}".format(split)):
            video_uid = scod["video_uid"]
            if video_uid not in scod_data_info:
                scod_data_info[video_uid] = {}

            narration = None
            if all_narrations is not None:
                clip_narrations, closest_narrations = get_scod_clipped_narrations(
                    scod,
                    all_narrations,
                    curr_videos_root,
                    narration_pass=narration_pass,
                    top_k=5,
                    anchor_frame="pnr",
                    verbose=False,
                    get_video=False,
                )
                if len(closest_narrations) > 0:
                    narration = closest_narrations[0][-1]
                    narration = refine_or_trim_narration(narration)
                    narration_info[split]["with_narration"] += 1
                else:
                    narration_info[split]["no_narration"] += 1

            for frame in ["pre", "pnr", "post"]:
                f = scod["{}_frame".format(frame)]
                frame_num = scod["{}_frame".format(frame)]["frame_number"]
                if frame_num not in scod_data_info[video_uid]:
                    scod_data_info[video_uid][frame_num] = []
                scod_data_info[video_uid][frame_num].append("{}_frame".format(frame))
                bbox_info = f["bbox"]
                yes_ooc = False
                yes_tool = False
                for curr_bbox_info in bbox_info:
                    if curr_bbox_info["object_type"] == "object_of_change":
                        yes_ooc = True
                        # break
                    if curr_bbox_info["object_type"] == "tool":
                        yes_tool= True
                if yes_ooc:
                    ooc_info[split]["with_ooc"] += 1
                else:
                    ooc_info[split]["nope_ooc"] += 1
                if yes_tool:
                    ooc_info[split]["with_tool"] += 1
                else:
                    ooc_info[split]["nope_tool"] += 1
                if narration is not None:
                    gt_structured_noun = None
                    for curr_bbox_info in bbox_info:
                        if curr_bbox_info["object_type"] == "object_of_change":
                            gt_structured_noun = curr_bbox_info["structured_noun"]
                            break
                        pass
                    if gt_structured_noun is None:
                        narration_info[split]["gt_noun_not_in_narration_{}".format(frame)] += 1
                    else:
                        gt_nouns = parse_ego4d_scod_structured_noun(gt_structured_noun)
                        narration_tokens = narration.split()
                        gt_noun_in_narration = False
                        for t in narration_tokens:
                            for gt_noun in gt_nouns:
                                if gt_noun in t:
                                    gt_noun_in_narration = True
                                    break
                                pass
                            pass
                            if gt_noun_in_narration:
                                break
                        if gt_noun_in_narration:
                            narration_info[split]["gt_noun_in_narration_{}".format(frame)] += 1
                        else:
                            narration_info[split]["gt_noun_not_in_narration_{}".format(frame)] += 1
                    ####
                ####
            ####
        ####

    scod_coco_data = []
    for scod_coco_json in scod_coco_jsons:
        scod_coco_data += json.load(open(scod_coco_json))["images"]

    stats = {"{}_frame".format(frame): 0 for frame in ["pre", "pnr", "post"]}
    for scod_coco in scod_coco_data:
        f = scod_coco["file_name"].split(".")[0]
        video_uid, frame_name = f.split("/")
        frame_name = int(frame_name)
        assert video_uid in scod_data_info and frame_name in scod_data_info[video_uid], (
            "Video ID: {}  Frame Num: {}".format(video_uid, frame_name)
        )
        for frame in scod_data_info[video_uid][frame_name]:
            stats[frame] += 1

    print("--- Stats ----")
    pprint.pprint(stats)
    for split in ["train", "val", "test"]:
        if split not in ooc_info: continue
        print("---- {} ----".format(split))
        pprint.pprint(ooc_info[split])
        if all_narrations is not None:
            pprint.pprint(narration_info[split])
    return None


def extract_ooc_spans(
    narration,
    possible_gt_nouns,
    additional_functions=None,
    tokens_positive_method=None,
    extract_tool_phrase=False,
    force_gt=False,
):
    new_narration = None
    ooc_spans = []

    if (
        ("random_word" == tokens_positive_method
         or "first_word" == tokens_positive_method)
        and not extract_tool_phrase
        and not force_gt
    ):
        tokens = narration.split()
        if "first" in tokens_positive_method:
            random_token_idx = 0
        else:
            random_token_idx = np.random.randint(0, len(tokens))
        curr_beg, curr_end = 0, 0
        for i in range(len(tokens)):
            token = tokens[i]
            curr_end = curr_beg + len(token)
            if i == random_token_idx:
                break
            if i < len(tokens) - 1:
                curr_beg += len(token) + 1
        ooc_spans = [[curr_beg, curr_end]]
        new_narration = " ".join(tokens)
        return ooc_spans, new_narration
    
    gt_included = False
    if (
        "gt" in tokens_positive_method
        or extract_tool_phrase
        or force_gt
    ):
        w_sep_tokens = narration.split(" ")
        curr_beg, curr_end = 0, 0
        span_word_mapping = []
        # print(narration, w_sep_tokens)
        for t_idx in range(len(w_sep_tokens)):
            t = w_sep_tokens[t_idx]
            if len(t) == 0:
                wo_punct_curr_end = curr_beg + len(t)
                print("Weird narration: {} with token `{}`.".format(narration, t))
            elif t[-1].isalnum():
                wo_punct_curr_end = curr_beg + len(t)
            else:
                wo_punct_curr_end = curr_beg + len(t) - 1 # Puncts.
            span_word_mapping.append((
                curr_beg, wo_punct_curr_end, t
            ))
            # The last token does not have white space.
            if t_idx == len(w_sep_tokens) - 1:
                curr_beg += len(t)
            else:
                curr_beg += len(t) + 1  # Account for white space.

        span_beg, span_end = None, None
        for b, e, w in span_word_mapping:
            gt_noun_found = False
            for possible_gt_noun in possible_gt_nouns:
                if possible_gt_noun in w:
                    gt_noun_found = True
                    break
            if gt_noun_found:
                span_beg, span_end = b, e
                # FIXME: For now, we only take the first occurence of GT noun.
                break
        if span_beg is not None and span_end is not None:
            ooc_spans.append([span_beg, span_end])
            gt_included = True

        # FIXME: For tool we only consider gt inclusion now.
        if extract_tool_phrase or force_gt:
            return ooc_spans, narration

    # NOTE: If no easy way to use GT to find the spans, or indicated not to use
    # the GT at all.
    if (
        (
            len(ooc_spans) <= 0
            or ("srl" in tokens_positive_method and "only" in tokens_positive_method)
            or ("v_" in tokens_positive_method)
        )
        and not tokens_positive_method == "ground_full_sentence"
    ):
        no_func_avail = True
        for func_key in additional_functions:
            if additional_functions[func_key] is not None:
                no_func_avail = False
        if no_func_avail:
            raise ValueError(
                "No `additional_functions` available for"
                " extracting the proper phrases to be grounded"
                " in the narration. Consider using"
                " `srl` or `dp`, or `nltk`."
            )
        if (
            "srl" in additional_functions
            and additional_functions["srl"] is not None
            and "srl" in tokens_positive_method
        ):
            new_narration = None
            srl_model = additional_functions["srl"]
            srl_res = srl_model.predict(sentence=narration)
            parsed_args_list = extract_SRL_args(srl_res)
            # `parsed_args` are ordered dict.
            # Getting the right SRL args.
            parsed_args_to_use = None
            for parsed_args, srl in parsed_args_list:
                found_parsed_args = True
                for _arg in ["ARG1"]:
                    if _arg not in parsed_args:
                        found_parsed_args = False
                    pass
                if found_parsed_args:
                    parsed_args_to_use = parsed_args
                    break
                pass
            # Currently we only consider ARG1.
            if parsed_args_to_use is not None:
                span_beg, span_end = None, None
                curr_beg, curr_end = 0, 0
                new_narration = ""
                for _arg in parsed_args_to_use:
                    seg = parsed_args_to_use[_arg]
                    curr_end = curr_beg + len(seg)
                    # For `V` tag.
                    if _arg == "V" and "v_" in tokens_positive_method:
                        span_beg, span_end = curr_beg, curr_end
                        ooc_spans.append([span_beg, span_end])
                    if _arg == "ARG1" and not gt_included:
                        span_beg, span_end = curr_beg, curr_end
                        ooc_spans.append([span_beg, span_end])
                    new_narration += seg + " "
                    curr_beg += len(seg) + 1
                ooc_spans = sorted(ooc_spans)
                return ooc_spans, new_narration
        else:
            raise NotImplementedError("Other `additional_functions` is not done yet!")

    # If still none of above function can deal with the narration,
    # simply just use the entire narration, or when `tokens_positive_method` is
    # set to be `ground_full_sentence`.
    if (
        len(ooc_spans) <= 0
        or tokens_positive_method == "ground_full_sentence"
    ):
        ooc_spans = [[0, len(narration)]]
        new_narration = None
        return ooc_spans, new_narration

    return ooc_spans, new_narration


def refine_or_trim_narration(
        narration,
        remove_hand_desc=False,
    ):
    tokens = narration.split()
    character_symbol = None
    if "#" in tokens[0]:
        character_symbol = tokens[0].split("#")[-1]
    filtered_tokens = []
    for t in tokens:
        if "#" in t or (character_symbol is not None and t == character_symbol):
            continue
        if len(t) == 0:
            continue
        filtered_tokens.append(t)
    narration = " ".join(filtered_tokens)
    narration = narration.replace("#C C", "A person")
    narration = narration.replace("#Unsure", "")
    narration = narration.replace("#unsure", "")
    narration_tokens = narration.split()  # Remove white excessive white spaces.
    narration_tokens = [t for t in narration_tokens if len(t) != 0]
    narration = " ".join(narration_tokens)
    narration = narration.strip()
    if character_symbol is not None and character_symbol.lower() == "c":
        narration = "Someone " + narration
    else:
        print("No character symbol: {}".format(narration))
    if remove_hand_desc:
        # regex_str = "\ (?:with|using)\ ([a-zA-Z]+ )+(?:hands|hand|arms|arm)"
        regex_str = (
            "\ (?:in|with|using|into|from)\ (?:his|her|the|their|both|a)?\s*"
            "(?:gloved)?\s*(?:right|left|two|both|of)?\s*"
            "(?:his|her|the|their|gloved)?\s*"
            "(?:hands|hand|arms|arm)"
        )
        m = re.search(regex_str, narration)
        while m is not None:
            hand_str = m.group()
            narration = narration.replace(hand_str, "")
            m = re.search(regex_str, narration)
    return narration


def parse_ego4d_scod_structured_noun(s):
    if s is None:
        return []
    main_noun = s.split("(")[0]
    other_nouns = []
    if "(" in s and ")" in s:
        candidates = s.split("(")
        for j in range(len(candidates)):
            other_nouns += s.split("(")[j].split(")")[0].split(",")
    all_nouns = [main_noun] + other_nouns
    # all_nouns = [x.replace("_", "") for x in all_nouns]
    all_nouns = [" ".join(x.split("_")).strip() for x in all_nouns]
    all_nouns = sorted(list(set(all_nouns)))
    return all_nouns


def convert_scod_to_coco_format(
    scod_train_file=None,
    scod_val_file=None,
    scod_test_file=None,
    frames=["pre", "pnr", "post"],
    root_image_folder=None,
    image_extension=".jpg",
    output_folder=None,
    with_narration=False,
    narration_pass="narration_pass_1",
    all_narrations=None,
    curr_videos_root=None,
    output_annotation_suffix=None,
    additional_functions=None,
    tokens_positive_method=None,
    force_data_completeness=False,
):
    assert image_extension in [".png", ".jpg"]
    assert root_image_folder is not None
    assert output_folder is not None
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    data_dict = {}
    if scod_train_file is not None:
        data_dict["train"] = scod_train_file
    if scod_val_file is not None:
        data_dict["val"] = scod_val_file
    if scod_test_file is not None:
        data_dict["test"] = scod_test_file
    for split in data_dict:
        data_dict[split] = json.load(open(data_dict[split]))

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")

    coco_data_info = {                                                                         
        "description": None,
        "version": "1.0.0",
        "year": 2023,
        "contributor": "Alan Turing",
        "date_created": dt_string,
    }

    licenses = [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }
    ]

    categories = [
        {
            "id": 1,
            "name": "object_of_change",
            "supercategory": "object"
        }
    ]
    if "with_tool" in tokens_positive_method:
        categories.append({
            "id": 2,
            "name": "tool",
            "supercategory": "object"
        })

    for split in data_dict:
        proceed = True

        if output_annotation_suffix is not None:
            output_file = os.path.join(output_folder, "{}_{}.json".format(
                split, output_annotation_suffix))
        else:
            output_file = os.path.join(output_folder, "{}.json".format(split))
        if os.path.exists(output_file):
            to_proceed = input(
                "File {} exists! Proceed to overwrite (y/n)? ".format(output_file))
            if to_proceed.lower() == "y":
                proceed = True
            else:
                proceed = False
        if not proceed:
            print("File {} exists, skipping!".format(output_file))
            continue
        print("Will be saving to {}".format(output_file))

        description = "Ego4D Soft-launch v1 {} {} Objects".format(
            split, "-".join(frames)
        )
        coco_data_info["description"] = description

        output_data = {}
        output_data["info"] = coco_data_info.copy()
        output_data["licenses"] = licenses.copy()
        output_data["categories"] = categories.copy()
        output_data["images"] = []
        if split != "test":
            output_data["annotations"] = []

        data = data_dict[split]["clips"]
        image_cnt = 0
        image_id_cnt = 0
        image_bbox_id_cnt = 0
        with_bbox_cnt = 0
        without_bbox_cnt = 0
        for d in tqdm(data, desc="Processing SCOD {}".format(split)):
            video_uid = d["video_uid"]

            if with_narration:
                clip_narrations, closest_narrations = get_scod_clipped_narrations(
                    d,
                    all_narrations,
                    curr_videos_root,
                    narration_pass=narration_pass,
                    top_k=5,
                    anchor_frame="pnr",
                    verbose=False,
                    get_video=False,
                )

            # FIXME: probably cannot just skip no-narration ones?
            if (
                with_narration
                and len(closest_narrations) <= 0
                and not force_data_completeness
            ):
                image_cnt += len(frames)  # We will skip X frames.
                continue
            elif (
                with_narration
                and len(closest_narrations) <= 0
                and force_data_completeness
            ):
                closest_narrations = [["object_of_change"]]

            # NOTE: Find the bbox with annotated structured noun.
            caption, ooc_spans, ooc_spans_eval = None, None, None
            if with_narration and tokens_positive_method != "with_tool_only":
                narration = closest_narrations[0][-1]
                narration = refine_or_trim_narration(
                    narration,
                    remove_hand_desc="no_hand" in tokens_positive_method,
                )
                for fr in frames:
                    fr_d = d["{}_frame".format(fr)]
                    if "bbox" not in fr_d:  # Test-set.
                        continue
                    bbox_info = fr_d["bbox"]
                    for bbox_d in bbox_info:
                        # NOTE: narration and gt objects.
                        if bbox_d["object_type"] == "object_of_change":
                            curr_ooc = bbox_d["structured_noun"]
                            possible_gt_nouns = parse_ego4d_scod_structured_noun(curr_ooc)
                            ooc_spans, new_narration = extract_ooc_spans(
                                narration,
                                possible_gt_nouns,
                                additional_functions=additional_functions,
                                tokens_positive_method=tokens_positive_method,
                            )
                            if len(ooc_spans) == 0:
                                raise
                            ooc_spans_eval = ooc_spans
                            if new_narration is not None:
                                caption = new_narration
                            else:
                                caption = narration
                            break
                        pass
                    pass
                    # As long as we find the ooc and proper caption, stop.
                    # Does not have to go through all frames.
                    if with_narration and caption is not None:
                        break
                    pass
                pass
            elif with_narration and tokens_positive_method == "with_tool_only":
                narration = closest_narrations[0][-1]
                narration = refine_or_trim_narration(
                    narration,
                    remove_hand_desc="no_hand" in tokens_positive_method,
                )
                caption = narration
            pass
            ############

            for fr in frames:
                fr_d = d["{}_frame".format(fr)]
                frame_number = fr_d["frame_number"]
                height, width = fr_d["height"], fr_d["width"]
                file_name = os.path.join(video_uid, str(frame_number))
                file_name += image_extension
                assert os.path.exists(os.path.join(root_image_folder, file_name)), (
                    "Image {} not exist under {}!".format(file_name, root_image_folder)
                )

                ego4d_scod_id = "{}_{}_{}_{}".format(video_uid, frame_number, fr, image_cnt)
                image_dict = {
                    "file_name": file_name,
                    "height": height,
                    "width": width,
                    "id": image_id_cnt,
                    "ego4d_scod_id": ego4d_scod_id,
                }

                curr_narration_contains_tool = False
                if "bbox" in fr_d:  # Train-and-Val-sets.
                    bbox_info = fr_d["bbox"]
                    if "tool" in tokens_positive_method:
                        _all_tool_spans = []
                        for bbox_d in bbox_info:
                            if bbox_d["object_type"] == "tool":
                                # note: narration and tool objects.
                                tool = bbox_d["structured_noun"]
                                possible_gt_tools = parse_ego4d_scod_structured_noun(tool)
                                tool_spans, _ = extract_ooc_spans(
                                    narration,
                                    possible_gt_tools,
                                    additional_functions=additional_functions,
                                    tokens_positive_method=tokens_positive_method,
                                    extract_tool_phrase=True,
                                )
                                _all_tool_spans += tool_spans
                            pass
                        pass
                        if len(_all_tool_spans) > 0:
                            curr_narration_contains_tool = True
                    pass

                    # NOTE: Log bbox annotations.
                    curr_bbox_d = None
                    for bbox_d in bbox_info:
                        if bbox_d["object_type"] == "object_of_change":
                            curr_bbox_d = bbox_d["bbox"]
                            curr_bbox = [
                                curr_bbox_d["x"],     curr_bbox_d["y"],
                                curr_bbox_d["width"], curr_bbox_d["height"],
                            ]
                            curr_area = curr_bbox[2] * curr_bbox[3]
                            # NOTE: For now, only class is `object_of_change`.
                            category_id = 1
                            # Annotation dict.
                            annot_dict = {
                                "segmentation": [],
                                "area": curr_area,
                                "iscrowd": 0,
                                "ignore": 0,
                                "bbox": curr_bbox,
                                "category_id": category_id,
                                "id": image_bbox_id_cnt,
                                "image_id": image_id_cnt,
                                "ego4d_scod_id": ego4d_scod_id,
                            }
                            if with_narration:
                                annot_dict["tokens_positive"] = ooc_spans
                            if "with_tool_strict" in tokens_positive_method and curr_narration_contains_tool:
                                output_data["annotations"].append(annot_dict)
                                image_bbox_id_cnt += 1
                            elif "with_tool_strict" in tokens_positive_method and not curr_narration_contains_tool:
                                pass  # do nothing.
                            elif tokens_positive_method != "with_tool_only":
                                output_data["annotations"].append(annot_dict)
                                image_bbox_id_cnt += 1
                            # break
                        pass
                    pass

                    if curr_bbox_d is None:
                        without_bbox_cnt += 1
                    else:
                        with_bbox_cnt += 1

                    # NOTE: For tools!
                    if "with_tool" in tokens_positive_method:
                        all_tool_spans = []
                        curr_bbox_d = None
                        for bbox_d in bbox_info:
                            if bbox_d["object_type"] == "tool":
                                # note: narration and tool objects.
                                tool = bbox_d["structured_noun"]
                                possible_gt_tools = parse_ego4d_scod_structured_noun(tool)
                                tool_spans, _ = extract_ooc_spans(
                                    narration,
                                    possible_gt_tools,
                                    additional_functions=additional_functions,
                                    tokens_positive_method=tokens_positive_method,
                                    extract_tool_phrase=True,
                                )
                                # If no gt tool found in narration, just use all.
                                if len(tool_spans) == 0:
                                    tool_spans = [[0, len(narration)]]
                                all_tool_spans += tool_spans
                                curr_bbox_d = bbox_d["bbox"]
                                curr_bbox = [
                                    curr_bbox_d["x"],     curr_bbox_d["y"],
                                    curr_bbox_d["width"], curr_bbox_d["height"],
                                ]
                                curr_area = curr_bbox[2] * curr_bbox[3]
                                # NOTE: For now, tool class is always just `tool`.
                                category_id = 2
                                # Annotation dict.
                                annot_dict = {
                                    "segmentation": [],
                                    "area": curr_area,
                                    "iscrowd": 0,
                                    "ignore": 0,
                                    "bbox": curr_bbox,
                                    "category_id": category_id,
                                    "id": image_bbox_id_cnt,
                                    "image_id": image_id_cnt,
                                    "ego4d_scod_id": ego4d_scod_id,
                                }

                                if with_narration:
                                    annot_dict["tokens_positive"] = tool_spans
                                if "with_tool_strict" in tokens_positive_method and curr_narration_contains_tool:
                                    output_data["annotations"].append(annot_dict)
                                    image_bbox_id_cnt += 1
                                elif "with_tool_strict" in tokens_positive_method and not curr_narration_contains_tool:
                                    pass  # do nothing.
                                else:
                                    output_data["annotations"].append(annot_dict)
                                    image_bbox_id_cnt += 1
                            pass
                        pass
                # NOTE: Annotations need to come from either LLMs for SRL!
                else:  # Test-set
                    # FIXME: change the below:
                    if "srl_arg1_only" in tokens_positive_method:
                        ooc_spans, new_narration = extract_ooc_spans(
                            narration,
                            possible_gt_nouns=None,
                            additional_functions=additional_functions,
                            tokens_positive_method=tokens_positive_method,
                        )
                        ooc_spans_eval = ooc_spans
                        if new_narration is not None:
                            caption = new_narration
                        else:
                            caption = narration
                    pass

                if with_narration:
                    assert caption is not None
                    image_dict["caption"] = caption
                    image_dict["closest_narrations"] = closest_narrations
                    image_dict["tokens_positive_eval"] = [ooc_spans_eval]
                    if "with_tool_strict" in tokens_positive_method and not curr_narration_contains_tool:
                        continue
                    elif tokens_positive_method == "with_tool_only" and len(all_tool_spans) == 0:
                        continue
                    elif tokens_positive_method == "with_tool_only":
                        image_dict["tokens_positive_eval"] = [all_tool_spans, all_tool_spans]
                    elif "with_tool" in tokens_positive_method and len(all_tool_spans) > 0:
                        image_dict["tokens_positive_eval"] = \
                            [ooc_spans_eval] + [all_tool_spans]
                output_data["images"].append(image_dict)

                image_cnt += 1
                image_id_cnt += 1

            # if False:
            # if tool_cnt > 1:
            #     pprint.pprint(output_data)
            #     raise
            # break

        pass
        json.dump(
            output_data,
            open(output_file, "w"),
            indent=4,
        )
        print("Saving {} ({} data points) file to {}".format(
            split, len(output_data["images"]), output_file))
        print("With bbox: {}  Without bbox {}".format(with_bbox_cnt, without_bbox_cnt))

    return None


def convert_scod_bbox_to_clip_caption_data(
    scod_file=None,
    frames=["pre", "pnr", "post"],
    root_image_folder=None,
    image_extension=".jpg",
    output_image_folder=None,
    output_annotation_folder=None,
    output_annotation_suffix=None,
    narration_pass="narration_pass_1",
    all_narrations=None,
    curr_videos_root=None,
    enlarge_bbox_cropping_width=None,
    enlarge_bbox_cropping_height=None,
    additional_functions=None,
):
    assert image_extension in [".png", ".jpg"]
    assert root_image_folder is not None
    assert output_image_folder is not None
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    assert output_annotation_folder is not None
    if not os.path.exists(output_annotation_folder):
        os.makedirs(output_annotation_folder)

    assert narration_pass in [
        "narration_pass_1",
        "narration_pass_2",
    ]
    
    scod_file_name = scod_file.split("/")[-1].split(".")[0]
    new_file_name = scod_file_name + ".csv"
    if output_annotation_suffix is not None:
        new_file_name = scod_file_name+"_{}.csv".format(output_annotation_suffix)
    output_annotation_file = os.path.join(
        output_annotation_folder,
        new_file_name,
    )
    csv_writer = csv.DictWriter(
        open(output_annotation_file, "w"),
        fieldnames=["image_path", "caption"],
    )
    csv_writer.writeheader()

    data = json.load(open(scod_file))["clips"]

    for d in tqdm(data, desc="SCOD To CLIP-Caption"):
        video_uid = d["video_uid"]

        curr_output_image_folder = os.path.join(output_image_folder, video_uid)
        if not os.path.exists(curr_output_image_folder):
            os.makedirs(curr_output_image_folder)

        clip_narrations, closest_narrations = get_scod_clipped_narrations(
            d,
            all_narrations,
            curr_videos_root,
            narration_pass=narration_pass,
            top_k=5,
            anchor_frame="pnr",
            verbose=False,
        )
        if len(closest_narrations) <= 0:
            continue

        narration = closest_narrations[0][-1]
        narration = refine_or_trim_narration(narration)

        if "srl" in additional_functions and additional_functions["srl"] is not None:
            srl_model = additional_functions["srl"]
            srl_res = srl_model.predict(sentence=narration)
            parsed_args_list = extract_SRL_args(srl_res)
            srl_text = []
            for parsed_args, srl in parsed_args_list:
                for _arg in ["V", "ARG1"]:
                    if _arg in parsed_args:
                        srl_text.append(parsed_args[_arg])
                pass
            srl_text = " ".join(srl_text)
            pass
            narration = srl_text

        for fr in frames:
            fr_d = d["{}_frame".format(fr)]
            frame_number = fr_d["frame_number"]
            height, width = fr_d["height"], fr_d["width"]
            bbox_info = fr_d["bbox"]
            file_name = os.path.join(video_uid, str(frame_number))
            file_name += image_extension
            file_path = os.path.join(root_image_folder, file_name)
            assert os.path.exists(file_path), (
                "Image {} not exist under {}!".format(file_name, root_image_folder)
            )

            curr_bbox_d = None
            for bbox_d in bbox_info:
                if bbox_d["object_type"] == "object_of_change":
                    curr_bbox_d = bbox_d
            if curr_bbox_d is None:
                continue
            curr_bbox = curr_bbox_d["bbox"]
            
            # curr_bbox["x"] = max(0, curr_bbox["x"])
            # curr_bbox["y"] = max(0, curr_bbox["y"])

            curr_bbox = [
                curr_bbox["x"], curr_bbox["y"],
                curr_bbox["width"], curr_bbox["height"],
            ]

            # print(file_name)
            # print(curr_bbox)

            #######################################################################################
            # NOTE: Enlarging the bbox.
            image_width, image_height = Image.open(file_path).size

            if enlarge_bbox_cropping_width is None:
                enlarge_bbox_cropping_height = enlarge_bbox_cropping_width
            elif enlarge_bbox_cropping_width < 1.0:
                enlarge_bbox_cropping_width = int(enlarge_bbox_cropping_width * curr_bbox[2])
                enlarge_bbox_cropping_height = int(enlarge_bbox_cropping_height * curr_bbox[3])
            elif enlarge_bbox_cropping_width is not None:
                enlarge_bbox_cropping_height = enlarge_bbox_cropping_width
            else:
                pass

            org_bbox = curr_bbox.copy()

            if enlarge_bbox_cropping_width is not None:
                curr_bbox[0] = max(0, org_bbox[0] - enlarge_bbox_cropping_width)
                curr_bbox[1] = max(0, org_bbox[1] - enlarge_bbox_cropping_height)
                curr_bbox[2] = min(image_width-org_bbox[0], org_bbox[2]+2*enlarge_bbox_cropping_width)
                curr_bbox[3] = min(image_height-org_bbox[1], org_bbox[3]+2*enlarge_bbox_cropping_height)
            # print(curr_bbox)
            # raise
            #######################################################################################

            # Cropping images.
            cropped_bboxed_frame_image = Image.open(file_path).crop((
                curr_bbox[0],              curr_bbox[1],
                curr_bbox[0]+curr_bbox[2], curr_bbox[1]+curr_bbox[3]
            ))
            # print(cropped_bboxed_frame_image.size)

            output_image_file = os.path.join(
                curr_output_image_folder,
                "{}_bbox.jpg".format(frame_number)
            )
            cropped_bboxed_frame_image.save(output_image_file)
            # print(output_image_file)
            row = {
                "image_path": output_image_file,
                "caption": narration,
            }
            # pprint.pprint(row)
            csv_writer.writerow(row)
            # raise

    print("Saving all cropped bbox images under: {}".format(output_image_folder))
    print("Saving annotation file to: {}".format(output_annotation_file))

    return None
        

if __name__ == "__main__":

    def get_srl_func(additional_functions):
        from allennlp_models import pretrained
        from allennlp.predictors.predictor import Predictor
        predictor_srl = pretrained.load_predictor(
            "structured-prediction-srl-bert",
        )
        additional_functions["srl"] = predictor_srl
        output_annotation_suffix = "srl_arg1"
        return output_annotation_suffix

    annots_root = "/local1/telinwu/research/resources/Ego4D/ego4d_data/v1/annotations/"
    scod_train_file = "/local1/telinwu/research/resources/Ego4D/ego4d_data/v1/annotations/fho_scod_train.json"
    scod_val_file = "/local1/telinwu/research/resources/Ego4D/ego4d_data/v1/annotations/fho_scod_val.json"
    scod_data_jsons = [
        scod_train_file,
        scod_val_file,
    ]
    scod_coco_jsons = [
        # "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/coco_annotations/train.json",
        "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/coco_annotations_all_frames/train.json",
        # "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/coco_annotations/val.json",
        "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/coco_annotations_all_frames/val.json",
    ]
    curr_videos_root = "/local1/hu528/ego4d_data/v1/full_scale/"
    frame_save_folder = "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/pre_pnr_post_frames"

    all_narrations_file = os.path.join(annots_root, "narration.json")
    all_narrations = json.load(open(all_narrations_file))


    ###########################################################################
    # NOTE: Statistics and png to jps conversions.
    """
    scod_data_jsons_dict = {
        "train": scod_train_file,
        "val": scod_val_file,
        "test": None,
    }

    stats_on_frames_of_data(
        scod_data_jsons_dict,
        scod_coco_jsons,
        all_narrations=all_narrations,
        curr_videos_root=curr_videos_root,
        narration_pass="narration_pass_1",
    )
    exit(-1)

    # convert_png_to_jpg(
    #     root_folder="/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge",
    #     src_scod_images_folder="pre_pnr_post_png_frames",
    #     dsc_scod_images_folder="pre_pnr_post_frames",
    # )
    # raise
    """
    ###########################################################################


    ###########################################################################
    # NOTE: Extracting pre/pnr/post-frames from ego4d videos.
    """
    scod_test_file = "/local1/telinwu/research/resources/Ego4D/ego4d_data/v1/annotations/fho_scod_test_unannotated.json"
    frame_save_folder = "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/pre_pnr_post_frames"
    scod_data_jsons = [
        scod_test_file,
    ]

    videos_list_file = None
    # videos_list_file = "scripts/ego4d_scod_test_video_list1.txt"
    # videos_list_file = "scripts/ego4d_scod_test_video_list2.txt"
    # videos_list_file = "scripts/ego4d_scod_test_video_list3.txt"

    bulk_extract_ego4d_scod_frames(
        scod_data_jsons,
        curr_videos_root,
        frame_save_folder=frame_save_folder,
        # extension="png",
        extension="jpg",
        verbose=False,
        # verbose=True,
        videos_list_file=videos_list_file,
    )
    exit(-1)
    """
    ###########################################################################


    ###########################################################################
    # NOTE: Create COCO-like json file datasets.

    # TODO: Args.
    with_narration = True
    only_on_test_file = True
    scod_test_file = "/local1/telinwu/research/resources/Ego4D/ego4d_data/v1/annotations/fho_scod_test_unannotated.json"
    # with_narration = False
    # tokens_positive_method = "v_gt_srl_arg1_no_hand"
    # tokens_positive_method = "v_gt_srl_arg1"
    # tokens_positive_method = "gt_srl_arg1_no_hand"
    # tokens_positive_method = "gt_srl_arg1_with_tool_strict"
    # tokens_positive_method = "gt_srl_arg1"
    # tokens_positive_method = "gt_srl_arg1_with_tool"
    # tokens_positive_method = "with_tool_only"
    # tokens_positive_method = "srl_arg1_only"
    tokens_positive_method = "srl_arg1_only_no_hand"
    # tokens_positive_method = "ground_full_sentence"
    # tokens_positive_method = "random_word"
    # tokens_positive_method = "first_word"
    # tokens_positive_method = None
    # frames_to_use = ["pre", "pnr", "post"]
    frames_to_use = ["pnr"]

    # Process.
    assert tokens_positive_method in [
        None,
        "v_gt_srl_arg1_no_hand",
        "v_gt_srl_arg1",
        "gt_srl_arg1_no_hand",
        "gt_srl_arg1_with_tool_strict",
        "gt_srl_arg1",
        "gt_srl_arg1_with_tool",
        "with_tool_only",
        "srl_arg1_only",
        "srl_arg1_only_no_hand",
        "ground_full_sentence",
        "random_word",
        "first_word",
    ]

    if tokens_positive_method is not None and "srl" in tokens_positive_method:
        use_srl = True
    else:
        use_srl = False

    if len(frames_to_use) >= 3:
        coco_dir = "coco_annotations_all_frames"
        assert frames_to_use == ["pre", "pnr", "post"], (
            "Order has to be `pre`->`pnr`-->`post`."
        )
    elif len(frames_to_use) == 1 and frames_to_use[0] == "pnr":
        coco_dir = "coco_annotations"
    else:
        raise NotImplementedError(
            "Not done yet with only {} frames to use".format(coco_dir))

    if with_narration:
        output_annotation_suffix = "narrated_" + tokens_positive_method
        if tokens_positive_method is None:
            tokens_positive_method = "gt_srl_arg1"
        print("Setting `tokens_positive_method` to: {}".format(tokens_positive_method))
    else:
        # No need to use any SRL if no narration is provided.
        use_srl = False
        output_annotation_suffix = None

    additional_functions = {
        "srl": None
    }

    if use_srl is True:
        _ = get_srl_func(additional_functions)

    # """
    scod_train_file_to_coco = scod_train_file
    scod_val_file_to_coco = scod_val_file
    scod_test_file_to_coco = scod_test_file
    # Use only test.
    if only_on_test_file:
        scod_train_file_to_coco = None
        scod_val_file_to_coco = None

    convert_scod_to_coco_format(
        scod_train_file=scod_train_file_to_coco,
        scod_val_file=scod_val_file_to_coco,
        scod_test_file=scod_test_file_to_coco,
        frames=frames_to_use,
        root_image_folder=(
            "/local1/telinwu/research/resources/Ego4D"
            "/ego4d_scod_challenge/pre_pnr_post_frames"
        ),
        image_extension=".jpg",
        output_folder=(
            "/local1/telinwu/research/resources/Ego4D"
            "/ego4d_scod_challenge/{}".format(coco_dir)
        ),
        narration_pass="narration_pass_1",
        all_narrations=all_narrations,
        curr_videos_root=curr_videos_root,
        with_narration=with_narration,
        output_annotation_suffix=output_annotation_suffix,
        additional_functions=additional_functions,
        tokens_positive_method=tokens_positive_method,
        force_data_completeness=True,
    )

    print("All done!")
    exit(-1)
    # """
    ###########################################################################


    ###########################################################################
    # NOTE: Create CLIP csv-based dataset.
    use_srl = True

    enlarge_bbox_cropping_width = None
    enlarge_bbox_cropping_height = None
    output_annotation_suffix = None
    scod_file = scod_train_file

    # enlarge_bbox_cropping_width = 0.2
    # enlarge_bbox_cropping_height = 0.2
    # output_annotation_suffix = "enlarged_bbox0p2"
    # scod_file = scod_val_file

    additional_functions = {
        "srl": None
    }

    if use_srl is True:
        output_annotation_suffix = get_srl_func(additional_functions)

    convert_scod_bbox_to_clip_caption_data(
        scod_file=scod_file,
        frames=["pre", "pnr", "post"],
        root_image_folder=(
            "/local1/telinwu/research/resources/Ego4D"
            "/ego4d_scod_challenge/pre_pnr_post_frames"
        ),
        image_extension=".jpg",
        output_image_folder=(
            "/local1/telinwu/research/resources/Ego4D"
            "/ego4d_scod_challenge/pre_pnr_post_enlarged_bbox_images"
        ),
        output_annotation_folder=(
            "/local1/telinwu/research/resources/Ego4D"
            "/ego4d_scod_challenge/clip_annotations_all_frames"
        ),
        output_annotation_suffix=output_annotation_suffix,
        narration_pass="narration_pass_1",
        all_narrations=all_narrations,
        curr_videos_root=curr_videos_root,
        enlarge_bbox_cropping_width=enlarge_bbox_cropping_width,
        enlarge_bbox_cropping_height=enlarge_bbox_cropping_height,
        additional_functions=additional_functions,
    )

    print("All Done!")
    exit(-1)
    ###########################################################################
