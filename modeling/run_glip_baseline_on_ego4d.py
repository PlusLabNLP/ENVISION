import csv, os, sys, re, string, json, glob, shutil, random, datetime, math
import pprint
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

import cv2
import openai
OPENAI_KEY = None
import pandas as pd
import numpy as np

import torch
from transformers import AutoTokenizer
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# GPTs.
# from ChatGPT_webUI import prompt_chatgpt_and_parse_condition_response, chatbot

# Evals.
from coco_eval import get_avg_precision_at_iou
from coco_eval import get_single_image_results
from coco_eval import calc_precision_recall

# Ego4D.


def get_ego4d_scod_clips_and_frames(
    sampled_scod_clip,
    curr_videos_root,
    show_name="show",
    get_frames=True,
):
    video_uid = sampled_scod_clip["video_uid"]
    ego4d_video_path = os.path.join(
        curr_videos_root,
        "{}.mp4".format(video_uid),
    )
    print("Mother video: {}".format(ego4d_video_path))

    cap = cv2.VideoCapture(ego4d_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = float(frame_cnt) / float(fps)
    cap.release()

    print("The fps of video `{}` is: {} of frame counts: {} and duration: "
          "{} sec.".format(ego4d_video_path, fps, frame_cnt, duration))

    pre_frame = sampled_scod_clip["pre_frame"]["frame_number"]
    pnr_frame = sampled_scod_clip["pnr_frame"]["frame_number"]
    pos_frame = sampled_scod_clip["post_frame"]["frame_number"]

    pre_secs = pre_frame / fps
    pnr_secs = pnr_frame / fps
    pos_secs = pos_frame / fps

    pre_timestamp = str(datetime.timedelta(seconds=pre_secs))
    pnr_timestamp = str(datetime.timedelta(seconds=pnr_secs))
    pos_timestamp = str(datetime.timedelta(seconds=pos_secs))

    print(pre_frame, pnr_frame, pos_frame)
    print(pre_timestamp, pnr_timestamp, pos_timestamp)

    pre_timestamp_secs = math.floor(float(pre_timestamp.split(":")[-1]))
    pos_timestamp_secs = math.ceil(float(pos_timestamp.split(":")[-1]))

    pre_timestamp = ":".join(pre_timestamp.split(":")[:-1]) \
                             + ":" + str(pre_timestamp_secs)
    pos_timestamp = ":".join(pos_timestamp.split(":")[:-1]) \
                             + ":" + str(pos_timestamp_secs)

    # show_name = "show"
    if os.path.exists("./media/{}.mp4".format(show_name)):
        os.remove("./media/{}.mp4".format(show_name))

    ffmpeg_video_cmd = (
        "/home/telinwu/ffmpeg-git-20220910-amd64-static/ffmpeg -ss {} -to {} "
        "-i {} -c copy -loglevel panic ./media/{}.mp4".format(
            pre_timestamp,
            pos_timestamp,
            ego4d_video_path,
            show_name,
        )
    )
    os.system(ffmpeg_video_cmd)
    print("Executed command: {}".format(ffmpeg_video_cmd))
    
    if get_frames:
        for frame_name in ["pre_frame", "pnr_frame", "post_frame"]:
            if os.path.exists("./media/{}.png".format(frame_name)):
                os.remove("./media/{}.png".format(frame_name))
            frame_cnt = sampled_scod_clip[frame_name]["frame_number"]
            ffmpeg_frame_cmd = (
                "/home/telinwu/ffmpeg-git-20220910-amd64-static/ffmpeg "
                "-i {} "
                "-vf select='between(n\,{}\,{})' -vsync 0 "
                "-loglevel panic ./media/{}.png".format(
                    ego4d_video_path,
                    frame_cnt,
                    frame_cnt,
                    frame_name,
                )
            )
            os.system(ffmpeg_frame_cmd)
            print("Executed command: {}".format(ffmpeg_frame_cmd))

    # pprint.pprint(sampled_scod_clip)
    return video_uid, {"pre_frame": pre_frame,
                       "pnr_frame": pnr_frame,
                       "post_frame": pos_frame}
    
    
def get_scod_clipped_narrations_old(
    scod_clip,
    curr_videos_root,
    all_narrations,
    narration_pass="narration_pass_1",
):    
    video_uid = scod_clip["video_uid"]
    ego4d_video_path = os.path.join(
        curr_videos_root,
        "{}.mp4".format(video_uid),
    )

    cap = cv2.VideoCapture(ego4d_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = float(frame_cnt) / float(fps)
    cap.release()
    
    clip_uid = scod_clip["clip_uid"]
    pre_frame = scod_clip["pre_frame"]
    pnr_frame = scod_clip["pnr_frame"]
    pos_frame = scod_clip["post_frame"]
    if narration_pass in all_narrations[video_uid]:
        curr_narrations = all_narrations[video_uid][narration_pass]["narrations"]
    else:
        print("Video: {} does not have narration of pass: {}, "
              "skipping!".format(video_uid, narration_pass))
        return None, None
    
    # obj_of_change = pnr_frame["bbox"][1]["structured_noun"]
    
    pre_frame_num = pre_frame["frame_number"]
    pnr_frame_num = pnr_frame["frame_number"]
    pos_frame_num = pos_frame["frame_number"]
    
    pre_frame_sec = round(pre_frame_num/fps, 2)
    pnr_frame_sec = round(pnr_frame_num/fps, 2)
    pos_frame_sec = round(pos_frame_num/fps, 2)
    
    clip_narrations = []
    closest_narrations = []
    for i in range(len(curr_narrations)):
        timestamp_sec = curr_narrations[i]["timestamp_sec"]
        timestamp_frame = curr_narrations[i]["timestamp_frame"]
        narration_text = curr_narrations[i]["narration_text"]
        if (
            timestamp_frame >= pre_frame_num
            and timestamp_frame <= pos_frame_num
        ):
            clip_narrations.append((
                timestamp_sec, timestamp_frame, narration_text
            ))
        
        closest_narrations.append((
            abs(timestamp_frame-pre_frame_num),
            timestamp_sec, timestamp_frame, narration_text,
        ))
        
    closest_narrations = sorted(closest_narrations)[:5]
    closest_narrations = [(a, b, c) for _, a, b, c in closest_narrations]
    
    # print("Object of change: {}".format(obj_of_change))
    print("FPS: {}".format(fps))
    print("Pre/PNR/Post frame = {} / {} / {}".format(
        pre_frame_num, pnr_frame_num, pos_frame_num))
    print("Pre/PNR/Post sec   = {} / {} / {}".format(
        str(datetime.timedelta(seconds=pre_frame_sec))[:10],
        str(datetime.timedelta(seconds=pnr_frame_sec))[:10],
        str(datetime.timedelta(seconds=pos_frame_sec))[:10])
    )
    
    for a, b, c in closest_narrations:
        print("[{}]({}) {}".format(str(datetime.timedelta(seconds=a))[:10], b, c))
    
    return clip_narrations, closest_narrations


def get_scod_clipped_narrations(
    scod_clip,
    all_narrations,
    curr_videos_root,
    narration_pass="narration_pass_1",
    anchor_frame="pre",
    top_k=5,
    verbose=False,
):    
    video_uid = scod_clip["video_uid"]
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
    
    obj_of_changes = []
    for frame in ["pre", "pnr", "post"]:
        obj_info = scod_clip["{}_frame".format(frame)]["bbox"]
        for o in obj_info:
            if o["object_type"] == "object_of_change":
                obj_of_changes.append(o["structured_noun"])
    obj_of_changes = list(set(obj_of_changes))
    
    pre_frame_num = pre_frame["frame_number"]
    pnr_frame_num = pnr_frame["frame_number"]
    pos_frame_num = pos_frame["frame_number"]
    
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
            and timestamp_frame <= pos_frame_sec
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
    
    if verbose:
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


def create_glip_config_file(
    narration,
    video_uid,
    frames_dict,
    glip_images_folder="../../GLIP/DATASET/ego4d/frames/test",
    glip_config_file_template="../../GLIP/configs/ego4d/ego4d_template.yaml",
    glip_config_file_output="../../GLIP/configs/ego4d/ego4d_test.yaml",
    move_images=True,
    llm_condition=None,
    processed_scod_image_folder=None,
):
    narration_tokens = word_tokenize(narration)
    is_noun = lambda pos: pos[:2] == 'NN'
    narration_nouns = sorted(list(set([word for (word, pos) in
        nltk.pos_tag(narration_tokens) if is_noun(pos)])))
    print(narration)
    if "C " in narration and "C" in narration_nouns:
        narration_nouns.pop(narration_nouns.index("C"))
    if "Someone" in narration_nouns:
        narration_nouns.pop(narration_nouns.index("Someone"))
    print(narration_nouns)

    override_category = []
    override_category_mapping = {}
    caption_prompt = []
    override_category_id = 1
    for noun in narration_nouns:
        override_category.append({
            "id": override_category_id,
            "name": noun,
            "supercategory": "objects",
        })
        override_category_mapping[override_category_id] = noun
        override_category_id += 1

        prefix = []
        name = []
        suffix = []
        now_suffix = False
        for token in narration_tokens:
            if token == noun:
                name.append(noun)
                now_suffix = True
            elif now_suffix:
                suffix.append(token)
            else:
                prefix.append(token)
        prefix = " ".join(prefix)
        name = " ".join(name)
        if llm_condition is not None:
            suffix.append(llm_condition)
        suffix = " ".join(suffix)
        caption_prompt.append({
            "prefix": prefix,
            "name": name,
            "suffix": suffix,
        })

    num_classes = len(narration_nouns)

    to_revise = {
        "CAPTION_PROMPT": str(caption_prompt),
        "OVERRIDE_CATEGORY": str(override_category),
        "NUM_CLASSES": str(num_classes+1),
        "img_dir": "ego4d/frames/test",
    }
    if processed_scod_image_folder is not None:
        to_revise["img_dir"] = processed_scod_image_folder

    # pprint.pprint(to_revise)
    # print(video_uid, pre_frame, pnr_frame, post_frame)

    # Copy frames to GLIP folder.
    if move_images:
        for frame in ["pre_frame", "pnr_frame", "post_frame"]:
            src_img = "./media/{}.png".format(frame)
            dst_img = os.path.join(
                glip_images_folder,
                "{}_{}.png".format(video_uid, frames_dict[frame])
            )
            print("Moving image from {} to {}".format(src_img, dst_img))
            shutil.copy(src_img, dst_img)

    glip_config_file_output = open(glip_config_file_output, "w")

    for line in open(glip_config_file_template):
        line = line.rstrip()
        for key in to_revise:
            if key in line and key != "NUM_CLASSES":
                line = line.replace("TODO", to_revise[key].replace("\'", "\""))
            elif key in line:
                line = "    NUM_CLASSES: {}".format(to_revise[key])
        glip_config_file_output.write(line+"\n")
    glip_config_file_output.close()

    return override_category, override_category_mapping, caption_prompt, narration_tokens


def create_glip_dataset_file(
    override_category,
    video_uid,
    frames_dict,
    coco_annots_test_json="../../GLIP/DATASET/ego4d/annotations/image_info_test-dev2017.json",
    output_dataset_file="../../GLIP/DATASET/ego4d/annotations/ego4d_test.json",
    glip_images_folder="../../GLIP/DATASET/ego4d/frames/test",
    processed_scod_image_folder=None,
):
    coco_annots_test = json.load(open(coco_annots_test_json))

    new_d = {
        "info": coco_annots_test["info"],
        "images": [],
        "licenses": coco_annots_test["licenses"],
        "categories": override_category,
    }

    for frame in ["pre_frame", "pnr_frame", "post_frame"]:

        if processed_scod_image_folder is not None:
            file_name = "{}/{}.png".format(video_uid, frames_dict[frame])
            file_path = os.path.join(processed_scod_image_folder, file_name)
        else:
            file_name = "{}_{}.png".format(video_uid, frames_dict[frame])
            file_path = os.path.join(glip_images_folder,
                "{}_{}.png".format(video_uid, frames_dict[frame]))
        assert os.path.exists(file_path), "Image {} does not exist!".format(file_path)

        img = Image.open(file_path)
        width = img.width
        height = img.height

        image_d = {
            "coco_url": "",
            "date_captured": "",
            "file_name": file_name,
            "height": height,
            "width": width,
            "license": 6,
            "id": "{}_{}".format(video_uid, frames_dict[frame]),
        }
        new_d["images"].append(image_d)

    json.dump(
        new_d,
        open(output_dataset_file, "w"),
        indent=4,
    )
    
    return new_d


def run_glip_on_scods(
    downloaded_scod_clips,
    curr_videos_root,
    all_narrations,
    output_results_json_file,
    max_runs=None,
    processed_scod_image_folder=None,
    narration_pass="narration_pass_1",
    llm_prompting=None,
    glip_images_folder="../../GLIP/DATASET/ego4d/frames/test",
    glip_config_file_template="../../GLIP/configs/ego4d/ego4d_template.yaml",
    glip_config_file_output="../../GLIP/configs/ego4d/ego4d_test.yaml",
    coco_annots_test_json="../../GLIP/DATASET/ego4d/annotations/image_info_test-dev2017.json",
    output_dataset_file="../../GLIP/DATASET/ego4d/annotations/ego4d_test.json",
    run_glip_commands=[
        "sh scripts/run_glip.sh",
    ],
    glip_inference_json_file = (
        "../../GLIP/test_ego4d_eval_outputs/eval/"
        "glip_tiny_model_o365_goldg_cc_sbu/"
        "inference/test/bbox.json"
    ),
):

    # LLMs.
    if llm_prompting == "gpt-3":
        raise NotImplementedError("Not done with this: `{}` llm_prompting"
                                  " yet. ".format(llm_prompting))

    all_results = []
    processed_image_names = []
    if os.path.exists(output_results_json_file):
        all_results = json.load(open(output_results_json_file))
        for res in all_results:
            pre_frame = res["pre_frame"]
            image_name = pre_frame["image_name"]
            if "/" in image_name:
                image_name = "{}_{}".format(*image_name.split("/"))
            processed_image_names.append(image_name)

    if max_runs is None:
        max_runs = len(downloaded_scod_clips) + 1e8
    print("Maximum runs = {}".format(max_runs))

    curr_run = 0
    for scod_clip in tqdm(downloaded_scod_clips, desc="SCOD"):

        if curr_run >= max_runs:
            print("Reaching max runs of {}".format(max_runs))
            print("Aborting!")
            break
        curr_run += 1

        pre_frame_name = "{}_{}.png".format(
            scod_clip["video_uid"], scod_clip["pre_frame"]["frame_number"])
        if pre_frame_name in processed_image_names:
            print("Skipping frame: {} since already done".format(pre_frame_name))
            continue

        # clip_narrations, closest_narrations = get_scod_clipped_narrations(
        #     scod_clip,
        #     curr_videos_root,
        #     all_narrations,
        #     narration_pass=narration_pass,
        # )
        clip_narrations, closest_narrations = get_scod_clipped_narrations(
            scod_clip,
            all_narrations,
            curr_videos_root,
            narration_pass=narration_pass,
            top_k=5,
            anchor_frame="pnr",
            verbose=False,
        )

        if clip_narrations is None and closest_narrations is None:
            continue
        if len(closest_narrations) <= 0:
            continue

        if processed_scod_image_folder is None:
            get_frames = True
            move_images = True
            if pre_frame_name in os.listdir(glip_images_folder):
                get_frames = False
                move_images = False
            if not get_frames:
                print("Existing frame: {}".format(pre_frame_name))
            video_uid, frames_dict = get_ego4d_scod_clips_and_frames(
                scod_clip,
                curr_videos_root,
                show_name="scod",
                get_frames=get_frames,
            )
        else:
            get_frames = False
            move_images = False
            pre_frame = scod_clip["pre_frame"]["frame_number"]
            pnr_frame = scod_clip["pnr_frame"]["frame_number"]
            pos_frame = scod_clip["post_frame"]["frame_number"]
            video_uid = scod_clip["video_uid"]
            frames_dict = {"pre_frame": pre_frame,
                           "pnr_frame": pnr_frame,
                           "post_frame": pos_frame}

        if narration_pass in ["narration_pass_1", "narration_pass_2"]:
            narration = closest_narrations[0][-1].split("#C ")[-1].strip()

            # narration = closest_narrations[0][-1].split("#C ")[-1].strip()
            narration = closest_narrations[0][-1]
            trimmed_narration = narration.split("#C ")[-1]
            trimmed_narration = trimmed_narration.strip()
            trimmed_narration = trimmed_narration.replace("C ", "Someone ")
            narration = trimmed_narration

        sampled_condition = None
        if llm_prompting == "gpt-3":
            raise NotImplementedError("Not done with this: `{}` llm_prompting"
                                      " yet. ".format(llm_prompting))
        elif llm_prompting == "chatgpt_web":
            condition = "postcondition"
            prompt = "List three {}s of \"{}\"?".format(condition, narration)
            # FIXME: hack
            if "#unsure" in prompt:
                prompt = prompt.replace("#unsure", "object")
            parsed_conditions = []

            while len(parsed_conditions) == 0:
                parsed_conditions = prompt_chatgpt_and_parse_condition_response(
                    chatbot,
                    prompt,
                    condition=condition,
                    method="chatgpt_web",
                )

            print(parsed_conditions)
            sampled_condition = np.random.choice(parsed_conditions)
            print(sampled_condition)
            raise
        elif llm_prompting == "chatgpt_api":
            openai.api_key = OPENAI_KEY
            condition = "postcondition"
            prompt = "List three {}s of \"{}\"?".format(condition, narration)
            # FIXME: hack
            if "#unsure" in prompt:
                prompt = prompt.replace("#unsure", "object")
            parsed_conditions = []

            while len(parsed_conditions) == 0:
                parsed_conditions = prompt_chatgpt_and_parse_condition_response(
                    openai.ChatCompletion,
                    prompt,
                    condition=condition,
                    method="chatgpt_api",
                    openai_key=OPENAI_KEY,
                )

            # print(parsed_conditions)
            sampled_condition = np.random.choice(parsed_conditions)
            # print(sampled_condition)
            # raise

        (override_category,
         override_category_mapping,
         caption_prompt,
         narration_tokens) = create_glip_config_file(
            narration,
            video_uid,
            frames_dict,
            glip_images_folder=glip_images_folder,
            glip_config_file_template=glip_config_file_template,
            glip_config_file_output=glip_config_file_output,
            move_images=move_images,
            llm_condition=sampled_condition,
            processed_scod_image_folder=processed_scod_image_folder,
        )

        data_dict = create_glip_dataset_file(
            override_category,
            video_uid,
            frames_dict,
            coco_annots_test_json=coco_annots_test_json,
            output_dataset_file=output_dataset_file,
            glip_images_folder=glip_images_folder,
            processed_scod_image_folder=processed_scod_image_folder,
        )

        for cmd in run_glip_commands:
            os.system(cmd)
        os.remove("temp.txt")

        label_mapping = {
            x["id"]: x["name"] for x in data_dict["categories"]
        }

        # TODO: process the json output files to here.
        pred_bboxes_on_images = {}
        pred_bboxes = json.load(open(glip_inference_json_file))
        for bbox in pred_bboxes:
            image_id = bbox["image_id"]
            if processed_scod_image_folder is None:
                img_path = os.path.join(
                    glip_images_folder,
                    "{}.png".format(image_id)
                )
            else:
                video_uid, frame_num = image_id.split("_")
                img_path = os.path.join(
                    processed_scod_image_folder,
                    "{}/{}.png".format(video_uid, frame_num)
                )
            assert os.path.exists(img_path), (
                "Image {} does not exist!?".format(img_path)
            )
            if image_id not in pred_bboxes_on_images:
                pred_bboxes_on_images[image_id] = {
                    "img_path": img_path,
                    "bboxes": []
                }
            # FIXME: hack.
            if bbox["category_id"] in override_category_mapping:
                curr_pred_label = override_category_mapping[bbox["category_id"]]
            else:
                curr_pred_label = "object_category:{}".format(bbox["category_id"])

            pred_bboxes_on_images[image_id]["bboxes"].append({
                "score": bbox["score"],
                "bbox": bbox["bbox"],
                "label": curr_pred_label,
            })

        for image_id in pred_bboxes_on_images:
            pred_bboxes_on_images[image_id]["bboxes"] = sorted(
                pred_bboxes_on_images[image_id]["bboxes"],
                key=lambda x: x["score"],
                reverse=True
            )

        images_data = data_dict["images"]

        curr_results = {}
        for frame in ["pre_frame", "pnr_frame", "post_frame"]:
            image_id = "{}_{}".format(video_uid, frames_dict[frame])
            if processed_scod_image_folder is None:
                image_name = "{}_{}.png".format(video_uid, frames_dict[frame])
                image_folder = glip_images_folder
            else:
                image_name = "{}/{}.png".format(video_uid, frames_dict[frame])
                image_folder = processed_scod_image_folder
            all_gt_bboxes = scod_clip[frame]["bbox"]
            gt_bboxes = []
            for gt_bbox in all_gt_bboxes:
                if gt_bbox["object_type"] == "object_of_change":
                    gt_bboxes.append(gt_bbox)

            # No predicted bboxes.
            if image_id not in pred_bboxes_on_images:
                pred_bboxes_on_images[image_id] = {"bboxes": []}
            if len(pred_bboxes_on_images[image_id]["bboxes"]) == 0:
                pred_bboxes_on_images[image_id]["bboxes"] = [{
                    "bbox": [0, 0, 1, 1],
                    "score": 1.0,
                }]

            curr_results[frame] = {
                "video_uid": video_uid,
                "frame_cnt": frames_dict[frame],
                "image_name": image_name,
                "image_folder": image_folder,
                "narration_tokens": narration_tokens,
                "caption_prompt": caption_prompt,
                "label_mapping": label_mapping,
                "gt_bboxes": gt_bboxes,
                "pred_bboxes": pred_bboxes_on_images[image_id]["bboxes"],
            }

        all_results.append(curr_results)

        # NOTE: periodically save incase breakdown.
        json.dump(
            all_results,
            open(output_results_json_file, "w"),
            indent=4,
        )
        print("Saving results file to: {}".format(output_results_json_file))
        # raise

    # NOTE: periodically save incase breakdown.
    json.dump(
        all_results,
        open(output_results_json_file, "w"),
        indent=4,
    )
    print("Saving results file to: {}".format(output_results_json_file))

    return all_results


def coco_eval_results(
    output_results_json_file,
    top_k=5,
    verbose=True,
    first_k=None,
    best_k=None,
    frame_keys=["pre", "pnr", "post"],
):

    if best_k is not None:
        assert best_k >= top_k, ("Best-K is required to be >= Top-K for easier"
            "implementation, currently best_k = {}, and top_k = {}".format(best_k, top_k))

    if type(output_results_json_file) is str:
        res = json.load(open(output_results_json_file))
        print("Total results: {} of {}".format(len(res), output_results_json_file))
    else:
        assert type(output_results_json_file) is dict
        key = list(output_results_json_file.keys())[0]
        res = output_results_json_file[key]
        assert type(res) is list
        print("Total results: {} of {}".format(len(res), key))

    overall_results = []
    keyed_results = {}

    if first_k is not None:
        res = res[:first_k]

    for d in tqdm(res, desc="Results..."):
        frame_res = {}
        curr_video_uid = None
        curr_keyed_results = {}
        frames_dict = {}
        for frame in ["pre_frame", "pnr_frame", "post_frame"]:
            if frame not in d:
                continue
            fr = d[frame]
            gt_bboxes = fr["gt_bboxes"]
            curr_video_uid = fr["video_uid"]
            frames_dict[frame] = fr["frame_cnt"]
            gt_bboxes_dict = {}
            for gt_bbox in gt_bboxes:
                key = gt_bbox["structured_noun"]
                bbox = gt_bbox["bbox"]
                bbox = [bbox["x"], bbox["y"],
                        bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]]
                if key not in gt_bboxes_dict:
                    gt_bboxes_dict[key] = []
                gt_bboxes_dict[key].append(bbox)

                # FIXME: Hack for now, ego4d baseline only predicts the class
                # `object_of_change`.
                key = "object_of_change"
                if key not in gt_bboxes_dict:
                    gt_bboxes_dict[key] = []
                gt_bboxes_dict[key].append(bbox)

            pred_bboxes = fr["pred_bboxes"]
            pred_bboxes_dict = {}
            pred_labels = []
            for pred_bbox in pred_bboxes:
                if "label" not in pred_bbox:
                    pred_bbox["label"] = "NULL-OBJECT"
                label = pred_bbox["label"]
                pred_labels.append(label)
                bbox = pred_bbox["bbox"]
                score = pred_bbox["bbox"]
                curr_label = None
                for key in gt_bboxes_dict:
                    if key is None: continue
                    if label in key or key in label: 
                        curr_label = key
                        break
                # FIXME: Hack for now, ego4d baseline only predicts the class
                # `object_of_change`.
                if curr_label is None and label == "object_of_change":
                    curr_label = "object_of_change"
                if curr_label not in pred_bboxes_dict:
                    pred_bboxes_dict[curr_label] = []
                bbox = [bbox[0], bbox[1],
                        bbox[0] + bbox[2], bbox[1] + bbox[3]]
                pred_bboxes_dict[curr_label].append(bbox)
            for key in pred_bboxes_dict:
                if best_k is not None:
                    pred_bboxes_dict[key] = pred_bboxes_dict[key][:best_k]
                else:
                    pred_bboxes_dict[key] = pred_bboxes_dict[key][:top_k]

            pred_labels = sorted(list(set(pred_labels)))
            # pprint.pprint(gt_bboxes_dict)
            # pprint.pprint(pred_bboxes_dict)
            # print(pred_labels)
            
            curr_frame_res = {}
            # pprint.pprint(gt_bboxes_dict)
            # pprint.pprint(pred_bboxes_dict)
            for key in pred_bboxes_dict:
                # FIXME: have not deal with excessively predicted classes
                if key not in gt_bboxes_dict:
                    continue
                curr_frame_res[key] = {}

                # IOU.
                def _get_aps(gt_boxes, pred_boxes, best_k=None):
                    overall_prec, overall_rec = [], []
                    for ap in np.linspace(50, 95, num=10):
                        iou_thr = ap / 100.0
                        map_res = get_single_image_results(gt_boxes, pred_boxes, iou_thr=iou_thr)
                        map_res = {key: map_res}
                        map_res = calc_precision_recall(map_res)
                        if best_k is None:
                            ap_key = "AP{}".format(int(ap))
                            curr_frame_res[key][ap_key] = map_res
                        else:
                            ap_key = "Best-{}-AP{}".format(best_k, int(ap))
                            if ap_key in curr_frame_res[key]:
                                curr_frame_res[key][ap_key] = \
                                    (max(curr_frame_res[key][ap_key][0], map_res[0]),
                                     max(curr_frame_res[key][ap_key][1], map_res[1]))
                            else:
                                curr_frame_res[key][ap_key] = map_res
                        overall_prec.append(map_res[0])
                        overall_rec.append(map_res[1])
                    overall_prec = np.mean(np.asarray(overall_prec))
                    overall_rec = np.mean(np.asarray(overall_rec))
                    return overall_prec, overall_rec

                gt_boxes = gt_bboxes_dict[key]
                # Best-K.
                if best_k is not None:
                    overall_prec, overall_rec = 0, 0
                    for _pred_boxes in pred_bboxes_dict[key]:
                        _curr_pred_boxes = [_pred_boxes]
                        _overall_prec, _overall_rec = _get_aps(
                            gt_boxes, _curr_pred_boxes, best_k=best_k)
                        overall_prec = max(overall_prec, _overall_prec)
                        overall_rec = max(overall_rec, _overall_rec)
                    ap_key = "Best-{}-AP".format(best_k)
                    curr_frame_res[key][ap_key] = (overall_prec, overall_rec)
                # Standard top-K.
                pred_boxes = pred_bboxes_dict[key][:top_k]
                overall_prec, overall_rec = _get_aps(gt_boxes, pred_boxes)
                curr_frame_res[key]["AP"] = (overall_prec, overall_rec)

            frame_res[frame] = curr_frame_res
            curr_keyed_results[frame] = curr_frame_res
        
        # NOTE: average across all the objects.
        # pprint.pprint(frames_dict)
        # pprint.pprint(curr_keyed_results);raise
        # print(curr_video_uid)
        curr_key = [curr_video_uid]
        new_curr_keyed_results = {}
        final_curr_keyed_results = {}
        for frame_prefix in ["pnr", "pre", "post"]:
            if "{}_frame".format(frame_prefix) not in frames_dict:
                continue
            if frame_prefix in frame_keys:
                curr_key.append(str(frames_dict["{}_frame".format(frame_prefix)]))
            new_curr_keyed_results["{}_frame".format(frame_prefix)] = {}
            final_curr_keyed_results["{}_frame".format(frame_prefix)] = {}

            # AP50 to AP95.
            ap_key_prefixes = ["AP"]
            retained_ap_keys = ["", "50", "75"]
            if best_k is not None:
                ap_key_prefixes.append("Best-{}-AP".format(best_k))
            for ap_key_prefix in ap_key_prefixes:
                overall_pres, overall_recs = [], []
                for iou_thr in np.linspace(50, 95, num=10):
                    ap = int(iou_thr)

                    ap_key = "{}{}".format(ap_key_prefix, ap)
                    new_curr_keyed_results["{}_frame".format(frame_prefix)][ap_key] = {
                        "precision": None,
                        "recall": None,
                    }
                    curr_precs, curr_recas = [], []
                    for obj, perf in curr_keyed_results["{}_frame".format(frame_prefix)].items():
                        perf = perf[ap_key]
                        curr_precs.append(perf[0])
                        curr_recas.append(perf[1])
                        overall_pres.append(perf[0])
                        overall_recs.append(perf[1])
                    new_curr_keyed_results["{}_frame".format(frame_prefix)][ap_key]["precision"] = \
                        np.mean(np.asarray(curr_precs))
                    new_curr_keyed_results["{}_frame".format(frame_prefix)][ap_key]["recall"] = \
                        np.mean(np.asarray(curr_recas))

                # Overall AP.
                avg_pres = np.mean(np.asarray(overall_pres))
                avg_recs = np.mean(np.asarray(overall_recs))
                new_curr_keyed_results["{}_frame".format(frame_prefix)][ap_key_prefix] = {
                    "precision": None,
                    "recall": None,
                }
                new_curr_keyed_results["{}_frame".format(frame_prefix)][ap_key_prefix]["precision"] = \
                    avg_pres
                new_curr_keyed_results["{}_frame".format(frame_prefix)][ap_key_prefix]["recall"] = \
                    avg_recs
                all_ap_keys = list(new_curr_keyed_results["{}_frame".format(frame_prefix)].keys())
                for ap_key in sorted(all_ap_keys):
                    if ap_key.split(ap_key_prefix)[-1] in retained_ap_keys:
                        final_curr_keyed_results["{}_frame".format(frame_prefix)][ap_key] = \
                            new_curr_keyed_results["{}_frame".format(frame_prefix)][ap_key]
                    pass
                pass
            pass
            # pprint.pprint(new_curr_keyed_results)
            # pprint.pprint(final_curr_keyed_results)
            # raise
        if len(curr_key) > 1:
            curr_key = "_".join(curr_key)
            keyed_results[curr_key] = final_curr_keyed_results

        ####
        overall_results.append(frame_res)

    # pprint.pprint(overall_results)
    ap_keys = ["AP50", "AP75", "AP"]
    if best_k is not None:
        for ap in ap_keys.copy():
            ap_keys.append("Best-{}-{}".format(best_k, ap))
    avg_res = {
        "pre_frame": {ap: [] for ap in ap_keys},
        "pnr_frame": {ap: [] for ap in ap_keys},
        "post_frame": {ap: [] for ap in ap_keys},
    }
    for res in overall_results:
        for frame in res:
            for obj in res[frame]:
                for ap in ap_keys:
                    avg_res[frame][ap].append(res[frame][obj][ap])
    for frame in avg_res:
        for ap in ap_keys:
            avg_res[frame][ap] = {
                "precision": np.mean(np.asarray([u for u,v in avg_res[frame][ap]])),
                "recall": np.mean(np.asarray([v for u,v in avg_res[frame][ap]])),
            }

    if verbose:
        pprint.pprint(avg_res)
    # raise

    return overall_results, avg_res, keyed_results


""" Usage
python3 run_glip_baseline_on_ego4d.py \
    ./glip_result_json_files/using_narrations.json \
    narration_pass_1
"""
if __name__ == "__main__":
    # TODO: Change the following args.
    output_results_json_file = sys.argv[1]
    narration_pass = sys.argv[2]
    max_runs = None
    llm_prompting = None
    if len(sys.argv) > 3:
        max_runs = int(sys.argv[3])
    if len(sys.argv) > 4:
        llm_prompting = sys.argv[4]
    top_k = 5
    processed_scod_image_folder = None
    if llm_prompting == "chatgpt_api":
        OPENAI_KEY = input("Your OpenAI key: ")
        openai.api_key = OPENAI_KEY

    # Ego4D args.
    # curr_videos_root = "/local1/jrbronkar/ego4d_videos/v1/full_scale"
    curr_videos_root = "/local1/hu528/ego4d_data/v1/full_scale"
    annots_root = "/local1/hu528/ego4d_data/v1/annotations/"
    scod_train_file = "fho_scod_train.json"
    scod_val_file = "fho_scod_val.json"
    processed_scod_image_folder = (
        "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge"
        "/pre_pnr_post_png_frames"
    )

    # GLIP args.
    glip_images_folder = "../../GLIP/DATASET/ego4d/frames/test"
    glip_config_file_template = "../../GLIP/configs/ego4d/ego4d_template.yaml"
    glip_config_file_output = "../../GLIP/configs/ego4d/ego4d_test.yaml"
    coco_annots_test_json = "../../GLIP/DATASET/ego4d/annotations/image_info_test-dev2017.json"
    output_dataset_file = "../../GLIP/DATASET/ego4d/annotations/ego4d_test.json"
    run_glip_commands=[
        "sh scripts/run_glip.sh > temp.txt",
    ]
    # Needs to be consistent with the configs.
    glip_inference_json_file = (
        "../../GLIP/test_ego4d_eval_outputs/eval/"
        "glip_tiny_model_o365_goldg_cc_sbu/"
        "inference/test/bbox.json"
    )

    assert narration_pass in [
        "narration_pass_1",
        "narration_pass_2",
    ]
    assert llm_prompting in [
        "gpt-3",
        "chatgpt_api",
        "chatgpt_web",
        None,
    ]

    all_narrations_file = os.path.join(annots_root, "narration.json")
    all_narrations = json.load(open(all_narrations_file))

    scod_train_file = os.path.join(annots_root, scod_train_file)
    scod_val_file = os.path.join(annots_root, scod_val_file)
    scod_train_data = json.load(open(scod_train_file))
    scod_val_data = json.load(open(scod_val_file))
    # scod_clips = scod_train_data["clips"] + scod_val_data["clips"]
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
        if narration_pass not in all_narrations[video_uid]:
            video_wo_narrations[video_uid] = True

    print("And there are {} clips available for scod.".format(len(downloaded_scod_clips)))

    print("The following videos do not have narrations at {}".format(narration_pass))
    for video_uid in sorted(video_wo_narrations):
        print(video_uid)
    print("In total, {} videos do not have narration: {}".format(
        len(video_wo_narrations), narration_pass))

    # """
    run_glip_on_scods(
        downloaded_scod_clips,
        curr_videos_root,
        all_narrations,
        output_results_json_file,
        max_runs=max_runs,
        processed_scod_image_folder=processed_scod_image_folder,
        narration_pass=narration_pass,
        llm_prompting=llm_prompting,
        glip_images_folder=glip_images_folder,
        glip_config_file_template=glip_config_file_template,
        glip_config_file_output=glip_config_file_output,
        coco_annots_test_json=coco_annots_test_json,
        output_dataset_file=output_dataset_file,
        run_glip_commands=run_glip_commands,
        glip_inference_json_file=glip_inference_json_file,
    )
    # """

    coco_eval_results(
        output_results_json_file,
        top_k=top_k,
    )
