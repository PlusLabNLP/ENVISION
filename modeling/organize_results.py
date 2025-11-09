import os, csv, json, random, math, argparse
import pprint
from tqdm import tqdm


def organize(
    inference_file,
    out_file,
    org_scod_data,
    coco_test_file,
    image_folder,
    args=None,
):
    coco_test_data = json.load(open(coco_test_file))
    coco_category_data = coco_test_data["categories"]
    coco_category_data = {
        x["id"]: x["name"] for x in coco_category_data
    }
    coco_image_data = coco_test_data["images"]
    coco_image_data = {
        x["id"]: x for x in coco_image_data
    }
    coco_annots_data = coco_test_data["annotations"]

    original_scod_frames = {}
    gt_bboxes = {}
    gt_bboxes_tool = {}
    for org_scod in org_scod_data:
        for fr in args.frames_to_use:
            video_uid = org_scod["video_uid"]
            frame_num = org_scod["{}_frame".format(fr)]["frame_number"]
            key = "{}_{}".format(video_uid, frame_num)
            if key not in original_scod_frames:
                original_scod_frames[key] = []
            if "{}_frame".format(fr) not in original_scod_frames[key]:
                original_scod_frames[key].append("{}_frame".format(fr))
            bbox_data = org_scod["{}_frame".format(fr)]["bbox"]
            for bbox_datum in bbox_data:
                if bbox_datum["object_type"] == "object_of_change":
                    if key not in gt_bboxes:
                        gt_bboxes[key] = {}
                    if "{}_frame".format(fr) not in gt_bboxes[key]:
                        gt_bboxes[key]["{}_frame".format(fr)] = []
                    gt_bboxes[key]["{}_frame".format(fr)].append(bbox_datum)
                elif bbox_datum["object_type"] == "tool":
                    if key not in gt_bboxes_tool:
                        gt_bboxes_tool[key] = {}
                    if "{}_frame".format(fr) not in gt_bboxes_tool[key]:
                        gt_bboxes_tool[key]["{}_frame".format(fr)] = []
                    gt_bboxes_tool[key]["{}_frame".format(fr)].append(bbox_datum)
                pass
            pass
        pass

    inference_data = json.load(open(inference_file))
    inference_dict = {}
    inference_dict_tool = {}
    for inference_datum in tqdm(inference_data, desc="Processing"):
        image_id = inference_datum["image_id"]
        if inference_datum["category_id"] == 1:
            curr_inference_dict = inference_dict
        elif inference_datum["category_id"] == 2:
            curr_inference_dict = inference_dict_tool
        if image_id not in curr_inference_dict:
            curr_inference_dict[image_id] = []
        curr_inference_dict[image_id].append({
            "bbox": inference_datum["bbox"],
            "score": inference_datum["score"],
            "label": coco_category_data[inference_datum["category_id"]],
        })

    # print(len(inference_dict))
    # print(len(coco_image_data))
    # print(len(original_scod_frames))
    # print(len(gt_bboxes))
    # print(args.frames_to_use)
    # raise

    # TODO: cope with un-narrated ones (with `object_of_change`).

    all_results = []
    for image_id in coco_image_data:

        # FIXME: Check for no image_id data.
        if (
            image_id not in inference_dict
            and args.no_skip_empty_preds
        ):
            inference_dict[image_id] = [{
                "bbox": [0, 0, 1, 1],
                "label": "object_of_change",
                "score": 1.0,
            }]
        else:
            inference_dict[image_id] = sorted(
                inference_dict[image_id],
                key=lambda x: float(x["score"]),
                reverse=True,
            )
            if image_id in inference_dict_tool:
                inference_dict_tool[image_id] = sorted(
                    inference_dict_tool[image_id],
                    key=lambda x: float(x["score"]),
                    reverse=True,
                )
        curr_results = {}

        file_name = coco_image_data[image_id]["file_name"]
        file_name = file_name.split(".")[0]
        video_uid, frame_num = file_name.split("/")
        key = "{}_{}".format(video_uid, frame_num)
        frame_names = original_scod_frames[key]

        for frame_name in frame_names:
            # FIXME: If GT bbox not exist in the current data?
            # No `object_of_change`!
            if key not in gt_bboxes:
                continue
            curr_results[frame_name] = {
                "coco_image_id": image_id,
                "video_uid": video_uid,
                "frame_cnt": frame_num,
                "image_name": coco_image_data[image_id]["file_name"],
                "image_folder": image_folder,
                "narration_tokens": [],
                "caption_prompt": [],
                "label_mapping": coco_category_data,
                # "gt_label": struc_nouns,
                # "gt_bboxes": gt_bboxes,
                "gt_bboxes": gt_bboxes[key][frame_name],
                "pred_bboxes": inference_dict[image_id],
            }
            # pprint.pprint(curr_results)
            # raise
            if len(gt_bboxes_tool) > 0:
                curr_results[frame_name]["gt_bboxes_tool"] = gt_bboxes_tool[key][frame_name]
            if image_id in inference_dict_tool:
                curr_results[frame_name]["pred_bboxes_tool"] = inference_dict_tool[image_id]
            all_results.append(curr_results)
        pass

    json.dump(
        all_results,
        open(out_file, "w"), 
        indent=4,
    )
    print(len(all_results))
    print("Saving results file to: {}".format(out_file))

    return None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--image_folder",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--coco_test_file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--org_scod_files",
        default=None,
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--inference_file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--frames_to_use",
        default=["pre", "pnr", "post"],
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--no_skip_empty_preds",
        action="store_true",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    data_root = args.data_root
    image_folder = args.image_folder
    coco_test_file = args.coco_test_file
    org_scod_files = args.org_scod_files

    # TODO: Change this for different experiment.
    inference_file = (
        "baselines/DETR/"
        "ego4dv1_pnr_objects"
        "/inference/coco_instances_results.json"
    )
    inference_file = (
        "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge"
        "/baselines/VideoIntern/ego4dv1_pnr_objects/inference/test.json"
    )

    inference_file = (
        "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge"
        "/baselines/DETR/ego4dv2_pre_pnr_post_objects/inference/coco_instances_results.json"
    )

    inference_file = (
        "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge"
        "/baselines/DETR/ego4dv2_pre_pnr_post_objects/inference/coco_instances_results_pnr2prepost.json"
    )

    inference_file = (
        "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge"
        "/baselines/DETR/ego4dv2_pre_pnr_post_objects/inference/coco_instances_results.json"
    )

    inference_file = (
        "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge"
        "/baselines/VideoIntern/ego4dv2_pre_pnr_post_objects/inference/coco_instances_results_pnr2prepost.json"
    )

    inference_file = args.inference_file
    print("Processing inference file: {}".format(inference_file))

    if "coco_annotations_all_frames" in args.coco_test_file:
        args.frames_to_use = ["pre", "pnr", "post"]
    else:
        args.frames_to_use = ["pnr"]

    # Files.
    # inference_file = os.path.join(data_root, inference_file)
    assert os.path.exists(inference_file), "File {} not exist!".format(inference_file)
    coco_test_file = os.path.join(data_root, coco_test_file)
    assert os.path.exists(coco_test_file), "File {} not exist!".format(coco_test_file)
    image_folder = os.path.join(data_root, image_folder)

    org_scod_data = []
    for org_scod_file in org_scod_files:
        org_scod_data += json.load(open(org_scod_file))["clips"]

    out_file = inference_file.replace(".json", "_with_gt.json")

    organize(
        inference_file=inference_file,
        out_file=out_file,
        org_scod_data=org_scod_data,
        coco_test_file=coco_test_file,
        image_folder=image_folder,
        args=args,
    )
