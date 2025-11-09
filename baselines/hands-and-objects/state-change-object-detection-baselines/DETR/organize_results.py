import os, csv, json, random, math
import pprint
from tqdm import tqdm


def organize(
    inference_file,
    out_file,
    org_scod_data,
    coco_test_file,
    image_folder,
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
    coco_annots_data = {
        x["id"]: x for x in coco_annots_data
    }

    org_scod_dict = {}
    for org_scod in org_scod_data:
        video_uid = org_scod["video_uid"]
        if video_uid not in org_scod_dict:
            org_scod_dict[video_uid] = {}
        for fr in ["pre", "pnr", "post"]:
            frame_info = {
                "frame_name": None,
                "bbox": [],
            }
            frame = org_scod["{}_frame".format(fr)]
            frame_num = frame["frame_number"]
            bbox_info = frame["bbox"]
            for bbox in bbox_info:
                if bbox["object_type"] == "object_of_change":
                    # gt_bbox = [
                    #     bbox["bbox"]["x"], bbox["bbox"]["y"],
                    #     bbox["bbox"]["width"], bbox["bbox"]["height"],
                    # ]
                    gt_bbox = bbox
                    # frame_info["bbox"].append({
                    #     "structured_noun": bbox["structured_noun"],
                    #     "gt_bbox": gt_bbox,
                    # })
                    frame_info["bbox"].append(gt_bbox)
            frame_info["frame_name"] = "{}_frame".format(fr)
            org_scod_dict[video_uid][frame_num] = frame_info

    inference_data = json.load(open(inference_file))
    inference_dict = {}
    for inference_datum in tqdm(inference_data, desc="Processing"):
        image_id = inference_datum["image_id"]
        if image_id not in inference_dict:
            inference_dict[image_id] = []
        inference_dict[image_id].append({
            "bbox": inference_datum["bbox"],
            "score": inference_datum["score"],
            "label": coco_category_data[inference_datum["category_id"]],
        })

    all_results = []
    for image_id in inference_dict:
        inference_dict[image_id] = sorted(
            inference_dict[image_id],
            key=lambda x: float(x["score"]),
            reverse=True,
        )
        curr_results = {}

        file_name = coco_image_data[image_id]["file_name"]
        file_name = file_name.split(".")[0]
        video_uid, frame_num = file_name.split("/")
        frame_num = int(frame_num)
        # print(file_name)
        # pprint.pprint(coco_annots_data[image_id])
        # pprint.pprint(org_scod_dict[video_uid][frame_num])
        # struc_nouns = [x["structured_noun"] for x in org_scod_dict[
        #     video_uid][frame_num]["bbox"]]
        # gt_bboxes = [x["gt_bbox"] for x in org_scod_dict[
        #     video_uid][frame_num]["bbox"]]
        frame_name = org_scod_dict[video_uid][frame_num]["frame_name"]

        curr_results[frame_name] = {
            "video_uid": video_uid,
            "frame_cnt": frame_num,
            "image_name": coco_image_data[image_id]["file_name"],
            "image_folder": image_folder,
            "narration_tokens": [],
            "caption_prompt": [],
            "label_mapping": coco_category_data,
            # "gt_label": struc_nouns,
            # "gt_bboxes": gt_bboxes,
            "gt_bboxes": org_scod_dict[video_uid][frame_num]["bbox"],
            "pred_bboxes": inference_dict[image_id],
        }
        # pprint.pprint(curr_results)
        # raise
        all_results.append(curr_results)

    json.dump(
        all_results,
        open(out_file, "w"), 
        indent=4,
    )
    print("Saving results file to: {}".format(out_file))

    return None


if __name__ == "__main__":
    data_dir = "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge"
    image_folder = "pre_pnr_post_frames"
    coco_test_file = "coco_annotations/val.json"
    org_scod_files = [
        "/local1/hu528/ego4d_data_old/v1/annotations/fho_scod_train.json",
        "/local1/hu528/ego4d_data_old/v1/annotations/fho_scod_val.json",
    ]

    # TODO: Change this for different experiment.
    inference_file = (
        "baselines/DETR/"
        "ego4dv1_pnr_objects"
        "/inference/coco_instances_results.json"
    )

    # Files.
    inference_file = os.path.join(data_dir, inference_file)
    assert os.path.exists(inference_file), "File {} not exist!".format(inference_file)
    coco_test_file = os.path.join(data_dir, coco_test_file)
    assert os.path.exists(coco_test_file), "File {} not exist!".format(coco_test_file)
    image_folder = os.path.join(data_dir, image_folder)

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
    )
