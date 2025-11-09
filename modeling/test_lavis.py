import os, json
import torch
from collections import OrderedDict
from tqdm import tqdm
from PIL import Image
from lavis.models import model_zoo
from lavis.models import load_model_and_preprocess


def blip_predict_caption(lavis_model, vis_processors, image):
    lavis_caption = lavis_model.generate(
        {
            "image": vis_processors["eval"](image).unsqueeze(0).to(device),
            # "image": torch.stack([vis_processors["eval"](image).to(device),vis_processors["eval"](image).to(device)]),
        },
        num_beams=1,
        repetition_penalty=0.9,
        num_captions=1,
        use_nucleus_sampling=True,
    )
    return lavis_caption


def caption_scod_video_aggregated_clips_list(
    list_file=None,
    image_root=None,
    lavis_model=None,
    vis_processors=None,
):
    clip_lists = json.load(open(list_file))
    video_uids = sorted(list(clip_lists.keys()))
    res = OrderedDict()
    for video_uid in tqdm(video_uids, desc="SCOD Video"):
        for clip_set in tqdm(clip_lists[video_uid], desc="Video: {}".format(video_uid)):
            pnr_frame_nums = [x[1] for x in clip_set]
            idx = len(pnr_frame_nums) // 2
            pnr_frame_num = pnr_frame_nums[idx]
            file_name = "{}/{}.jpg".format(video_uid, pnr_frame_num)
            file_path = os.path.join(image_root, file_name)
            assert os.path.exists(file_path)
            img = Image.open(file_path)
            blip2_caption = blip_predict_caption(lavis_model, vis_processors, img)
            for pnr_frame_num in pnr_frame_nums:
                file_name = "{}/{}.jpg".format(video_uid, pnr_frame_num)
                res[file_name] = blip2_caption
            pass
            json.dump(
                res,
                open("./glip_result_json_files/scod_testset_lavis_captions.json", "w"),
                indent=4,
            )
        pass
    pass

    return None


def caption_ego4d_scod_raw_file(
    files,
    image_root=None,
    lavis_model=None,
    vis_processors=None,
):
    res = OrderedDict()

    for f in tqdm(files, desc="Files"):
        clips = json.load(open(f))["clips"]
        for clip in tqdm(clips, desc="Processing {}".format(f.split("/")[-1])):
            video_uid  = clip["video_uid"]
            for fr in ["pre", "pnr", "post"]:
                fr_name = "{}_frame".format(fr)
                if fr_name not in clip:
                    continue
                frame_num = clip[fr_name]["frame_number"]
                file_name = "{}/{}.jpg".format(video_uid, frame_num)
                if file_name in res:
                    continue
                file_path = os.path.join(image_root, file_name)
                assert os.path.exists(file_path)
                img = Image.open(file_path)
                blip2_caption = blip_predict_caption(lavis_model, vis_processors, img)
                res[file_name] = blip2_caption
            pass
        pass
    pass

    json.dump(
        res,
        open("./glip_result_json_files/scod_val_lavis_captions.json", "w"),
        indent=4,
    )

    return None


if __name__ == "__main__":
    ###########################################################################
    device = torch.device("cuda")
    lavis_model, vis_processors = None, None
    lavis_model, vis_processors, _ = load_model_and_preprocess(
        # name="blip_caption",
        name="blip2_t5",
        # model_type="base_coco",
        model_type="caption_coco_flant5xl",
        is_eval=True,
        device=device
    )
    ###########################################################################

    ego4d_scod_raw_files = [
        # "/local1/telinwu/research/resources/Ego4D/ego4d_data/v1/annotations/fho_scod_train.json",
        "/local1/telinwu/research/resources/Ego4D/ego4d_data/v1/annotations/fho_scod_val.json"
    ]
    caption_ego4d_scod_raw_file(
        files=ego4d_scod_raw_files,
        image_root="/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/pre_pnr_post_frames",
        lavis_model=lavis_model,
        vis_processors=vis_processors,
    )

    """
    caption_scod_video_aggregated_clips_list(
        list_file="./glip_result_json_files/scod_testset_clip_sets.json",
        image_root="/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/pre_pnr_post_frames",
        lavis_model=lavis_model,
        vis_processors=vis_processors,
    )

    exit(-1)
    """

    """
    img_paths = [
        "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/pre_pnr_post_frames/36420847-b741-4b86-9a31-3a5bb4e296bc/114859.jpg",
        "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/pre_pnr_post_frames/36420847-b741-4b86-9a31-3a5bb4e296bc/115648.jpg",
        "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/pre_pnr_post_frames/36420847-b741-4b86-9a31-3a5bb4e296bc/121230.jpg",
    ]

    for img_path in img_paths:
        img = Image.open(img_path)
        blip2_caption = blip_predict_caption(lavis_model, vis_processors, img)
        print(blip2_caption)
    raise

    while True:
        img_path = input("Image path (-1 to stop): ").strip()
        if img_path == "-1":
            break
        while not os.path.exists(img_path):
            img_path = input("Retype image path (-1 to stop): ").strip()
            if img_path == "-1":
                break
        if img_path == "-1":
            break
        img = Image.open(img_path)
        blip2_caption = blip_predict_caption(lavis_model, vis_processors, img)
        print(blip2_caption)

    exit(-1)
    # """

    print("All done!")
