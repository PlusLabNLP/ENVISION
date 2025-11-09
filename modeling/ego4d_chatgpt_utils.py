import shutil
import numpy as np
import re, os, csv, json, random, math, copy, argparse
import pprint
from tqdm import tqdm

from video_processing_utils import extract_ooc_spans
from video_processing_utils import parse_ego4d_scod_structured_noun
from run_language_augmenting_bbox_preds import extract_SRL_args


def cleanse_data(
    input_file,
    ooc_category_id=1,
    tool_category_id=2,
):
    input_file_name = input_file.split("/")[-1].split(".")[0]
    input_file_folder = "/".join(input_file.split("/")[:-1])
    output_file_name = "cleansed_{}.json".format(input_file_name)
    output_file = os.path.join(input_file_folder, output_file_name)
    print(output_file)
    data = json.load(open(input_file))

    image_id_to_idx_map = {}
    for idx in range(len(data["images"])):
        image = data["images"][idx]
        image_id = image["id"]
        image_id_to_idx_map[image_id] = idx
    
    annot_image_id_to_category_id_map = {}
    for idx in range(len(data["annotations"])):
        annot = data["annotations"][idx]
        image_id = annot["image_id"]
        if image_id not in annot_image_id_to_category_id_map:
            annot_image_id_to_category_id_map[image_id] = {}
        category_id = annot["category_id"]
        annot_image_id_to_category_id_map[image_id][category_id] = True

    for image_id in annot_image_id_to_category_id_map:
        category_ids = annot_image_id_to_category_id_map[image_id]
        # Remove those without tool annotations but with tool
        # `tokens_positive_eval` to prevent from over-using bbox spatial
        # semantics.
        image_idx = image_id_to_idx_map[image_id]
        max_id = max(category_ids)
        if max_id == ooc_category_id:
            datum = data["images"][image_idx]
            tokens_positive_eval = datum["tokens_positive_eval"]
            assert len(tokens_positive_eval) >= 1
            tokens_positive_eval = [tokens_positive_eval[0]]
            data["images"][image_idx]["tokens_positive_eval"] = tokens_positive_eval
        assert (
            len(category_ids) == len(data["images"][image_idx]["tokens_positive_eval"])
            or
            max_id == len(data["images"][image_idx]["tokens_positive_eval"])
        ), (
            "image_id: {} category_ids: {} tokens_positive_eval: {}".format(
                image_id,
                category_ids,
                data["images"][image_idx]["tokens_positive_eval"]
            )
        )
        pass

    json.dump(
        data,
        open(output_file, "w"),
        indent=4,
    )
    print("Saving file to: {}".format(output_file))

    return output_file


def gpt_performance_from_raw_scod_files(
    gpt_data_files=None,
    raw_scod_train_file=None,
    raw_scod_val_file=None,
    verbose=False,
    print_correct=False,
):
    raw_scod_data = []
    if raw_scod_train_file is not None:
        raw_scod_data += json.load(open(raw_scod_train_file))["clips"]
    if raw_scod_val_file is not None:
        raw_scod_data += json.load(open(raw_scod_val_file))["clips"]

    # 5f7e3f1e-f4db-461e-8344-c8f130985635_71760_pre_0

    raw_scod_gt = {}
    frame_ooc_annotated_cnt = 0
    for clip in raw_scod_data:
        video_uid = clip["video_uid"]
        for fr in ["pre", "pnr", "post"]:
            fr_name = "{}_frame".format(fr)
            annots = clip[fr_name]
            frame_num = annots["frame_number"]
            frame_bboxes = annots["bbox"]
            ego4d_scod_id = "{}_{}_{}".format(video_uid, frame_num, fr)
            if  ego4d_scod_id in raw_scod_gt:
                continue
            raw_scod_gt[ego4d_scod_id] = {
                "ooc": [],
                "tool": [],
            }
            ooc_exists = False
            for frame_bbox in frame_bboxes:
                object_type = frame_bbox["object_type"]
                structured_noun = frame_bbox["structured_noun"]
                if object_type == "object_of_change":
                    raw_scod_gt[ego4d_scod_id]["ooc"].append(structured_noun)
                    if not ooc_exists:
                        ooc_exists = False
                        frame_ooc_annotated_cnt += 1
                elif object_type == "tool":
                    raw_scod_gt[ego4d_scod_id]["tool"].append(structured_noun)
            # raise
            pass
        pass
    print("OOC annotation counts: {}".format(frame_ooc_annotated_cnt))

    assert gpt_data_files is not None
    if type(gpt_data_files) is str:
        gpt_data_files = [gpt_data_files]
    assert type(gpt_data_files) is list and type(gpt_data_files[0]) is str

    if not print_correct:
        print("-"*50)
    for gpt_data_file in gpt_data_files:
        ooc_perf_arr = []
        tool_perf_arr = []
        captions_used = {}

        print("Processing {}".format(gpt_data_file))
        if print_correct:
            print("-"*50)

        gpt_data = json.load(open(gpt_data_file))["images"]
        for image in tqdm(gpt_data, desc="GPT Results"):
            # Skipped repetitive captions.
            caption = image["caption"]
            if caption not in captions_used:
                captions_used[caption] = True
            else:
                continue
            # Skipped non-annotated data points.
            if "symbolic" not in image:
                continue
            assert "ego4d_scod_id" in image
            ego4d_scod_id = "_".join(image["ego4d_scod_id"].split("_")[:-1])
            assert ego4d_scod_id in raw_scod_gt
            ooc_labels = raw_scod_gt[ego4d_scod_id]["ooc"]
            tool_labels = raw_scod_gt[ego4d_scod_id]["tool"]

            gpt_ooc = image["symbolic"]["ooc"]
            gpt_tool = image["symbolic"]["tool"]
            
            has_ooc_labels, has_tool_labels = False, False
            ooc_found, tool_found = False, False

            for ooc_label in ooc_labels:
                if ooc_label is None: continue
                assert ooc_label is not None
                has_ooc_labels = True
                allowed_names = parse_ego4d_scod_structured_noun(ooc_label)
                new_allowed_names = []
                for allowed_name in allowed_names:
                    new_allowed_names += allowed_name.split()
                new_allowed_names = sorted(list(new_allowed_names))
                allowed_names = new_allowed_names
                for allowed_name in allowed_names:
                    if (
                        (gpt_ooc is not None and allowed_name in gpt_ooc)
                        or
                        (gpt_ooc is not None and gpt_ooc in allowed_name)
                    ):
                        ooc_found = True
                    pass
                pass
            pass

            for tool_label in tool_labels:
                if tool_label is None: continue
                assert tool_label is not None
                has_tool_labels = True
                allowed_names = parse_ego4d_scod_structured_noun(tool_label)
                for allowed_name in allowed_names:
                    if (
                        (gpt_tool is not None and allowed_name in gpt_tool and allowed_name != "tool")
                        or
                        (gpt_tool is not None and gpt_tool in allowed_name and allowed_name != "tool")
                    ):
                        tool_found = True
                    pass
                pass
            pass

            if has_ooc_labels:
                if ooc_found:
                    ooc_perf_arr.append(1)
                    if print_correct:
                        print("[V] [OOC]  ChatGPT: {}  GT: {} | of: {}".format(
                            gpt_ooc, ooc_labels, caption))
                else:
                    ooc_perf_arr.append(0)
                    if verbose or print_correct:
                        print("[X] [OOC]  ChatGPT: {}  GT: {} | of: {}".format(
                            gpt_ooc, ooc_labels, caption))
            elif print_correct:
                print("[-] [OOC] Raw data no labels | of: {}".format(caption))
            if has_tool_labels:
                if tool_found:
                    tool_perf_arr.append(1)
                    if print_correct:
                        print("[V] [TOOL] ChatGPT: {}  GT: {} | of: {}".format(
                            gpt_tool, tool_labels, caption))
                else:
                    tool_perf_arr.append(0)
                    if verbose or print_correct:
                        print("[X] [TOOL] ChatGPT: {}  GT: {} | of: {}".format(
                            gpt_tool, tool_labels, caption))
            elif print_correct:
                print("[-] [TOOL] Raw data no labels | of: {}".format(caption))
            if print_correct:
                print("-"*50)
            pass
        pass

        ooc_acc = np.mean(ooc_perf_arr)
        tool_acc = np.mean(tool_perf_arr)

        print("OOC  performance accuracy: {:.3f}".format(ooc_acc))
        print("TOOL performance accuracy: {:.3f}".format(tool_acc))
        print("-"*50)
        
    return None


def compare_gpt_with_gt(
    gpt_data_file=None,
    coco_gt_file=None,
):
    gpt_data = json.load(open(gpt_data_file))["images"]
    coco_gt = json.load(open(coco_gt_file))["images"]

    assert len(gpt_data) == len(coco_gt), (len(gpt_data), len(coco_gt))

    ooc_span_correct_cnt = 0
    ooc_span_total_cnt = 0
    overlapping_ratios = []
    not_em_overlapping_ratios = []

    tool_span_correct_cnt = 0
    tool_span_total_cnt = 0
    missing_tool_span_total_cnt = 0
    tool_overlapping_ratios = []

    for i in range(len(gpt_data)):
        gpt_image = gpt_data[i]
        coco_image = coco_gt[i]

        ooc_gpt_span = gpt_image["tokens_positive_eval"][0][0]
        ooc_coco_span = coco_image["tokens_positive_eval"][0][0]
        if ooc_coco_span == ooc_gpt_span:
            ooc_span_correct_cnt += 1
        ooc_span_total_cnt += 1

        overlapping = len(
            range(
                max(ooc_gpt_span[0], ooc_coco_span[0]),
                min(ooc_gpt_span[1], ooc_coco_span[1]),
            )
        )
        overlapping_ratio = overlapping / (ooc_coco_span[1]-ooc_coco_span[0])
        # overlapping_ratio = overlapping / (ooc_gpt_span[1]-ooc_gpt_span[0])
        overlapping_ratios.append(overlapping_ratio)
        if ooc_coco_span != ooc_gpt_span:
            # print(ooc_gpt_span, ooc_coco_span)
            # print(overlapping_ratio)
            not_em_overlapping_ratios.append(overlapping_ratio)

        if (
            len(gpt_image["tokens_positive_eval"]) > 1
            and
            len(coco_image["tokens_positive_eval"]) > 1
        ):
            tool_gpt_span = gpt_image["tokens_positive_eval"][1][0]
            tool_coco_span = coco_image["tokens_positive_eval"][1][0]
            if tool_coco_span == tool_gpt_span:
                tool_span_correct_cnt += 1
            tool_span_total_cnt += 1

            overlapping = len(
                range(
                    max(tool_gpt_span[0], tool_coco_span[0]),
                    min(tool_gpt_span[1], tool_coco_span[1]),
                )
            )
            overlapping_ratio = overlapping / (tool_coco_span[1]-tool_coco_span[0])
            tool_overlapping_ratios.append(overlapping_ratio)
        elif (
            len(gpt_image["tokens_positive_eval"]) <= 1
            and
            len(coco_image["tokens_positive_eval"]) > 1
        ):
            missing_tool_span_total_cnt += 1
            
    print("OOC exact match count = {} / {} = {:.3f}%".format(
        ooc_span_correct_cnt, ooc_span_total_cnt,
        ooc_span_correct_cnt/ooc_span_total_cnt*100.0)
    )
    avg_overlapping_ratio = np.mean(overlapping_ratios)
    print("OOC avg overlapping ratio = {:.3f}".format(avg_overlapping_ratio))

    avg_not_em_overlapping_ratios = np.mean(not_em_overlapping_ratios)
    print("OOC avg not exact match overlapping ratio = {:.3f}".format(avg_not_em_overlapping_ratios))

    print("Tool exact match count = {} / {} = {:.3f}%".format(
        tool_span_correct_cnt, tool_span_total_cnt,
        tool_span_correct_cnt/tool_span_total_cnt*100.0)
    )
    avg_tool_overlapping_ratio = np.mean(tool_overlapping_ratios)
    print("Tool avg overlapping ratio = {:.3f}".format(avg_tool_overlapping_ratio))
    print("Tool missing span count    = {}".format(missing_tool_span_total_cnt))

    return None


class SRLParser(object):
    def __init__(self, srl_model):
        if type(srl_model) is str:
            from allennlp_models import pretrained
            from allennlp.predictors.predictor import Predictor
            self.srl_model = pretrained.load_predictor(
                srl_model
            )
        else:
            self.srl_model = srl_model
        pass

    def get_arg_by_tags(self, sentence, tags_of_interests=["ARG1"]):
        if type(tags_of_interests) is str:
            tags_of_interests = [tags_of_interests]
        srl_res = self.srl_model.predict(sentence=sentence)
        parsed_args_list = extract_SRL_args(srl_res)
        # `parsed_args` are ordered dict.
        # Getting the right SRL args.
        parsed_args_to_use = None
        for parsed_args, srl in parsed_args_list:
            found_parsed_args = True
            for _arg in tags_of_interests:
                if _arg not in parsed_args:
                    found_parsed_args = False
                pass
            if found_parsed_args:
                parsed_args_to_use = parsed_args
                break
            pass
        res_args_dict = None
        if parsed_args_to_use is not None:
            res_args_dict = {}
            for _arg in tags_of_interests:
                res_args_dict[_arg] = parsed_args_to_use[_arg]
        return res_args_dict


def cleanse_gpt_v2up_data(
    raw_f,
    ref_f,
):
    assert "raw_" in raw_f, (
        "Please rename the file {} with a prefix `raw_` before"
        " the last name.json".format(raw_f)
    )

    raw_data = json.load(open(raw_f))
    ref_data = json.load(open(ref_f))

    raw_images = raw_data["images"]
    raw_annots = raw_data["annotations"]
    ref_images = ref_data["images"]
    ref_annots = ref_data["annotations"]

    assert len(raw_images) == len(ref_images)
    assert len(raw_annots) == len(ref_annots)

    raw_annots_dict = {}
    for a_idx in range(len(raw_annots)):
        raw_annot = raw_annots[a_idx]
        image_id = raw_annot["image_id"]
        if image_id not in raw_annots_dict: raw_annots_dict[image_id] = []
        raw_annots_dict[image_id].append(a_idx)

    scenario_cnt = 0
    ref_tool_none_but_raw_has_cnt = 0

    for i_idx in tqdm(range(len(raw_images)), desc="Cleansing"):
        raw_image = raw_images[i_idx]
        ref_image = ref_images[i_idx]
        raw_caption  = raw_image["caption"]
        ref_caption  = ref_image["caption"]

        raw_image_id = raw_image["id"]
        ref_image_id = ref_image["id"]
        assert raw_image_id == ref_image_id
        image_id = raw_image_id = ref_image_id

        if raw_caption != ref_caption:
            if image_id in raw_annots_dict:
                curr_raw_annots = raw_annots_dict[image_id]
                for a_idx in curr_raw_annots:
                    # NOTE: Use the ref annotations directly.
                    raw_data["annotations"][a_idx] = ref_annots[a_idx]
                    # pprint.pprint(raw_annots[a_idx])
                    # pprint.pprint(ref_annots[a_idx])
                    pass
                pass
            ungroundable_tool_symbolic = None
            if (
                "Object of change is" in raw_caption
                and
                "Tool is" in raw_caption
            ):
                raise NotImplementedError("Not dealt with this yet!")
            elif (
                "Tool is" in raw_caption
            ):
                # Only augment the tool if reference one does not have.
                if ref_image["symbolic"]["tool"] is None:
                    ungroundable_tool_symbolic = raw_image["symbolic"]
                    ref_tool_none_but_raw_has_cnt += 1
                pass
            elif (
                "Object of change is" in raw_caption
            ):
                pass  # Do nothing here as the image data will be replaced.

            # NOTE: Use the ref image dict directly.
            raw_data["images"][i_idx] = ref_image
            raw_data["images"][i_idx]["symbolic"][
                "ungroundable_tool_symbolic"] = ungroundable_tool_symbolic
            # pprint.pprint(raw_image)
            # pprint.pprint(ref_image)

            scenario_cnt += 1

    print("Scenario counts: {}".format(scenario_cnt))
    print("Ref tool none but raw tool is not: {}".format(ref_tool_none_but_raw_has_cnt))

    out_f = raw_f.replace("raw_", "")
    json.dump(
        raw_data,
        open(out_f, "w"),
        indent=4,
    )
    print("Saving file to {}".format(out_f))

    return None


if __name__ == "__main__":

    """
    raw_files = [
        (
            "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge"
            "/paper_data/all_frames/narrated/gpt_v2_with_tool"
            "/raw_train_scod_all_frames_narrated_gpt_v2_with_tool.json"
        ),
        (
            "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge"
            "/paper_data/all_frames/narrated/gpt_v2_with_tool"
            "/raw_val_scod_all_frames_narrated_gpt_v2_with_tool.json"
        ),
    ]
    ref_files = [
        (
            "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge"
            "/paper_data/all_frames/narrated/gpt_v1_srl_arg1_with_tool"
            "/train_scod_all_frames_narrated_gpt_v1_srl_arg1_with_tool.json"
        ),
        (
            "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge"
            "/paper_data/all_frames/narrated/gpt_v1_srl_arg1_with_tool"
            "/val_scod_all_frames_narrated_gpt_v1_srl_arg1_with_tool.json"
        ),
    ]

    for raw_file, ref_file in zip(raw_files, ref_files):
        cleanse_gpt_v2up_data(
            raw_file,
            ref_file,
        )
    pass
    exit(-1)
    """

    """
    # Testing SRL class.
    srl_parser = SRLParser(srl_model="structured-prediction-srl-bert")
    sentence = "Please dip the sponge into the bucket."
    srl_res = srl_parser.get_arg_by_tags(
        sentence=sentence,
        tags_of_interests=["ARG1"],
    )
    if srl_res is not None:
        pprint.pprint(srl_res)
        arg1_seg = srl_res["ARG1"]
        print(arg1_seg)
    exit(-1)
    """

    # Data cleansing.
    """
    input_file = "./chatgpt_results/gpt_xxx.json"
    output_file = cleanse_data(input_file)
    exit(-1)
    """

    # Performance from raw files.
    """
    raw_scod_train_file = (
        "/local1/telinwu/research/resources/Ego4D/ego4d_data/v1"
        "/annotations/fho_scod_train.json"
    )
    raw_scod_val_file = (
        "/local1/telinwu/research/resources/Ego4D/ego4d_data/v1"
        "/annotations/fho_scod_val.json"
    )

    gpt_data_files = [
        # "./chatgpt_results/gpt_first100_narrated_gt_srl_arg1_with_tool_strict.json",
        # "/local1/bryanzhou008/jarvis/GPT/train_res.json",
        # "/local1/bryanzhou008/jarvis/GPT/val_res.json",
        # "/local1/bryanzhou008/jarvis/GPT/train_res_mark_2.json",
        # "/local1/bryanzhou008/jarvis/GPT/val_res_mark_2.json",
        "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/paper_data/all_frames/narrated/gpt_v3_with_tool/raw_train_scod_all_frames_narrated_gpt_v3_with_tool.json",
        "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/paper_data/all_frames/narrated/gpt_v3_with_tool/raw_val_scod_all_frames_narrated_gpt_v3_with_tool.json",
    ]

    gpt_performance_from_raw_scod_files(
        gpt_data_files=gpt_data_files,
        raw_scod_train_file=raw_scod_train_file,
        raw_scod_val_file=raw_scod_val_file,
        verbose=False,
        print_correct=False,
    )
    exit(-1)
    """

    # """
    gpt_data_file = "/local1/bryanzhou008/jarvis/GPT/val_res.json"
    gpt_data_file = "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/paper_data/all_frames/narrated/gpt_v3_with_tool/raw_val_scod_all_frames_narrated_gpt_v3_with_tool.json"
    coco_gt_file = (
        "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge"
        "/coco_annotations_all_frames/val_narrated_gt_srl_arg1_with_tool.json"
    )
    coco_gt_file = "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/paper_data/all_frames/narrated/gt_srl_arg1/val_scod_all_frames_narrated_gt_srl_arg1_with_tool.json"

    # """
    gpt_data_file = (
        "/local1/bryanzhou008/jarvis/git_jarvis/project_jarvis/modeling"
        "/chatgpt_results_trek150_mark_1/gpt_trek150_1st_frame_only_narrated_gt.json"
    )
    coco_gt_file = (
        "/local1/telinwu/research/resources/TREK-150/coco_annotations"
        "/trek150_1st_frame_only_narrated_gt.json"
    )
    # """

    compare_gpt_with_gt(
        gpt_data_file=gpt_data_file,
        coco_gt_file=coco_gt_file,
    )
    exit(-1)
    # """

    print("All done!")
