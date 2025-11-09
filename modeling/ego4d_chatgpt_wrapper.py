import openai
import shutil
import numpy as np
import re, os, csv, json, random, math, copy, argparse
import pprint
from tqdm import tqdm

from ChatGPT import GPTSymbolic, _ground_fit_transform
from video_processing_utils import extract_ooc_spans


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_coco_format_files",
        default=None,
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--entity_vocab_file",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_folder",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--phase",
        choices=[
            "only_do_symbolic",
            "symbolic_and_definition",
            "only_do_definition",
        ],
        default=None,
        type=str,
    )
    parser.add_argument(
        "--object_not_narrated_remedial",
        choices=[
            "naive_append",
            None,
        ],
        default=None,
        type=str,
    )
    parser.add_argument(
        "--gpt_model_name",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
    )
    parser.add_argument(
        "--rerun_gpt",
        action="store_true",
        help=(
            "Useful if we want to re-augment the gpt results for the input file."
        )
    )
    parser.add_argument(
        "--ooc_category_id",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--tool_category_id",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--save_every",
        default=100,
        type=int,
        help=(
            "To prevent halting."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    parser.add_argument(
        "--gpt_debug",
        action="store_true",
    )
    args = parser.parse_args()
    return args


# Reset the gpt execution passes for each image data.
def clear_gpt_pass_cnt(data):
    for d_idx in range(len(data["images"])):
        data["images"][d_idx]["gpt_pass"] = 0
    pass


def gpt_on_coco_file(gpt_model, f, out_f_path=None, args=None):
    # If the output file actually exists, we can continue from it.
    if os.path.exists(out_f_path):
        contd = input("Output file: {} exists, do "
            "extend/overwrite/post-process (y/n)? ".format(out_f_path))
        if contd.lower() == "y":
            pass
        else:
            print("Aborting...")
            exit(-1)
        data = json.load(open(out_f_path))
        
    else:
        data = json.load(open(f))
        clear_gpt_pass_cnt(data)
        # Initial saving (for gpt_passes).
        json.dump(
            data,
            open(out_f_path, "w"),
            indent=4,
        )
        print("Clearing gpt passes to coco file: {}".format(out_f_path))

    new_data = copy.deepcopy(data)
    # Organize the annotations according to the image ids.
    annots_dict = {}
    if "annotations" in data:
        for annot_idx in range(len(data["annotations"])):
            annot = data["annotations"][annot_idx]
            image_id = annot["image_id"]
            if image_id not in annots_dict:
                annots_dict[image_id] = []
            annots_dict[image_id].append(annot_idx)

    # Entity vocabulary.
    if os.path.exists(args.entity_vocab_file):
        entity_vocab = json.load(open(args.entity_vocab_file))
    else:
        entity_vocab = {}

    # Dict for same-caption results.
    caption_results = {}

    # Params.
    process_cnt = 0
    ooc_not_found_cnt = [0, 0, 0]
    tool_not_found_cnt = [0, 0, 0]

    for idx in tqdm(range(len(data["images"])), desc="Running GPT"):

        # Info.
        ego4d_scod_id = data["images"][idx]["ego4d_scod_id"]
        current_frame_type = ego4d_scod_id.split("_")[-2]

        # Symbolic knowledge definition.
        symb_kg = {
            "conditions": {
                "ooc": {
                    "pre": [],"pnr": [],"post": []
                },
                "tool": {
                    "pre": [],"pnr": [],"post": []
                }
            },
            "current_frame_type": current_frame_type,
            "size": None,
            "spatial": None,
            "ooc": None,
            "tool": None,
            "definitions": {
                "ooc": None,
                "tool": None,
            }
        }

        caption = data["images"][idx]["caption"]
        # Un-narrated data points, just insert dummy symbolic dict.
        if "object_of_change" in caption:
            data["images"][idx]["symbolic"] = symb_kg
            continue

        caption = caption.lower()
        
        if args.dry_run:
            caption_results[caption] = True
            continue

        needs_symbolic, needs_definitions = True, True
        # Get gpt responses.
        if data["images"][idx]["gpt_pass"] > 0 and not args.rerun_gpt:
            # Skips if processed unless re-running.
            symb_kg = data["images"][idx]["symbolic"]
            needs_symbolic = False
            if args.debug:
                print("Processed idx = `{}`".format(idx))
        elif caption not in caption_results:
            raw_gpt_responses = gpt_model.pipeline(caption)
            gpt_responses = raw_gpt_responses["parsed_responses"]
            caption_results[caption] = gpt_responses
        else:
            gpt_responses = caption_results[caption]

        # Control the phases.
        if args.phase == "symbolic_and_definition":
            pass
        elif args.phase == "only_do_definition":
            needs_symbolic = False
        elif args.phase == "only_do_symbolic":
            needs_definitions = False

        # Get entity definitions.
        ooc_def, tool_def = None, None
        if needs_definitions:
            if needs_symbolic:
                assert "ooc" in gpt_responses and "tool" in gpt_responses
                ooc, tool = gpt_responses["ooc"], gpt_responses["tool"]
            else:
                assert "symbolic" in data["images"][idx], (
                    "The `only_do_definition` phase can only be run if the"
                    " phase `symbolic_and_definition` is run before!"
                )
                symb_kg = data["images"][idx]["symbolic"]
                ooc, tool = symb_kg["ooc"], symb_kg["tool"]
            if ooc is not None:
                if ooc not in entity_vocab:
                    ooc_def = gpt_model._define(ooc)["gpt_responses"]
                else:
                    ooc_def = entity_vocab[ooc]
                    if type(ooc_def) is list:
                        ooc_def = np.random.choice(ooc_def)
            if tool is not None:
                if tool not in entity_vocab:
                    tool_def = gpt_model._define(tool)["gpt_responses"]
                else:
                    tool_def = entity_vocab[tool]
                    if type(tool_def) is list:
                        tool_def = np.random.choice(tool_def)

            if args.phase == "only_do_definition":
                object_info = {
                    "ooc": {"name": ooc, "def": ooc_def},
                    "tool": {"name": tool, "def": tool_def},
                }
                for obj in ["ooc", "tool"]:
                    obj_name = object_info[obj]["name"]
                    if obj_name is not None:
                        key = "definitions"
                        symb_kg[key][obj] = object_info[obj]["def"]
                data["images"][idx]["symbolic"] = symb_kg

        # NOTE: For symbolic knowledge.
        if needs_symbolic:
            ooc, tool = gpt_responses["ooc"], gpt_responses["tool"]
            # NOTE: Organize the symbolic knowledge.
            object_info = {
                "ooc": {
                    "name": ooc,
                    "def": ooc_def,
                },
                "tool": {
                    "name": tool,
                    "def": tool_def,
                },
            }
            for obj in ["ooc", "tool"]:
                obj_name = object_info[obj]["name"]
                if obj_name is not None:
                    # Objects.
                    symb_kg[obj] = obj_name
                    # Conditions.
                    key = "{}_state_change".format(obj)
                    if "yes" in gpt_responses[key]["state_change"].lower():
                        if gpt_responses[key]["pre_state"] is not None:
                            symb_kg["conditions"][obj]["pre"].append(
                                gpt_responses[key]["pre_state"]
                            )
                        # Let's just use the post state for pnr (for now).
                        if gpt_responses[key]["post_state"] is not None:
                            symb_kg["conditions"][obj]["pnr"].append(
                                gpt_responses[key]["post_state"]
                            )
                            symb_kg["conditions"][obj]["post"].append(
                                gpt_responses[key]["post_state"]
                            )
                    pass
                    # Definitions.
                    key = "definitions"
                    symb_kg[key][obj] = object_info[obj]["def"]
                    # Spatial-relations.
                    key = "spatial"
                    symb_kg[key] = gpt_responses[key]
                    # Sizes.
                    key = "size"
                    symb_kg[key] = gpt_responses[key]
                pass
            pass
            # Augment the image data.
            data["images"][idx]["symbolic"] = symb_kg
            data["images"][idx]["gpt_pass"] += 1

        # Process the tokens positive.
        image_id = data["images"][idx]["id"]
        caption = data["images"][idx]["caption"].lower()
        ooc, tool = symb_kg["ooc"], symb_kg["tool"]
        if ooc is not None: ooc = ooc.lower()
        if tool is not None: tool = tool.lower()

        obj_type_dict = {
            "ooc": {
                "name": ooc,
                "category_id": args.ooc_category_id,
                "remedial_suffix": "Object of change is {}."
            },
            "tool": {
                "name": tool,
                "category_id": args.tool_category_id,
                "remedial_suffix": "Tool is {}."
            }
        }

        # TODO(telinwu): maybe use some srl here to default?
        all_obj_spans = []
        for obj_type in ["ooc", "tool"]: # Has to go in this order!
            obj_name = obj_type_dict[obj_type]["name"]
            if obj_name is None:
                # NOTE: We only force existence for ooc.
                if obj_type == "ooc":
                    obj_spans = [[0, len(caption)]]
                else:
                    obj_spans = None
            else:
                # Insert definitions.
                obj_def = symb_kg["definitions"][obj_type]
                entity_vocab[obj_name] = obj_def
                if obj_type == "ooc":
                    ooc_not_found_cnt[1] += 1
                elif obj_type == "tool":
                    tool_not_found_cnt[1] += 1
                matching = re.search(obj_name, caption)
                if matching is not None:
                    found = matching.span()
                    obj_spans = [[found[0], found[1]]]
                else:
                    if args.debug:
                        print("[NOT FOUND] {}: `{}` cannot be found in `{}`".format(
                            obj_type, obj_name, caption))
                    if obj_type == "ooc":
                        ooc_not_found_cnt[0] += 1
                    elif obj_type == "tool":
                        tool_not_found_cnt[0] += 1
                    obj_spans = [[0, len(caption)]]
                    # FIXME(telinwu): Current remedial process.
                    if args.object_not_narrated_remedial == "naive_append":
                        remedial_suffix = obj_type_dict[obj_type]["remedial_suffix"]
                        remedial_suffix = remedial_suffix.format(obj_name)
                        suffixed_caption = caption + " " + remedial_suffix
                        matching = re.search(obj_name, suffixed_caption)
                        assert matching is not None
                        found = matching.span()
                        obj_spans = [[found[0], found[1]]]
                        data["images"][idx]["caption"] = suffixed_caption
                    else:
                        raise NotImplementedError(
                            "args.object_not_narrated_remedial `{}` "
                            "not done yet!".format(
                            args.object_not_narrated_remedial)
                        )
            if obj_spans is not None:
                all_obj_spans.append(obj_spans)
            # Update annotations.
            obj_category_id = obj_type_dict[obj_type]["category_id"]
            curr_annots_idx = []
            if image_id in annots_dict:
                curr_annots_idx = annots_dict[image_id]
            obj_type_in_annot = False
            for a_idx in curr_annots_idx:
                if (
                    data["annotations"][a_idx]["category_id"] == obj_category_id
                    and obj_spans is not None
                ):
                    data["annotations"][a_idx]["tokens_positive"] = obj_spans
                if data["annotations"][a_idx]["category_id"] == obj_category_id:
                    obj_type_in_annot = True
                pass
            if obj_type_in_annot:
                if obj_type == "ooc":
                    ooc_not_found_cnt[2] += 1
                else:
                    tool_not_found_cnt[2] += 1

        # Updates the token positive for eval.
        assert len(all_obj_spans) >= 1
        data["images"][idx]["tokens_positive_eval"] = all_obj_spans

        # Periodically saving.
        if process_cnt % args.save_every == 0:
            json.dump(
                data,
                open(out_f_path, "w"),
                indent=4,
            )
            print("Periodically saving coco file to: {}".format(out_f_path))
            # Copy a file just in case.
            copy_f_path = out_f_path.replace(".json", "_copied.json")
            shutil.copy(out_f_path, copy_f_path)
            # Save entity vocab.
            json.dump(
                entity_vocab,
                open(args.entity_vocab_file, "w"),
                indent=4,
            )
            print("Periodically saving vocab file to: {}".format(args.entity_vocab_file))
        process_cnt += 1

    if args.dry_run:
        print("This file in total has {}"
              " unique captions to process.".format(len(caption_results)))
        pass

    # Final saving.
    json.dump(
        data,
        open(out_f_path, "w"),
        indent=4,
    )
    json.dump(
        entity_vocab,
        open(args.entity_vocab_file, "w"),
        indent=4,
    )
    print("Saving coco file to: {}".format(out_f_path))
    print("Saving vocab file to: {}".format(args.entity_vocab_file))

    # Some stats.
    print("OOC  NOT FOUND count: {} / {} ({} has annotations)".format(
        ooc_not_found_cnt[0], ooc_not_found_cnt[1], ooc_not_found_cnt[2]))
    print("TOOL NOT FOUND count: {} / {} ({} has annotations)".format(
        tool_not_found_cnt[0], tool_not_found_cnt[1], tool_not_found_cnt[2]))

    return 


def run(gpt_model, args=None):
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    for f in args.input_coco_format_files:
        f_name = f.split("/")[-1].split(".")[0]
        new_f_name = "gpt_" + f_name + ".json"
        out_f_path = os.path.join(args.output_folder, new_f_name)

        print("-"*50)
        print("Processing file: {} ...".format(f_name+".json"))
        gpt_on_coco_file(gpt_model, f, out_f_path=out_f_path, args=args)

    print("-"*50)
    return None


if __name__ == "__main__":
    # TODO(bryan): OpenAI relevant information.
    openai.organization = "org-Igmvps22Goq7QU5eddDp2SyR"
    os.environ["OPENAI_API_KEY"] = "REMOVED_OPENAI_KEY"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.Model.list()

    # Get arguments.
    args = get_args()
    entity_vocab_folder = "/".join(args.entity_vocab_file.split("/")[:-1])
    if not os.path.exists(entity_vocab_folder):
        os.makedirs(entity_vocab_folder)

    # Defines GPT model.
    gpt_mode = "run"
    if args.gpt_debug:
        gpt_mode = "debug"
    gpt_model = GPTSymbolic(model=args.gpt_model_name, mode=gpt_mode)

    # Bulk-run.
    if args.phase in [
        "only_do_symbolic",
        "symbolic_and_definition",
    ]:
        run(gpt_model, args=args)
    elif args.phase in [
        "only_do_definition"
    ]:
        run(gpt_model, args=args)

    print("All done!")
