# GPT Bulk Run

python3 ego4d_chatgpt_wrapper.py \
    --input_coco_format_files \
        "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/coco_annotations_all_frames/first100_narrated_gt_srl_arg1_with_tool_strict.json" \
    --entity_vocab_file \
        "./chatgpt_results/ego4d_object_definitions.json" \
    --output_folder "./chatgpt_results" \
    --phase "symbolic_and_definition" \
    --gpt_model_name "gpt-3.5-turbo" \
    --object_not_narrated_remedial "naive_append" \
    --save_every 5 \
    --debug \
    # --gpt_debug \
    # --dry_run \
    #     "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/coco_annotations/test_narrated_srl_arg1_only_no_hand.json" \
    #     "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/coco_annotations_all_frames/first1_narrated_gt_srl_arg1_with_tool_strict.json" \


    # --phase "only_do_symbolic" \
    # --phase "only_do_definition" \

    # --rerun_gpt \
