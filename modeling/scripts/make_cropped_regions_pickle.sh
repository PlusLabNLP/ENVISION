
python3 run_language_augmenting_bbox_preds.py \
    --ego4d_videos_root "/local1/hu528/ego4d_data/v1/full_scale" \
    --ego4d_annotations_root "/local1/hu528/ego4d_data/v1/annotations/" \
    --processed_scod_image_folder "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/pre_pnr_post_png_frames" \
    --processed_inference_json_file "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/baselines/VideoIntern/ego4dv1_pnr_objects/inference/test_with_gt.json" \
    --processed_inference_json_file "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/baselines/DETR/ego4dv1_pnr_objects/inference/coco_instances_results_with_gt.json" \
    --processed_inference_json_file "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/baselines/DETR/ego4dv2_pre_pnr_post_objects/inference/coco_instances_results_with_gt.json" \
    --process_first_k 100000 \
    --narration_pass "narration_pass_1" \
    --top_k_bboxes 5 \
    --best_k 5 \
    --bbox_inference_strategy "top-1" \
    --output_image_cropped_pickle_file "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/baselines/DETR/cropped_regions/ego4d_scod_val_all_frames_top_5_cropped_regions.pickle" \


    # --verbose \
    # --use_data_criteria \
    # --use_results_file "./glip_result_json_files/lang_aug_on_detr_all_top_5_best_5.json" \
    # --use_results_file "./glip_result_json_files/lang_aug_on_detr_first_500_top_4_best_4.json" \
    # --use_results_file "./glip_result_json_files/lang_aug_on_detr_fail_top_1_succeed_best_4.json" \
    # --use_data_criteria \
    # --output_image_cropped_pickle_file "./glip_result_json_files/ego4d_scod_val_top_5_cropped_regions.pickle" \
