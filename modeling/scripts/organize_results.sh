# COCO_TEST_FILE="coco_annotations/tiny_test.json"
# COCO_TEST_FILE="/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/coco_annotations/tiny_test_narrated_gt_srl_arg1.json"
# COCO_TEST_FILE="coco_annotations/val.json"
# COCO_TEST_FILE="/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/coco_annotations/val_narrated_gt_srl_arg1.json"
# COCO_TEST_FILE="/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/coco_annotations/val_narrated_srl_arg1_only.json"
# COCO_TEST_FILE="/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/coco_annotations/val_narrated_ground_full_sentence.json"

# INFERENCE_FILE="/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/baselines/GLIP/naive_scod_finetuning_v1/eval_ego4d_scod_val_finetuned_45K/eval/model_0045000/inference/test/bbox.json"
# INFERENCE_FILE="/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/baselines/GLIP/narrated_scod_finetuning_v1/eval_ego4d_narrated_scod_val_finetuned_45K_full_sent/eval/model_0045000/inference/narrated_ego4d_test/bbox.json"
# INFERENCE_FILE="/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/baselines/GLIP/narrated_scod_finetuning_v1/eval_ego4d_narrated_scod_val_finetuned_45K_full_sent_new/eval/model_0045000/inference/narrated_ego4d_test/bbox.json"


# COCO_TEST_FILE="coco_annotations_all_frames/val.json"
# COCO_TEST_FILE="/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/coco_annotations_all_frames/val_narrated_gt_srl_arg1.json"
# COCO_TEST_FILE="/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/coco_annotations_all_frames/val_narrated_srl_arg1_only.json"
# COCO_TEST_FILE="/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/coco_annotations_all_frames/val_narrated_ground_full_sentence.json"
COCO_TEST_FILE="/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/coco_annotations_all_frames/val_narrated_gt_srl_arg1_first_10_corrects.json"
COCO_TEST_FILE="/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/coco_annotations_all_frames/val_narrated_gt_srl_arg1_with_tool_strict.json"

# INFERENCE_FILE="/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/baselines/GLIP/narrated_scod_finetuning_v2/eval_ego4d_narrated_scod_val_finetuned_45K_gt_arg1/eval/model_0045000/inference/narrated_ego4d_test/bbox.json"
# INFERENCE_FILE="/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/baselines/GLIP/narrated_scod_finetuning_v2/eval_ego4d_narrated_scod_val_finetuned_45K_arg1_only/eval/model_0045000/inference/narrated_ego4d_test/bbox.json"
# INFERENCE_FILE="/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/baselines/GLIP/narrated_scod_finetuning_v2/eval_ego4d_narrated_scod_val_finetuned_45K_full_sent/eval/model_0045000/inference/narrated_ego4d_test/bbox.json"
# INFERENCE_FILE="pre_post_pnr_results.json"
INFERENCE_FILE="/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/baselines/GLIP/narrated_scod_finetuning_large_v2/first_10_corrects_new/eval/model_0045000/inference/narrated_ego4d_test/bbox.json"
INFERENCE_FILE="/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/baselines/GLIP/narrated_scod_finetuning_large_v2_gt_srl_arg1_with_tool/eval_ego4d_narrated_scod_val_finetuned_65K_with_tool_strict/eval/model_0065000/inference/narrated_ego4d_test/bbox.json"
# INFERENCE_FILE="/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/baselines/GLIP/narrated_scod_finetuning_large_v2/eval_ego4d_narrated_scod_val_finetuned_45K_gt_arg1/eval/model_0045000/inference/narrated_ego4d_test/bbox.json"


DATA_ROOT="/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge"
IMAGE_FOLDER="pre_pnr_post_frames"

python3 organize_results.py \
    --org_scod_files \
        "/local1/hu528/ego4d_data_old/v1/annotations/fho_scod_train.json" \
        "/local1/hu528/ego4d_data_old/v1/annotations/fho_scod_val.json" \
    --data_root "${DATA_ROOT}" \
    --image_folder "${IMAGE_FOLDER}" \
    --coco_test_file "${COCO_TEST_FILE}" \
    --inference_file "${INFERENCE_FILE}" \
    --frames_to_use "pnr" \
    --no_skip_empty_preds \


    # --frames_to_use "pre" "pnr" "post" \
