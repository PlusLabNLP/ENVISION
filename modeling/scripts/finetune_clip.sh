ANNOTS_FOLDER="/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/clip_annotations_all_frames"

TRAIN_FILE="${ANNOTS_FOLDER}/fho_scod_train_enlarged_bbox0p2.csv"
VALIDATION_FILE="${ANNOTS_FOLDER}/fho_scod_train_enlarged_bbox0p2.csv"
TEST_FILE="${ANNOTS_FOLDER}/fho_scod_train_enlarged_bbox0p2.csv"

TRAIN_FILE="${ANNOTS_FOLDER}/fho_scod_train.csv"
VALIDATION_FILE="${ANNOTS_FOLDER}/fho_scod_train.csv"
TEST_FILE="${ANNOTS_FOLDER}/fho_scod_train.csv"

TRAIN_FILE="${ANNOTS_FOLDER}/fho_scod_val.csv"
VALIDATION_FILE="${ANNOTS_FOLDER}/fho_scod_val.csv"
TEST_FILE="${ANNOTS_FOLDER}/fho_scod_val.csv"

TRAIN_FILE="${ANNOTS_FOLDER}/fho_scod_train_srl_v_arg1.csv"
VALIDATION_FILE="${ANNOTS_FOLDER}/fho_scod_train_srl_v_arg1.csv"
TEST_FILE="${ANNOTS_FOLDER}/fho_scod_train_srl_v_arg1.csv"

CUDA_VISIBLE_DEVICES="6" \
python3 finetune_clip.py \
    --output_dir "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/baselines/clip_scod_train_finetuned" \
    --output_dir "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/baselines/clip_scod_train_enlarged_bbox0p2_finetuned" \
    --output_dir "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/baselines/clip_scod_val_finetuned" \
    --output_dir "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/baselines/clip_large_scod_train_finetuned" \
    --output_dir "/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/baselines/clip_large_scod_train_srl_v_arg1_finetuned" \
    --model_name_or_path "openai/clip-vit-base-patch32" \
    --model_name_or_path "openai/clip-vit-large-patch14" \
    --max_seq_length 77 \
    --train_file "${TRAIN_FILE}" \
    --validation_file "${VALIDATION_FILE}" \
    --test_file "${TEST_FILE}" \
    --image_column image_path \
    --caption_column caption \
    --remove_unused_columns=False \
    --do_train \
    --do_eval \
    --per_device_train_batch_size="32" \
    --per_device_eval_batch_size="32" \
    --learning_rate="1e-6" \
    --warmup_steps="0" \
    --weight_decay 0.1 \
    --overwrite_output_dir \


    # --data_dir $PWD/data \
    # --dataset_name ydshieh/coco_dataset_script \
    # --dataset_config_name=2017 \
    # --push_to_hub
