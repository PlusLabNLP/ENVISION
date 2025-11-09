OUT_DIR="/local1/telinwu/research/resources/Ego4D/ego4d_scod_challenge/baselines/DETR"

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
CUDA_VISIBLE_DEVICES=5 \
python3 train_net.py \
    --num-gpus 1 \
    --config detr_256_6_6_torchvision_ego4dv2.yaml \
    --eval-only \
    MODEL.WEIGHTS "${OUT_DIR}/ego4dv2_pre_pnr_post_objects/model_final.pth" \
    # MODEL.WEIGHTS "${OUT_DIR}/ego4dv1_pnr_objects/model_0269999.pth" \


    # --config detr_256_6_6_torchvision_ego4dv1.yaml \
    # MODEL.WEIGHTS "${OUT_DIR}/ego4dv1_pnr_objects/model_0089999.pth" \
