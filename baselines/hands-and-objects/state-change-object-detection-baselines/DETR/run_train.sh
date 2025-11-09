# Check L23-26 in train_net.py is changing cooc annotations!

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
CUDA_VISIBLE_DEVICES=1,2,3,4 \
python3 train_net.py \
    --num-gpus 4 \
    --config detr_256_6_6_torchvision_ego4dv2.yaml \
    # --config detr_256_6_6_torchvision_ego4dv1.yaml \
