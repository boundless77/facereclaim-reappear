CUDA_VISIBLE_DEVICES=3 python reclaim_from_clip.py \
    --fake_root /mnt/home/chenxiaofu/facetrack/data/deepfakes/image/celeba64_std_100c/infoswap \
    --ckpt_path  /mnt/data/zhonghuanlong/models/faceReclaim/last.ckpt \
    --gnet_weights /mnt/data/zhonghuanlong/models/faceReclaim/attribute/injection/weights/G_net_epoch_1000.pth \
    --face_weights /mnt/data/zhonghuanlong/models/faceReclaim/attribute/injection/weights/resnet18_110.pth \
    --ratio -5