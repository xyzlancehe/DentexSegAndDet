# some command snippets for setup, training, etc.

##############################################
# build MultiScaleDeformableAttention
cd models/dino/ops
export TORCH_CUDA_ARCH_LIST="7.5;8.6"   # 7.5 for grand-challenge's online GPU (T4), 8.6 for 3090
python setup.py install



##############################################
# train diffusion det quadrant detection
python train_diffdet.py \
    --output-dir output_diffdet_quadrant \
    --config-file configs/diffdet/diffdet.dentex.swinbase.quadrant.yaml \
    MODEL.WEIGHTS checkpoints/swin_base_patch4_window7_224_22k.pkl



##############################################
# train dino res50 enumeration32 detection
python train_dino.py \
	--output_dir output_dino_res50_enum32 -c configs/dino/DINO_4scale_cls32.py --coco_path dentex_dataset/coco/enumeration32 \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0 \
	--pretrain_model_path checkpoints/dino_pretrained_checkpoint0033_4scale.pth --finetune_ignore label_enc.weight class_embed

# train dino swin enumeration32 detection
python train_dino.py \
    --output_dir output_dino_swin_enum32 -c configs/dino/DINO_4scale_swin_cls32.py --coco_path dentex_dataset/coco/enumeration32 \
    --options dn_scalar=100 embed_init_tgt=TRUE \
    dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
    dn_box_noise_scale=1.0 \
    --pretrain_model_path checkpoints/dino_pretrained_checkpoint0029_4scale_swin.pth --finetune_ignore label_enc.weight class_embed

# transfer dino swin enumeration32 detection to train disease detection
python train_dino.py \
	--output_dir output_dino_swin_disease -c config/DINO/DINO_4scale_swin_cls4.py --coco_path dentex_dataset/coco/disease \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0 \
	--pretrain_model_path output_dino_swin_enum32/checkpoint0020.pth --finetune_ignore label_enc.weight class_embed

# continue training on full disease dataset
python train_dino.py \
	--output_dir output_swin_disease_all -c config/DINO/DINO_4scale_swin_cls4.py --coco_path dentex_dataset/coco/disease_all \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0 \
	--resume output_dino_swin_disease/checkpoint0017.pth


##############################################
# train yolov8 disease detection
python train_yolo.py    # modify configs in train_yolo.py and config/yolo/

##############################################
# train unet segmentation
python train_unet.py \
    --output_dir output_unet_enum32_$(date +%m-%d_%H-%M) \
    --dataset_dir dentex_dataset/segmentation/enumeration32 \
    --num_classes 32 --model seunet

python train_unet.py \
    --output_dir output_unet_enum9_$(date +%m-%d_%H-%M) \
    --dataset_dir dentex_dataset/segmentation/enumeration9 \
    --num_classes 9 --model seunet