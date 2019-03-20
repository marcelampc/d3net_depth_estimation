# In code choose which tasks
# select method: gradnorm, uncertainty, etc...

# Standard
export CUDA_VISIBLE_DEVICES=$1
port=$2
display_id=$3

# Dataset
dataset='datasets_KITTI'
dataset_name='kitti'

# Training
begin=1
step=300 # also step to checkpoint
end=3000
# lr=2e-4 # parm change
net_architecture="D3net_multitask" # parm change, use SegNet
# net_architecture='segnet'
weightDecay=0
# init_method="normal" # use standard
batch_size=4
mask_thres=0.0
n_classes=14
imageSize=256

tasks='depth'
outputs_nc='1'

# tasks='depth'
# outputs_nc='1'

# Validation
val_freq=40000 # iterations

# Visualization and metric scale
scale=255.996094
max_d=20.0 # for visualization
alpha=1.5

# MTL
mtl_method="eweights"
# losses for each task are pre-defined in the paper
# model="multitask" # later add some new variable to get which tasks will be used for training
model="monotask"

kitti_split='rob'  # right image only

# data augmentation: hflip vfkip scale color rotate

name=$mtl_method"_d_"$model"_"$dataset_name"_"$net_architecture"_TFFFF_B"$batch_size"_"$imageSize"x"$imageSize

python ./main.py --name $name --dataroot ./datasets/$dataset --dataset_name $dataset_name --mtl_method $mtl_method --batchSize $batch_size --imageSize $imageSize --nEpochs $end --save_checkpoint_freq $step --model $model --display_id $display_id --port $port --net_architecture $net_architecture --val_freq $val_freq --val_split test --use_resize --use_dropout --pretrained --data_augmentation t f f f f --max_distance $max_d --scale_to_mm $scale --n_classes $n_classes --train --tasks $tasks --alpha $alpha --outputs_nc $outputs_nc --use_skips --kitti_split $kitti_split #--validate
