# In code choose which tasks
# select method: gradnorm, uncertainty, etc...

# Standard
export CUDA_VISIBLE_DEVICES=0
port=8001
display_id=1

# Dataset
dataset='nyu_dataset'
dataset_name='nyu'

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

# Validation
val_freq=40000 # iterations

# Visualization and metric scale
scale=10.0
max_d=10.0 # for visualization

# MTL
mtl_method="eweights"
# losses for each task are pre-defined in the paper
# model="multitask" # later add some new variable to get which tasks will be used for training
model="monotask"

# data augmentation: hflip vfkip scale color rotate

name=nyu_test

python ./main.py --name $name --dataroot ./datasets/std_datasets/$dataset --dataset_name $dataset_name --mtl_method $mtl_method --batchSize $batch_size --imageSize $imageSize --nEpochs $end --save_checkpoint_freq $step --model $model --display_id $display_id --port $port --net_architecture $net_architecture --val_freq $val_freq --val_split test --use_resize --use_dropout --pretrained --data_augmentation t f f f f --max_distance $max_d --scale_to_mm $scale --train --tasks $tasks --outputs_nc $outputs_nc --use_skips --train_split test
