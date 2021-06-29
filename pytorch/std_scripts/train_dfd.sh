# Train on DFD indoor dataset

# Standard
export CUDA_VISIBLE_DEVICES=0
port=8001
display_id=100

# Dataset
dataset='Cityscapes'
dataset_name='cityscapes'

# Training
begin=1
step=10 # also step to checkpoint
end=3000
# lr=2e-4 # parm change
# net_architecture="resupernet" # parm change, use SegNe
net_architecture="D3net_multitask" # parm change, use SegNett
# net_architecture='segnet'
weightDecay=1e-4
# init_method="normal" # use standard
batch_size=8
mask_thres=0.0
n_classes=19
imageSize=256

# tasks='semantics regression instance'
# outputs_nc='19 1 2'
# tasks='regression'
# outputs_nc='1'
tasks='semantics'
outputs_nc='19'
# tasks='instance'
# outputs_nc='2'

# tasks='depth'
# outputs_nc='1'

# Validation
val_freq=10000 #000 #0000 # iterations

# Visualization and metric scale
scale=1000.0
max_d=40.0
alpha=0.5

# MTL
# mtl_method="mgda"
mtl_method="eweights"
# losses for each task are pre-defined in the paper
# model="multitask" # later add some new variable to get which tasks will be used for training
model="multitask"

# data augmentation: hflip vfkip scale color rotate 

NAME="dfd_train" # Change the name when training a new project
DATAROOT="../../dfd_datasets/dfd_indoor/dfd_dataset_indoor_N2_8/train"


python ./main.py --name $NAME --dataroot $DATAROOT --train

# python ./main.py --name $name --dataroot ./datasets/$dataset --dataset_name $dataset_name --mtl_method $mtl_method --batchSize $batch_size --imageSize 512 $imageSize --nEpochs $end --save_checkpoint_freq $step --model $model --display_id $display_id --port $port --net_architecture $net_architecture --val_freq $val_freq --val_split val --use_resize --use_dropout --pretrained --data_augmentation t f f f t --max_distance $max_d --scale_to_mm $scale --n_classes $n_classes --train --tasks $tasks --alpha $alpha --outputs_nc $outputs_nc --use_skips --validate --display_freq 500 --lr 2e-4 --optim Adam --weightDecay $weightDecay --nThreads 1 #--inverse_depth
