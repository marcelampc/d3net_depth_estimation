# In code choose which tasks
# Test NYU Dataset for depth

# Standard
export CUDA_VISIBLE_DEVICES=0
name=$2

# Dataset
dataset='nyu_795'
dataset_name='nyu'

# Training
begin=1
step=30 # also step to checkpoint
end=3000
# lr=2e-4 # parm change
net_architecture="D3net_multitask" # parm change, use SegNet
weightDecay=0
# init_method="normal" # use standard
batch_size=8
mask_thres=0.0
n_classes=14
imageSize=256

tasks='depth semantics'

# Validation
val_freq=2000

# Visualization and metric scale
scale=1000.0
max_d=10.0

# MTL
mtl_method='gradnorm'
# losses for each task are pre-defined in the paper
# model="multitask" # later add some new variable to get which tasks will be used for training
model="multitask"

python ./main.py --name $name --dataroot ./datasets/$dataset --dataset_name $dataset_name --mtl_method $mtl_method --imageSize $imageSize --model $model --use_skips --test_split test --use_resize --use_dropout --scale_to_mm $scale --n_classes $n_classes --test --tasks $tasks --save_samples