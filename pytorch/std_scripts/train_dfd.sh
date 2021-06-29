# Train on DFD indoor dataset

# Standard
export CUDA_VISIBLE_DEVICES=0
port=8001
display_id=100

# Training
NEPOCHS=3000

weightDecay=1e-4

# data augmentation: hflip vfkip scale color rotate 

NAME="dfd_train" # Change the name when training a new project
DATAROOT="../dfd_datasets/dfd_indoor/dfd_dataset_indoor_N2_8/"
BATCHSIZE=8
MTL_MODEL="eweights" # This project is adaptable for single and multitask objectives
MODEL="monotask"

TASKS='depth'
OUTPUTS_NC='1'

NET_ARCHITECTURE="D3net_multitask"

# Visualization
DISPLAY_ID=1
PORT=8001
IMAGE_SIZE=256
SCALE=1000.0
MAX_D=10.0
DISPLAY_FREQ=50

SAVE_CKPT_FREQ=100

python ./main.py --name $NAME --dataroot $DATAROOT --train --batchSize $BATCHSIZE --model $MODEL --mtl_method $MTL_MODEL --port $PORT --display_id $DISPLAY_ID --net_architecture $NET_ARCHITECTURE --pretrained --use_crop --use_padding --data_augmentation t f f f t --max_distance $MAX_D --scale_to_mm $SCALE --train --tasks $TASKS --outputs_nc $OUTPUTS_NC --use_skips --display_freq $DISPLAY_FREQ --lr 2e-4 --optim Adam --weightDecay $weightDecay --nThreads 1 --imageSize $IMAGE_SIZE --nEpochs $NEPOCHS --save_checkpoint_freq $SAVE_CKPT_FREQ
