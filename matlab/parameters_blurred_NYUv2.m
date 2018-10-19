% Parameters used to blur NYUv2 dataset for first experiment with different
% optical parameter of the simulated lens (in m)
px=2.8*1e-6*2; % x2 because images are subsampled
N=2.8;
f=15*1e-3; %2.9*1e-3 pour kinect
dmode='gaussian'; % gaussian or disk

% minimal step of depth values in the depth map
step_depth=10/255;

max_depth = 10.0; % Even though max_depth from xtion is 3.5

% focus in [2,4,8] for the different experiments
focus=2;