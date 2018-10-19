% optical parameter of the simulated lens (in m)
px=6*1e-6*6; % x2 because images are subsampled 
N=2.5;
f=35*1e-3; %2.9*1e-3 for kinect
mode_='gaussian';

% minimal step of depth values in the depth map
step_depth=10/255;

% min value to filter dark images
min_mean = 20;

focus=1.7;

max_depth = 10.0; % Even though max_depth from xtion is 3.5

