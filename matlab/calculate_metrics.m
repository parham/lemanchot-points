
clear;
clc;

%% Loading Data

dataDir = '/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/results/aligned_pcs/colored_icp';

thermalFiles = dir(fullfile(dataDir, '*_thermal_*.ply'));
visibleFiles = dir(fullfile(dataDir, '*_visible_*.ply'));

