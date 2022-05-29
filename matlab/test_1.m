
clear;
clc;

%% Loading RGBD&T data
data = load('/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/vtd/vtd_1626967976820.mat');
data = data.rgbdt;

%% Preprocessing Steps
data = preprocess_rgbdt(data, [1 3.5]);

%% Convert to Point Cloud
% pos : position array
% colors : color array
% thermals : thermal array
[pc, pos, colors, thermals] = convert_rgbdt_to_pc(data);

%% Postprocessing point cloud
pc_res = postprocessing_pc(pc);

%% Visualize Point Cloud
show_pc_modalities(pc_res)
figure; pcshow(pc_res)