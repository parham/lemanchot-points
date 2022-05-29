
clear;
clc;

%% Loading RGBD&T data
data = load('/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/vtd/vtd_1625604434719.mat');
data = data.rgbdt;

%% Preprocessing Steps
data = preprocess_rgbdt(data, [1.0 3.0]);

%% Convert to Point Cloud
% pos : position array
% colors : color array
% thermals : thermal array
[pc, pos, colors, thermals] = convert_rgbdt_to_pc(data);

%% Postprocessing point cloud
pc_res = postprocessing_pc(pc);

%% Visualize Point Cloud
show_pc_modalities(pc_res)