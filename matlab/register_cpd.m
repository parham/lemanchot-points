clear;
clc;

%% Loading Data

mergeSize = 0.015;
depth_range = [1 2.5];
gridSize = 0.01;
batch = {
    'C:/Users/SAPOZ/Documents/Lemenchot Fusion/vtd-20220604T171647Z-001/vtd/vtd_1625604430816.mat', ...
    'C:/Users/SAPOZ/Documents/Lemenchot Fusion/vtd-20220604T171647Z-001/vtd/vtd_1625604432756.mat', ...
    'C:/Users/SAPOZ/Documents/Lemenchot Fusion/vtd-20220604T171647Z-001/vtd/vtd_1625604434719.mat', ...
    'C:/Users/SAPOZ/Documents/Lemenchot Fusion/vtd-20220604T171647Z-001/vtd/vtd_1625604436427.mat', ...
    'C:/Users/SAPOZ/Documents/Lemenchot Fusion/vtd-20220604T171647Z-001/vtd/vtd_1625604438092.mat', ...
    'C:/Users/SAPOZ/Documents/Lemenchot Fusion/vtd-20220604T171647Z-001/vtd/vtd_1625604439961.mat', ...
    'C:/Users/SAPOZ/Documents/Lemenchot Fusion/vtd-20220604T171647Z-001/vtd/vtd_1625604448633.mat'
};

d_count = size(batch,2);

data = cell(d_count,1);
for index = 1:d_count
    d = load(batch{index});
    data{index} = d.rgbdt;
end

%% Processing Data

pcs = cell(d_count,1);
for index = 1:d_count
    % Preprocessing Steps
    pdata = preprocess_rgbdt(data{index},depth_range);
    % Convert to Point Cloud
    % pos : position array
    % colors : color array
    % thermals : thermal array
    [pc, pos, colors, thermals] = convert_rgbdt_to_pc(pdata);
    % Postprocessing point cloud
    pcs{index} = postprocessing_pc(pc);
end

%% Consecutive Point Cloud Registration 
resulted_pc =  pcdownsample(pcs{1}, 'gridAverage', gridSize);
for index = 2:d_count
    moving = pcdownsample(pcs{index},'gridAverage', gridSize);
    [tform, movingReg,rmse] = pcregistercpd(moving, resulted_pc);
    disp(rmse)
    moving_tf = pctransform(movingReg, tform);
    resulted_pc = pcmerge(resulted_pc, moving_tf, mergeSize);
%     figure; pcshow(moving); title('Original');
%     figure; pcshow(moving_tf); title('Transformed');
%     figure; pcshow(resulted_pc); title('Registered');
end

figure; pcshow(resulted_pc, 'MarkerSize', 20);
title('Registered Point Cloud');