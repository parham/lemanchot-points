
clear;
clc;

%% Loading Data

mergeSize = 0.015;
depth_range = [1 2.5];

batch = {
    '/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/vtd/vtd_1626967963384.mat', ...
    '/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/vtd/vtd_1626967965865.mat', ...
    '/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/vtd/vtd_1626967973439.mat', ...
    '/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/vtd/vtd_1626967976820.mat'
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
    pdata = preprocess_rgbdt(data{index}, depth_range);
    % Convert to Point Cloud
    % pos : position array
    % colors : color array
    % thermals : thermal array
    [pc, pos, colors, thermals] = convert_rgbdt_to_pc(pdata);
    % Postprocessing point cloud
    pcs{index} = postprocessing_pc(pc);
end

%% Consecutive Point Cloud Registration 

resulted_pc = pcs{1};
accumTform = 0;
for index = 2:d_count
    moving = pcs{index};
    [tform, movingReg,rmse] = pcregisterndt(moving, resulted_pc, 0.1);
    disp(rmse)
    moving_tf = pctransform(movingReg, tform);
    resulted_pc = pcmerge(resulted_pc, moving_tf, mergeSize);
end

figure; pcshow(resulted_pc, 'MarkerSize', 10);
title('Registered Point Cloud');



