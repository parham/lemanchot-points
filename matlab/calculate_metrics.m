
clear;
clc;

%% Settings
dataDir = '/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/results/aligned_pcs/colored_icp';
metricsDir = '/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/results/final_pcs';
methodNames = {
    'colored_icp', 'cpd', 'filterreg', ...
    'gmmtree', 'svr', 'ndt'
};

fprintf('Root Dir >> %s', dataDir);

metrics = struct();
for mindx = 1:length(methodNames)
    mname = methodNames{mindx};
    fprintf('Analyzing metrics for %s ...\n', mname);
    mfile = fullfile(metricsDir,sprintf('%s_metrics.csv',mname));
    if ~isfile(mfile)
        continue
    end
    %% Loading metrics (from Python)
    m = readmatrix(mfile);
    metrics.(mname).('rmse') = m(:,1)';
    metrics.(mname).('fitness') = m(:,2)';
    %% Calculating metrics
    visibleFiles = dir(fullfile(dataDir, '*_visible_*.ply'));
    thermalFiles = dir(fullfile(dataDir, '*_thermal_*.ply'));
    pcs_viz = cell(length(visibleFiles),1);
    pcs_th = cell(length(thermalFiles),1);
    for index = 1:length(visibleFiles)
        vfile = fullfile(visibleFiles(index).folder, visibleFiles(index).name);
        tfile = fullfile(thermalFiles(index).folder, thermalFiles(index).name);
        pcs_viz{index} = pcread(vfile);
        pcs_viz{index}.Normal = pcnormals(pcs_viz{index});
        pcs_th{index} = pcread(tfile);
        pcs_th{index}.Normal = pcnormals(pcs_th{index});
    end
    % Configure PARAMS
    PARAMS.ATTRIBUTES.GEOM = true;
    PARAMS.ATTRIBUTES.NORM = true;
    PARAMS.ATTRIBUTES.CURV = false;
    PARAMS.ATTRIBUTES.COLOR = true;
    
    PARAMS.ESTIMATOR_TYPE = {'VAR'};
    PARAMS.POOLING_TYPE = {'Mean'};
    PARAMS.NEIGHBORHOOD_SIZE = 12;
    PARAMS.CONST = eps(1);
    PARAMS.REF = 0;
    % Analysis of visible point cloud
    metrics.(mname).('angular_sim_BA') = [];
    metrics.(mname).('angular_sim_AB') = [];
    metrics.(mname).('angular_sim_sym') = [];
    for index = 1:length(visibleFiles)-1
        % Angular Similarity metrics
        [asimBA, asimAB, asimSym] = pc_asim(pcs_viz{index}, pcs_viz{index+1}, 'Mean');
        metrics.(mname).('angular_sim_BA')(end+1) = asimBA;
        metrics.(mname).('angular_sim_AB')(end+1) = asimAB;
        metrics.(mname).('angular_sim_sym')(end+1) = asimSym;
        % Point SSIM
        pcA.geom = pcs_viz{index}.Location;
        pcA.color = pcs_viz{index}.Color;
        pcA.norm = pcs_viz{index}.Color;
        pcB.geom = pcs_viz{index+1}.Location;
        pcB.color = pcs_viz{index+1}.Color;

        % Compute structural similarity values based on selected PARAMS
        [pointssim] = pc_ssim(pcA, pcB, PARAMS);
        print('Point-SSIM')
    end
end



