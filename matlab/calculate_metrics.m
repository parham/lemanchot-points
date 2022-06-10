
clear;
clc;

%% Settings
dataset_name = 'concrete_vertical';
dataDir = '/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20220525_concrete_vertical/results/aligned_pcs';
metricsDir = '/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20220525_concrete_vertical/results/final_pcs';
methodNames = {
    'colored_icp', 'cpd', 'filterreg', ...
    'gmmtree', 'svr', 'ndt', 'phase_correlation', 'fmr'
};  

fprintf('%s is processing to provide metrics ...\n', dataset_name);
fprintf('Root Dir >> %s\n', dataDir);
metrics = struct();
for mindx = 1:length(methodNames)
    mname = methodNames{mindx};
    fprintf('Analyzing metrics for %s ...\n', mname);
    mfile = fullfile(metricsDir,sprintf('%s_metrics.csv',mname));
    if isfile(mfile)
        %% Loading metrics (from Python)
        m = readmatrix(mfile);
        metrics.(mname).('rmse') = m(:,1)';
        metrics.(mname).('fitness') = m(:,2)';
    end
    %% Calculating metrics
    metricDir = fullfile(dataDir, mname);
    if isfolder(metricDir)
        visibleFiles = dir(fullfile(metricDir, '*_visible_*.ply'));
        thermalFiles = dir(fullfile(metricDir, '*_thermal_*.ply'));
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

        resRowCount = 0;
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
            pcB.norm = pcs_viz{index+1}.Color;
            % Compute structural similarity values based on selected PARAMS
            [pointssim] = pc_ssim(pcA, pcB, PARAMS);
            ks = fieldnames(pointssim);
            for k=1:numel(ks)
                kmetrics = sprintf('pointssim_%s', ks{k});
                if ~isfield(metrics.(mname), kmetrics)
                    metrics.(mname).(kmetrics) = [];
                end
                metrics.(mname).(kmetrics)(end+1) = pointssim.(ks{k});
            end
        end
    end
end

%% Save the results
fprintf('Saving the metrics ...\n');
for mindx = 1:length(methodNames)
    mname = methodNames{mindx};
    resFile = fullfile(dataDir, sprintf('metrics_%s.csv', mname));
    fprintf('Saving %s metrics in %s \n', mname, resFile);
    res = [];
    if ~isfield(metrics,mname)
        continue;
    end
    mres = metrics.(mname);
    metricsNames = fieldnames(mres);
    colNum = numel(metricsNames);
    tbl = [];
    for k=1:colNum
        m = mres.(metricsNames{k});
        tbl = [tbl, m'];
    end
    csvwrite_with_headers(resFile,tbl,metricsNames)
end

%% Calculate Mean
for mindx = 1:length(methodNames)
    mname = methodNames{mindx};
    if ~isfield(metrics,mname)
        continue;
    end
    mres = metrics.(mname);
    metricsNames = fieldnames(mres);
    colNum = numel(metricsNames);
    for k=1:colNum
        m = mres.(metricsNames{k});
        fprintf('%s -- %s -- %f \n', mname, metricsNames{k}, mean(m));
    end
end
