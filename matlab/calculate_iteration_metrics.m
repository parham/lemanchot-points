
clear;
clc;

%% Settings
methodNames = {
    'colored_icp', 'cpd', 'filterreg', 'gmmtree', 'svr', 'ndt', 'phase_correlation', 'fmr'
}; 

dataDir = '/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/results/iteration_pcs';
tmp = dir(dataDir);
tmp = tmp(3:end);
iterationDir = {length(tmp),1};
for i = 1:length(tmp)
    if isfolder(fullfile(tmp(i).folder, tmp(i).name))
        iterationDir{i} = tmp(i).name;
    end
end

%%
metrics = struct();
for it = 1:length(iterationDir)
    iter = iterationDir{it};
    iterDir = fullfile(dataDir, iter);
    fprintf('iteration %d is processing to provide metrics ...\n', iter);
    fprintf('Root Dir >> %s\n', iterDir);
    for mindx = 1:length(methodNames)
        mname = methodNames{mindx};
        metrics.(mname).('iteration')(it) = str2num(iter);
        fprintf('Analyzing metrics for %s ...\n', mname);
        mfile = fullfile(iterDir,sprintf('%s_%s_metrics.csv',mname,iter));
        if isfile(mfile)
            %% Loading metrics (from Python)
            m = readmatrix(mfile);
            metrics.(mname).('rmse')(it) = m(1);
            metrics.(mname).('fitness')(it) = m(2);
        end
        %% Calculating metrics
        fviz_1 = fullfile(iterDir, sprintf('%s_visible_1.ply', mname));
        fviz_2 = fullfile(iterDir, sprintf('%s_visible_2.ply', mname));
        
        if ~(isfile(fviz_1) & isfile(fviz_1))
            continue;
        end

        pc1 = pcread(fviz_1);
        pc1.Normal = pcnormals(pc1);
        pc2 = pcread(fviz_2);
        pc2.Normal = pcnormals(pc2);
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
        % Angular Similarity metrics
        [asimBA, asimAB, asimSym] = pc_asim(pc1, pc2, 'Mean');
        metrics.(mname).('angular_sim_BA')(it) = asimBA;
        metrics.(mname).('angular_sim_AB')(it) = asimAB;
        metrics.(mname).('angular_sim_sym')(it) = asimSym;
        % Point SSIM
        pcA.geom = pc1.Location;
        pcA.color = pc1.Color;
        pcA.norm = pc1.Color;
        pcB.geom = pc2.Location;
        pcB.color = pc2.Color;
        pcB.norm = pc2.Color;
        % Compute structural similarity values based on selected PARAMS
        [pointssim] = pc_ssim(pcA, pcB, PARAMS);
        ks = fieldnames(pointssim);
        for k=1:numel(ks)
            kmetrics = sprintf('pointssim_%s', ks{k});
            metrics.(mname).(kmetrics)(it) = pointssim.(ks{k});
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




