function [res] = postprocessing_pc(pc)
%POSTPROCESSING_PC Summary of this function goes here
%   Detailed explanation goes here

z = pc.Location(:,3);
valid_indices = find(z);

pos = pc.Location(valid_indices,:);
colors = pc.Color(valid_indices,:);
intensity = pc.Intensity(valid_indices,:);

res = pointCloud(pos, 'Color', colors, 'Intensity', intensity);
res.Normal = pcnormals(res);

end

