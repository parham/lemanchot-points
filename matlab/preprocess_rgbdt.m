function [res] = preprocess_rgbdt(data, depth_range)
%PREPROCESS_RGBDT Apply preprocessing steps to RGBD&T

depth = data(:,:,3);
% Filter the depth data with the determined range
depth(depth < depth_range(1)) = 0;
depth(depth > depth_range(2)) = 0;

res = data;
res(:,:,3) = depth;

end

