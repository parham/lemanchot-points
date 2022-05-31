function [pos, colors, thermals, pc] = convert_to_pc(data)
%CONVERT_TO_PC convert RGBD&T data to multi-modal point cloud.
%   Input:
%       - data : multi-dimensional matrix presenting the multi-modal data
%   Output:
%       - result : X,Y,Z the positions

    sz = size(data);
    new_sz = [sz(1) * sz(2), sz(3)];
    pcdata = reshape(data, new_sz);
    
    pos = pcdata(:,[1,2,3]);
    colors = uint8(pcdata(:,[4,5,6]));
    thermals = pcdata(:,7);
    pc = pointCloud(pos,"Color",colors);
end

