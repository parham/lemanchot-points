function [pc, pos, colors, thermals] = convert_rgbdt_to_pc(data)
%CONVERT_RGBDT_TO_PC Converts RGBDT data to point cloud object

    % Size of data
    [height, width, ~] = size(data);

    pos = data(:,:,1:3);
    colors = data(:,:,4:6);
    thermal = data(:,:,7);

    pos = reshape(pos, (height * width), 3);
    colors = uint8(reshape(colors, (height * width), 3));
    thermals = reshape(thermal, (height * width), 1);

    pc = pointCloud(pos, 'Color', colors, 'Intensity', thermals);
    pc.Normal = pcnormals(pc);

end