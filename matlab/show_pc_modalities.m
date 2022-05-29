function [] = show_pc_modalities(pc)
%SHOW_PC Summary of this function goes here
%   Detailed explanation goes here

tmp = pc;
tmp.Color = uint8([pc.Intensity pc.Intensity pc.Intensity]);
pcshowpair(pc, tmp, 'VerticalAxis', 'Y', 'VerticalAxisDir', 'Down');

