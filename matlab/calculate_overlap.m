
clear;
clc;

imgDir = '/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20220525_concrete_vertical_2/visible/';
imgFiles = {
    'visible_1653490590865.png', ...
    'visible_1653490604908.png', ...
    'visible_1653490615087.png', ...
    'visible_1653490780947.png', ...
    'visible_1653490800869.png'
};

for i = 1:length(imgFiles)-1
    f1 = fullfile(imgDir, imgFiles{i});
    f2 = fullfile(imgDir, imgFiles{i + 1});
    img1 = imread(f1);
    img2 = imread(f2);
    mask1 = logical(ones(size(img1, [1 2])));
    mask2 = logical(ones(size(img2, [1 2])));
    img1 = rgb2gray(img1);
    img2 = rgb2gray(img2);
    % Default spatial referencing objects
    fixedRefObj = imref2d(size(img1));
    movingRefObj = imref2d(size(img2));

    % Detect SURF features
    fixedPoints = detectSURFFeatures(img1,'NumOctaves',3,'NumScaleLevels',5);
    movingPoints = detectSURFFeatures(img2,'NumOctaves',3,'NumScaleLevels',5);

    % Extract features
    [fixedFeatures,fixedValidPoints] = extractFeatures(img1,fixedPoints,'Upright',false);
    [movingFeatures,movingValidPoints] = extractFeatures(img2,movingPoints,'Upright',false);

    % Match features
    indexPairs = matchFeatures(fixedFeatures,movingFeatures,'MatchThreshold',50.000000,'MaxRatio',0.500000);
    fixedMatchedPoints = fixedValidPoints(indexPairs(:,1));
    movingMatchedPoints = movingValidPoints(indexPairs(:,2));

    % Apply transformation - Results may not be identical between runs because of the randomized nature of the algorithm
    tform = estimateGeometricTransform(movingMatchedPoints,fixedMatchedPoints,'similarity');

    timg2 = imwarp(img2, movingRefObj, tform, 'OutputView', fixedRefObj, 'SmoothEdges', true);
    mask2 = imwarp(mask2, movingRefObj, tform, 'OutputView', fixedRefObj, 'SmoothEdges', true);

    overlappingRate(i) = (sum(mask1 & mask2, 'all') / sum(mask1, 'all')) * 100.0;
end

fprintf('Overlapping : %f \n', mean(overlappingRate));
