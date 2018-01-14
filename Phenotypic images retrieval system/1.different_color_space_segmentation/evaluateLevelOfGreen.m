function [ res ] = evaluateLevelOfGreen( rgbPatch )
%EVALUATELEVELOFGREEN Summary of this function goes here
%   Detailed explanation goes here
%determines the green threshold in the hue channel
GREEN_RANGE = [65,170]/360;
INTENSITY_T = 0.1;
%converts to HSV color space
hsv = rgb2hsv(rgbPatch);
%generate a region of intereset (only areas which aren't black)
relevanceMask = rgb2gray(rgbPatch)>0;
%finds pixels within the specified range in the H and V channels
greenAreasMask = hsv(:,:,1)>GREEN_RANGE(1) & hsv(:,:,1) < GREEN_RANGE(2) & hsv(:,:,3) > INTENSITY_T;
%returns the mean in thie relevance mask
res = sum(greenAreasMask(:)) / sum(relevanceMask(:));
end