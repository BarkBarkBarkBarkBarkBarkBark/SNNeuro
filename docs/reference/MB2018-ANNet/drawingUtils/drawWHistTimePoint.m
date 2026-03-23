function [] = drawWHistTimePoint(WHist,timeIdx,nCols,nLines)
%DRAWWHISTTIMEPOINT Summary of this function goes here
%   Detailed explanation goes here
W=WHist(timeIdx,:);
drawWeights(W,nCols,nLines);
end

