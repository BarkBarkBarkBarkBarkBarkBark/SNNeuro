function [] = drawWeightsHistNeurEvo(WHist,neurIdx,nCols,nLines)
%DRAWWHISTTIMEPOINT Summary of this function goes here
%   Detailed explanation goes here
W=WHist(:,neurIdx);
drawWeights(W,nCols,nLines);
end

