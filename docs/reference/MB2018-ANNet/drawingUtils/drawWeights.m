function [] = drawWeights(W,nCols,nLines)
%DRAWWEIGHTS Summary of this function goes here
%   Detailed explanation goes here
figure;
nNeur=numel(W);
for i=1:nNeur
    a=subplot(nCols,nLines,i);
    h=customPColor(W{i});
    caxis([0,1]);
    colormap(gray);
    set(h, 'EdgeColor', 'none');
    set(a,'XDir','reverse');
    axis off;
end
end

