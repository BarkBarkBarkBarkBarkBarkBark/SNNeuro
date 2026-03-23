function [] = plotPotHist(hist,step)
%PLOTPOTHIST Summary of this function goes here
%   Detailed explanation goes here

nNeur=size(hist,2);
nTimes=size(hist,1);
times=(1:nTimes)*step;

C=myColors(nNeur);
for i=1:nNeur
    plot(times',hist(:,i),'Color',C(i,:)); hold on;
end

end

