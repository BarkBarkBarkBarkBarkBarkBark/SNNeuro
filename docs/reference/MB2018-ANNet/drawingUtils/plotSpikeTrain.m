function [] = plotSpikeTrain(spikes,start,stop)
%PLOTSPIKETRAIN Summary of this function goes here
%   Detailed explanation goes here
C=myColors(spikes.nNeurons);
for n=1:spikes.nNeurons
    idx=((spikes.neuronTags==n)&spikes.timeStamps>=start&spikes.timeStamps<stop);
    t=spikes.timeStamps(idx);
    x=[t;t];
    plot([t;t],[zeros(size(t));ones(size(t))],'Color',C(n,:)); hold on;
end
end

