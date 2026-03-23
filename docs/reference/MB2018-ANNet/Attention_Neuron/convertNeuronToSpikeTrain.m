function [ spikes ] = convertNeuronToSpikeTrain( neurons,simuStep)
%CONVERTNEURONTOSPIKELIST Summary of this function goes here
%   Detailed explanation goes here

spikes.timeStamps=[];
spikes.neuronTags=[];

spikes.nNeurons=length(neurons);
for i=1:length(neurons)
    l=min(neurons(i).nFiring,length(neurons(i).firingTime));
    firing=neurons(i).firingTime(1:l);
    spikes.timeStamps=[spikes.timeStamps, firing];
    spikes.neuronTags=[spikes.neuronTags, i*ones(1,l)];
end

[times,idx]=sort(spikes.timeStamps);
spikes.timeStamps=times;
spikes.neuronTags=spikes.neuronTags(idx);

spikes.timeStamps=spikes.timeStamps+1;
spikes.timeStamps=double(spikes.timeStamps)*simuStep;

end

