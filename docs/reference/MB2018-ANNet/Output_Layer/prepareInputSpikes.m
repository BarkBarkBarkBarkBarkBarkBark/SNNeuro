function [ INPUT_SPIKES ] = prepareInputSpikes( spikes,samplingFrequency ,delay,nbDelays)
%PREPAREINPUTSPIKES Summary of this function goes here
%   Detailed explanation goes here

    times=spikes.timeStamps;
    tags=spikes.neuronTags;
    spikes.neuronTags=[spikes.neuronTags;ones(1,size(tags,2))];
    for i=1:nbDelays
        spikes.timeStamps=[spikes.timeStamps times+i*delay];
        spikes.neuronTags=[spikes.neuronTags [tags;(i+1)*ones(1,size(tags,2))]];
    end
    
    spikes.neuronTags=(spikes.neuronTags(1,:)-1)*(nbDelays+1)+spikes.neuronTags(2,:);
    
    [times,idx]=sort(spikes.timeStamps);
    spikes.timeStamps=times;
    spikes.neuronTags=spikes.neuronTags(idx);

    nAff=spikes.nNeurons*(nbDelays+1);
    INPUT_SPIKES.lastPreSpike=(-1*ones(1,nAff));
    INPUT_SPIKES.timeStamps=(spikes.timeStamps*samplingFrequency);
    INPUT_SPIKES.afferents=uint16(spikes.neuronTags-1);
end

