function [ INPUT_SPIKES ] = prepareInputSpikes( spikes,samplingFrequency)
%PREPAREINPUTSPIKES Summary of this function goes here
%   Detailed explanation goes here

    [times,idx]=sort(spikes.timeStamps);
    spikes.timeStamps=times;
    spikes.neuronTags=spikes.neuronTags(idx);

    nAff=spikes.nNeurons;
    INPUT_SPIKES.lastPreSpike=int32(-1*ones(1,nAff));
    INPUT_SPIKES.timeStamps=int32(spikes.timeStamps*samplingFrequency);
    INPUT_SPIKES.afferents=uint16(spikes.neuronTags-1);
end

