function [ DN_SPIKES ] = prepareDNSpikes( spikes,samplingFrequency)
%PREPAREINPUTSPIKES Summary of this function goes here
%   Detailed explanation goes here

    [times,idx]=sort(spikes.timeStamps);
    spikes.timeStamps=times;
    spikes.neuronTags=spikes.neuronTags(idx);
    
    nAff=spikes.nNeurons;
    DN_SPIKES.lastPreSpike=(-1*ones(1,nAff));
    DN_SPIKES.timeStamps=(spikes.timeStamps*samplingFrequency);
    DN_SPIKES.afferents=uint16(spikes.neuronTags-1);
end

