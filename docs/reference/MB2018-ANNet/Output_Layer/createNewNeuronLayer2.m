function neuron = createNewNeuronLayer2(SIMU,INPUT_SPIKES)
nAfferents=length(INPUT_SPIKES.lastPreSpike);
for i=1:SIMU.L2_nNeurons
    neuron(i).nextFiring = (-1);
    neuron(i).firingTime = (zeros(1,length(INPUT_SPIKES.timeStamps)));
    neuron(i).nFiring = uint32(0);
    neuron(i).weight = rand(nAfferents/(SIMU.L2_nbDelays+1),SIMU.L2_nbDelays+1)*(SIMU.L2_WImax-SIMU.L2_WImin)+SIMU.L2_WImin;
    neuron(i).weightNeg = rand(nAfferents/(SIMU.L2_nbDelays+1),SIMU.L2_nbDelays+1)*(SIMU.L2_WNegImax-SIMU.L2_WNegImin)+SIMU.L2_WNegImin;
    neuron(i).threshold=SIMU.L2_threshold;
    
    neuron(i).lastComputedPotential=0;
    neuron(i).lastComputationTime=(0);
   
    
    %for potential and weight history
    neuron(i).PotHist = zeros(1,ceil(SIMU.PotHistDuration/SIMU.PotHistStep)+1);
    neuron(i).ThHist = zeros(1,ceil(SIMU.ThHistDuration/SIMU.ThHistStep)+1);
    neuron(i).WHist=cell(ceil(SIMU.WHistDuration/SIMU.WHistStep),1);
    for t=1:(ceil(SIMU.WHistDuration/SIMU.WHistStep))
        neuron(i).WHist{t}=zeros(nAfferents/(SIMU.L2_nbDelays+1),SIMU.L2_nbDelays+1);
    end
    neuron(i).WNegHist=cell(ceil(SIMU.WHistDuration/SIMU.WHistStep),1);
    for t=1:(ceil(SIMU.WHistDuration/SIMU.WHistStep))
        neuron(i).WNegHist{t}=zeros(nAfferents/(SIMU.L2_nbDelays+1),SIMU.L2_nbDelays+1);
    end
    
    %utils for STDP
    neuron(i).lastPostSpike=(-1);
    neuron(i).lastInhibition=(-1);
end

end
