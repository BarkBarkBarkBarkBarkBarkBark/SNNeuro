function neuron = createNewDNNeuron(SIMU,nSteps)

    neuron.nextFiring = int32(-1);
    neuron.firingTime = int32(zeros(1,ceil(nSteps*0.1)));
    neuron.nFiring = uint32(0);

    neuron.threshold=SIMU.DN_threshold;
    neuron.resetPotential=SIMU.DN_resetPotentialFactor*neuron.threshold;
    neuron.lastComputedPotential=0;
    neuron.lastComputationTime=int32(0);
    
    %for potential and weight history
    neuron.PotHist = zeros(1,ceil(SIMU.PotHistoryDuration/SIMU.PotHistoryStep)+1);

end
