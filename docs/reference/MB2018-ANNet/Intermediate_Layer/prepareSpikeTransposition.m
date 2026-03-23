function [ SPIKETRANSPOSITION ] = prepareSpikeTransposition( SIMU,signal )
%PREPARESPIKETRANSPOSITION Summary of this function goes here
%   Detailed explanation goes here

s_noise=median(abs(signal.data),2)/0.6745;

SPIKETRANSPOSITION=[];

for i=1:signal.nSignals
    SPIKETRANSPOSITION(i).overlap=SIMU.TR_overlap;
    SPIKETRANSPOSITION(i).DVm=SIMU.TR_DVmF*s_noise(i);
    
    SPIKETRANSPOSITION(i).twindow=uint16(SIMU.TR_winSize);
    SPIKETRANSPOSITION(i).stepSize=uint16(SIMU.TR_stepSize);
    
    minVal=min(min(signal.data));
    maxVal=max(max(signal.data));
    
    DVm=SPIKETRANSPOSITION(i).DVm;
    DVi=DVm*2/SIMU.TR_overlap;
    
    
    SPIKETRANSPOSITION(i).centers=(minVal-DVm):DVi:(maxVal+DVm);
    SPIKETRANSPOSITION(i).nAff=SPIKETRANSPOSITION(i).twindow*length(SPIKETRANSPOSITION(i).centers);
    SPIKETRANSPOSITION(i).currentSpikes=logical(zeros(length(SPIKETRANSPOSITION(i).centers),SPIKETRANSPOSITION(i).twindow));
    SPIKETRANSPOSITION(i).intermediateSpikes=logical(zeros(length(SPIKETRANSPOSITION(i).centers),SIMU.TR_stepSize*SPIKETRANSPOSITION(i).twindow));
    
    SPIKETRANSPOSITION(i).Prel=ones(length(SPIKETRANSPOSITION(i).centers),SPIKETRANSPOSITION(i).twindow);
    SPIKETRANSPOSITION(i).Fd=zeros(length(SPIKETRANSPOSITION(i).centers),SPIKETRANSPOSITION(i).twindow);
    SPIKETRANSPOSITION(i).PrelLastComputation=int32(zeros(length(SPIKETRANSPOSITION(i).centers),SPIKETRANSPOSITION(i).twindow));
    SPIKETRANSPOSITION(i).lastPreSpike=int32(-1*ones(length(SPIKETRANSPOSITION(i).centers),SPIKETRANSPOSITION(i).twindow));
    
end

end

