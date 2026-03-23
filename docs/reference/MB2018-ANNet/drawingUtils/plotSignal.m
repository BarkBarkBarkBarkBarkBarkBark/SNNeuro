function [] = plotSignal(signal)
%PLOTSIGNAL Summary of this function goes here
%   Detailed explanation goes here
nSig=size(signal.data,1);
nTimes=size(signal.data,2);
times=(1:nTimes)*signal.dt-signal.dt;

C=myColors(nSig);
for i=1:nSig
    plot(times',signal.data(i,:),'Color',C(i,:)); hold on;
end
end

