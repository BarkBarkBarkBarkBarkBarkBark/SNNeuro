%file names
inputFile='testData.mat';
inputFolder='../data/';

AN_simu_name='ANtest';
AN_saveFolder='';

%load files
load([inputFolder inputFile]);

%prepare parameters
simuparameter;
PARAM_DN=createParamDN(SIMU);
SPIKETRANSPOSITION=prepareSpikeTransposition(SIMU,signal);
neuron = createNewDNNeuron(SIMU,size(signal.data,2));

%run simulation
start=signal.start/signal.dt;
stop=start+size(signal.data,2);
tic
[neuron]=DNFromSignal(neuron,struct(signal),SPIKETRANSPOSITION,PARAM_DN,start,stop);
toc
  
%save results
%spikeTrainD=convertNeuronToSpikeTrain(neuron,1/signal.dt,start,stop);
spikeTrainAN=convertNeuronToSpikeTrain(neuron,signal.dt);
ANPotHist=neuron.PotHist;

resultfile=[AN_saveFolder AN_simu_name '_on_' inputFile];
save(resultfile,'spikeTrainAN','SIMU','ANPotHist');
        


