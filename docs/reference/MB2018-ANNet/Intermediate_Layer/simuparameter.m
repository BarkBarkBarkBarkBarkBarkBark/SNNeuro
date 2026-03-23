%intermediate layer parameters

global SIMU

%history parameters
SIMU.PotHistDuration=5/signal.dt;
SIMU.PotHistStep=1;
SIMU.WHistDuration=numel(signal.data);
SIMU.WHistStep=SIMU.WHistDuration/20;

SIMU.freezeSTDP=false;

%encoding parameters
SIMU.TR_overlap=10;
SIMU.TR_DVmF=1.75;
SIMU.TR_stepSize=4;
SIMU.TR_winSize=floor(0.5e-3/(signal.dt*SIMU.TR_stepSize));

%Intermediate layer

%structure
SIMU.L1_nNeurons=60;

%neuron param
k=3;
SIMU.L1_DNWeight=SIMU.TR_overlap*k*1.5*signal.nSignals;
SIMU.L1_tm = 0.5*SIMU.TR_stepSize;
%SIMU.L1_tm = SIMU.TR_winSize/4;
SIMU.L1_initWmin=(SIMU.TR_winSize-2*k)/SIMU.TR_winSize;
SIMU.L1_initWmax=1;
SIMU.L1_refractoryPeriod=1;
SIMU.L1_threshold=(SIMU.L1_DNWeight+SIMU.TR_overlap*(SIMU.TR_winSize-k)*signal.nSignals)*(1-exp(-SIMU.TR_stepSize/SIMU.L1_tm))/(1-exp(-1/SIMU.L1_tm));
SIMU.L1_wiFactor=0;%Wi=WiFactor*Th
SIMU.L1_resetPotentialFactor=0;%resetPot=Th*resetPotFactor


%stdp
SIMU.L1_stdp_t= SIMU.TR_stepSize+0.5;
SIMU.L1_stdp_a=0.005;
SIMU.L1_stdp_post=-0.55*SIMU.L1_stdp_a;



