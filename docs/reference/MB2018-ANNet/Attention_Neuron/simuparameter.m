%simulation parameters for the attention neuron
% the simulation step is signal.dt, used as time unit

global SIMU

%history parameters
SIMU.PotHistoryDuration=10/signal.dt;
SIMU.PotHistoryStep=1;

%encoding parameters
SIMU.TR_overlap=10;
SIMU.TR_DVmF=1.75;
SIMU.TR_stepSize=4; %time between two encoding delays
SIMU.TR_winSize=floor(0.5e-3/(signal.dt*SIMU.TR_stepSize)); %number of encoding delays


%attention neuron parameters
SIMU.DN_tm = 0.5*SIMU.TR_stepSize;
SIMU.DN_threshold=0.4252*SIMU.TR_overlap*SIMU.TR_winSize*signal.nSignals/(1-exp(-1/SIMU.DN_tm));
SIMU.DN_resetPotentialFactor=0.15*(1/exp(-1/SIMU.DN_tm) -1);
SIMU.DN_std_t=10*SIMU.TR_winSize*SIMU.TR_stepSize;
SIMU.DN_std_fd=1-exp(-6.5/SIMU.DN_std_t);
