global SIMU

%history parameters

SIMU.PotHistDuration=10/signal.dt;
SIMU.PotHistStep=1;
SIMU.ThHistDuration=size(signal.data,2);
SIMU.ThHistStep=1/signal.dt;
SIMU.WHistDuration=size(signal.data,2);
SIMU.WHistStep=SIMU.WHistDuration/20;

SIMU.freezeSTDP=false;


%OUTPUT LAYER
SIMU.L2_nNeurons=10;
SIMU.L2_WTA_max=true;

%delays
step=1/20000;
SIMU.L2_nbDelays=floor(2.5e-3/step);
SIMU.L2_delay=step;

%neuron param
SIMU.L2_tm = 3e-3/signal.dt;
SIMU.L2_WImin=0;
SIMU.L2_WImax=1;
SIMU.L2_WNegImin=0;
SIMU.L2_WNegImax=0;
SIMU.L2_WnegFactor=1;
SIMU.L2_refractoryPeriod=3e-3/signal.dt;
SIMU.L2_resetPotentialFactor=-10;

%inhibition
SIMU.L2_wiFactor=10;
SIMU.L2_inhibitoryPeriod=3e-3/signal.dt;

%DN
SIMU.L2_DNWeight=0;

%stdp
SIMU.L2_stdp_t_neg = 0.2e-3/signal.dt;
SIMU.L2_stdp_t_pos = 3e-3/signal.dt;
SIMU.L2_stdp_t_lat_pos = 3e-3/signal.dt;
SIMU.L2_stdp_t_lat_neg = 3e-3/signal.dt;
SIMU.L2_stdp_a = 0.05;
SIMU.L2_stdp_post=-0.7*SIMU.L2_stdp_a;
SIMU.L2_stdp_lat=-0*SIMU.L2_stdp_a;
SIMU.L2_stdp_pre=0*SIMU.L2_stdp_a;
SIMU.L2_stdp_a_neg = -0.05;
SIMU.L2_stdp_post_neg=-0.1*SIMU.L2_stdp_a_neg;
SIMU.L2_stdp_lat_neg = 0.000;
SIMU.L2_stdp_pre_neg=-0.0*SIMU.L2_stdp_lat_neg;

alpha=0.6;
SIMU.L2_ip_dth_post=-0.01;
SIMU.L2_ip_dth_pre=-alpha*SIMU.L2_ip_dth_post;
SIMU.L2_ip_t_neg=SIMU.L2_stdp_t_neg;
SIMU.L2_ip_t_pos=SIMU.L2_stdp_t_neg*alpha/(1-alpha);

SIMU.L2_minTh=8;
SIMU.L2_maxTh=0.35*SIMU.L2_minTh*SIMU.L2_nbDelays;
SIMU.L2_threshold =SIMU.L2_maxTh;