function [ PARAM_L2 ] = createParamLayer2( SIMU )
%CREATEPARAMLAYER2 Summary of this function goes here
%   Detailed explanation goes here

    %simu times
    PARAM_L2.ThHistDuration=(SIMU.ThHistDuration);
    PARAM_L2.ThHistStep=(SIMU.ThHistStep);
    PARAM_L2.PotHistDuration=int32(SIMU.PotHistDuration);
    PARAM_L2.PotHistStep=(SIMU.PotHistStep);
    PARAM_L2.WHistDuration=int32(SIMU.WHistDuration);
    PARAM_L2.WHistStep=(SIMU.WHistStep);
    
    %wta
    PARAM_L2.wta_max=SIMU.L2_WTA_max;

    %neuron
	PARAM_L2.tm = SIMU.L2_tm;
    PARAM_L2.refractoryPeriod=((SIMU.L2_refractoryPeriod));
    PARAM_L2.resetPotentialFactor=SIMU.L2_resetPotentialFactor;
    PARAM_L2.wiFactor=SIMU.L2_wiFactor;
    PARAM_L2.inhibitoryPeriod=SIMU.L2_inhibitoryPeriod;
    
    %DN
    PARAM_L2.DNWeight=SIMU.L2_DNWeight;
    
    %stdp
    
    PARAM_L2.freezeSTDP=SIMU.freezeSTDP;
    
    PARAM_L2.stdp_t_pos = ((SIMU.L2_stdp_t_pos));
    PARAM_L2.stdp_t_neg = ((SIMU.L2_stdp_t_neg));
    PARAM_L2.stdp_t_lat_pos = ((SIMU.L2_stdp_t_lat_pos));
    PARAM_L2.stdp_t_lat_neg = ((SIMU.L2_stdp_t_lat_neg));
    PARAM_L2.stdp_a_pair = SIMU.L2_stdp_a;
    PARAM_L2.stdp_a_post = SIMU.L2_stdp_post;
    PARAM_L2.stdp_a_lat = SIMU.L2_stdp_lat;
    PARAM_L2.stdp_a_pre = SIMU.L2_stdp_pre;
    PARAM_L2.stdp_a_pair_neg = SIMU.L2_stdp_a_neg;
    PARAM_L2.stdp_a_post_neg = SIMU.L2_stdp_post_neg;
    PARAM_L2.stdp_a_lat_neg = SIMU.L2_stdp_lat_neg;
    PARAM_L2.stdp_a_pre_neg = SIMU.L2_stdp_pre_neg;
    
    PARAM_L2.WnegFactor=SIMU.L2_WnegFactor;
  
    PARAM_L2.ip_t_pos=((SIMU.L2_ip_t_pos));
    PARAM_L2.ip_t_neg=((SIMU.L2_ip_t_neg));
    PARAM_L2.ip_dth_pre=SIMU.L2_ip_dth_pre;
    PARAM_L2.ip_dth_post=SIMU.L2_ip_dth_post;
    PARAM_L2.maxTh=SIMU.L2_maxTh;
    PARAM_L2.minTh=SIMU.L2_minTh;
    

end

