function [ PARAM_L1 ] = createParamLayer1( SIMU )
%CREATEPARAMLAYER1 Summary of this function goes here
%   Detailed explanation goes here

    %simu times
    PARAM_L1.WHistDuration=int32(SIMU.WHistDuration);
    PARAM_L1.WHistStep=int32(SIMU.WHistStep);
    PARAM_L1.PotHistDuration=int32(SIMU.PotHistDuration);
    PARAM_L1.PotHistStep=int32(SIMU.PotHistStep);
    
    %stdp
    
    PARAM_L1.freezeSTDP=SIMU.freezeSTDP;
    
    PARAM_L1.stdp_t_pos = int32(floor(SIMU.L1_stdp_t));
    PARAM_L1.stdp_t_neg = int32(0);
    PARAM_L1.stdp_a_pos = SIMU.L1_stdp_a;
    PARAM_L1.stdp_a_neg = 0;
    
    PARAM_L1.cte_ltd=SIMU.L1_stdp_post;
    
    
    %neuron
    PARAM_L1.tm = SIMU.L1_tm;
    
    %************************
    %* computation tabs     *
    %************************
    
    times=0:(PARAM_L1.tm*20);
    PARAM_L1.expTm=exp(-times/PARAM_L1.tm);
    
end

