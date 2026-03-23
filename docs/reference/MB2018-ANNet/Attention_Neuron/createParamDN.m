function [ PARAM_DN ] = createParamDN( SIMU )
%CREATEPARAMLAYER1 Summary of this function goes here
%   Detailed explanation goes here

    %simu times
    PARAM_DN.PotHistDuration=int32(SIMU.PotHistoryDuration);
    PARAM_DN.PotHistStep=int32(SIMU.PotHistoryStep);
    
    %stdp
    
    PARAM_DN.std_t=SIMU.DN_std_t;
    PARAM_DN.std_fd=SIMU.DN_std_fd;
    
    
    %neuron
    PARAM_DN.tm = SIMU.DN_tm;
    
    %************************
    %* computation tabs     *
    %************************
    
    times=0:(PARAM_DN.tm*20);
    PARAM_DN.expTm=exp(-times/PARAM_DN.tm);
    times=0:(PARAM_DN.std_t*20);
    PARAM_DN.expTd=exp(-times/PARAM_DN.std_t);
    
end

