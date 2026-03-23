#include "mex.h"
#include <math.h>
#include "param.c"
#include "inputSpikes.c"
#include <stdlib.h>
#include <time.h>

const int nNeuronField = 14;

const	char *neuronField[14] = {
		"weight", 
        "weightNeg",
   		"nextFiring",
		"firingTime",
		"nFiring",
        "lastComputedPotential",
        "lastComputationTime",
        "threshold",
        "lastPostSpike",
        "lastInhibition",
        "ThHist",
        "PotHist",
        "WHist",
        "WNegHist"
};

typedef struct tag_neuron {
        double  *weight;
        double  *weightNeg;
        
        double  *nextFiring;
        double  *firingTime;
        unsigned int  *nFiring;
        unsigned int     NFIRING;

        double  *lastComputedPotential;
        double  *lastComputationTime;
        
        double *threshold;

        double *lastPostSpike;
        double *lastInhibition;

        double *ThHist;
        double *PotHist;
        double **WHist;
        double **WNegHist;
        
} NEURON,*NEURONS;

NEURON matlabToC_neuron_byIdx(mxArray *matlabNeuron, int idx) {
	NEURON neuron;
	int i;
    int n,j;
	mxArray *field;

	for(i=0; i<nNeuronField; i++) {
		field = mxGetField(matlabNeuron,idx,neuronField[i]);

		if(field==NULL) { /* missing neuron field*/
			mexErrMsgIdAndTxt("missingNeuronField",neuronField[i]);
		}

		switch(i) {
			case 0:
				neuron.weight = mxGetPr(field);
				break; 
            case 1:
				neuron.weightNeg = mxGetPr(field);
				break;
			case 2:
				neuron.nextFiring = mxGetPr(field);
				break;
			case 3:
				neuron.firingTime = mxGetPr(field);
				neuron.NFIRING = (unsigned int) mxGetN(field);
				break;
			case 4:
				neuron.nFiring =(unsigned int *)  mxGetPr(field);
				break; 
            case 5:
				neuron.lastComputedPotential =  mxGetPr(field);
				break;
            case 6:
				neuron.lastComputationTime =  mxGetPr(field);
				break;
            case 7:
				neuron.threshold =  mxGetPr(field);
				break;
            case 8:
				neuron.lastPostSpike = mxGetPr(field);
				break;
            case 9:
				neuron.lastInhibition = mxGetPr(field);
				break;
            case 10:
				neuron.ThHist = mxGetPr(field);
				break;
            case 11:
				neuron.PotHist = mxGetPr(field);
				break;
            case 12:
                n=mxGetNumberOfElements(field);
                neuron.WHist=mxMalloc(n*sizeof(double*));
                for (j=0;j<n;j++)
                {
                    neuron.WHist[j] = mxGetPr(mxGetCell(field,j));
                }
				break;
            case 13:
                n=mxGetNumberOfElements(field);
                neuron.WNegHist=mxMalloc(n*sizeof(double*));
                for (j=0;j<n;j++)
                {
                    neuron.WNegHist[j] = mxGetPr(mxGetCell(field,j));
                }
				break;
		} 
	}
	return neuron;
}
NEURON matlabToC_neuron(mxArray *matlabNeuron) {
	return matlabToC_neuron_byIdx(matlabNeuron,0);
}

NEURONS matlabToC_neurons(mxArray *matlabNeuron) {
	int nNeuron = (int) mxGetN(matlabNeuron);
	NEURONS neurons = mxMalloc(nNeuron*sizeof(NEURON));
	int i;
	for(i=0; i<nNeuron; i++)
		neurons[i] = matlabToC_neuron_byIdx(matlabNeuron,i);

	return neurons;
}

double applyBoundaries(double x, double downLim,double upLim)
{
    if(x<downLim)
    {
        return downLim;
    }
    if(x>upLim)
    {
        return upLim;
    }
    return x;  
}


void updatePotential(NEURON *neuron, double time, double value ,bool absolute,PARAM *param)
{
    double dt=time-(*neuron->lastComputationTime);
        
    /*update potential*/
    
    if(!absolute&&dt<param->tm*20)
    {
        (*neuron->lastComputedPotential)=(*neuron->lastComputedPotential)*exp(-dt/param->tm)+value;
    }
    else
    {
        (*neuron->lastComputedPotential)=value;
    }
    (*neuron->lastComputationTime)=time;
    
    /*update next firing time*/
    if(mxIsInf(*neuron->threshold)||*(neuron->lastComputedPotential)<*neuron->threshold)
    {
        *(neuron->nextFiring)=-1;//no firing scheduled, no additional computation needed
    }
    else
    {
        double nextft=time;
        if(*neuron->lastPostSpike>=0&&*neuron->lastPostSpike+ param->refractoryPeriod>nextft)
        {
             nextft=*neuron->lastPostSpike+ param->refractoryPeriod;
        }
        if(*neuron->lastInhibition>=0&&*neuron->lastInhibition+ param->inhibitoryPeriod>time)
        {
             nextft=*neuron->lastInhibition+ param->inhibitoryPeriod;
        }
        
        if(nextft>time)
        {
            double pot=0;
            if((nextft-time)<param->tm*20)
            {
                pot=(*(neuron->lastComputedPotential)*exp(-(nextft-time)/param->tm));//potential at the end of refractory period
            }
            if(pot>=*neuron->threshold)
            {
                *(neuron->nextFiring)=nextft;//next spike at the end of refractory period
            }
            else
            {
                *(neuron->nextFiring) = -1;
            }
        }
        else
        {
            *(neuron->nextFiring)=time;
        }
    }
}

void integrate(NEURON *neuron,INPUTSPIKES* inputSpikes, double time, unsigned short afferent, PARAM *param) {

    double w;
    w=neuron->weight[afferent]-param->WnegFactor*neuron->weightNeg[afferent];
    
    /*STDP*/
    if(!param->freezeSTDP)
    { 
        neuron->weight[afferent]+=param->stdp_a_pre;//stdp
        neuron->weightNeg[afferent]+=param->stdp_a_pre_neg;
        if(*(neuron->lastPostSpike)>=0)
        {
            //stdp
            if((time-*(neuron->lastPostSpike))<=param->stdp_t_neg)
            {
                //apply STDP for paired post-pre
                neuron->weight[afferent]+=param->stdp_a_pair;//stdp_a_pair>0
                neuron->weightNeg[afferent]+=param->stdp_a_pair_neg;//stdp_a_pair_neg<0
            }
            if((time-*(neuron->lastInhibition))<=param->stdp_t_lat_neg)
            {
                //apply STDP for paired ihn-pre
                neuron->weight[afferent]+=param->stdp_a_lat;
                neuron->weightNeg[afferent]+=param->stdp_a_lat_neg;
            }
            //ip
            if((time-*(neuron->lastPostSpike))<=param->ip_t_neg)
            {
                *(neuron->threshold)+=param->ip_dth_pre*w;
                *(neuron->threshold)=applyBoundaries(*(neuron->threshold),param->minTh,param->maxTh);
            }
        }
        neuron->weightNeg[afferent]=applyBoundaries(neuron->weightNeg[afferent],0,1);
        neuron->weight[afferent]=applyBoundaries(neuron->weight[afferent],0,1);

    }
    
    /*update potential and next firing time*/
    updatePotential(neuron,time,w,false,param);
    
}


void fire(NEURON *neuron,INPUTSPIKES* inputSpikes, double time, PARAM *param) {
    
    /*STDP*/
    
    int aff=0;
    *(neuron->lastPostSpike)=time;
    if(!param->freezeSTDP)
    {
        
        for(aff=0;aff<inputSpikes->nAfferents;aff++)
        {
            // stdp
            neuron->weight[aff]+=param->stdp_a_post;//stdp_a_post<0
            neuron->weightNeg[aff]+=param->stdp_a_post_neg;//stdp_a_post_neg>0
            if(inputSpikes->lastPreSpike[aff]>=0&&(time-inputSpikes->lastPreSpike[aff])<=param->stdp_t_pos)
            {
                neuron->weight[aff]+=param->stdp_a_pair;//stdp_a_pair>0
                neuron->weightNeg[aff]+=param->stdp_a_pair_neg;//stdp_a_pair_neg<0
            }
            
            neuron->weight[aff]=applyBoundaries(neuron->weight[aff],0,1);
            neuron->weightNeg[aff]=applyBoundaries(neuron->weightNeg[aff],0,1);
            
            //ip
            if(inputSpikes->lastPreSpike[aff]>=0&&(time-inputSpikes->lastPreSpike[aff])<param->ip_t_pos)
            {
                double w;
                w=neuron->weight[aff]-param->WnegFactor*neuron->weightNeg[aff];
                *(neuron->threshold)+=param->ip_dth_pre*w;
            }
            
        }
        //ip
        *(neuron->threshold)+=param->ip_dth_post * (*(neuron->threshold));
        *(neuron->threshold)=applyBoundaries(*(neuron->threshold),param->minTh,param->maxTh);
        
    }
    
    /* postsynaptique spike count */
    *(neuron->nFiring) += 1;
    
    /* postsynaptique spike times */
    (*neuron).firingTime[((int)*(*neuron).nFiring-1)%(*neuron).NFIRING] = time;
    
    /* update Potential */
    updatePotential(neuron,time,param->resetPotentialFactor*(*(neuron->threshold)),true,param);
    
}

void applyInhibition(NEURON *postNeuron,INPUTSPIKES* inputSpikes,int preNeuronIdx, double time, PARAM *param)
{
    updatePotential(postNeuron,time,-(param->wiFactor)*(*postNeuron->threshold),true,param);
    
    //stdp
    if(!param->freezeSTDP)
    {
        int aff;
        for(aff=0;aff<inputSpikes->nAfferents;aff++)
        {
            if((time-(inputSpikes->lastPreSpike[aff]))<=param->stdp_t_lat_pos)
            {
                //apply STDP for paired ihn-pre
                postNeuron->weight[aff]+=param->stdp_a_lat;
                postNeuron->weightNeg[aff]+=param->stdp_a_lat_neg;
                postNeuron->weight[aff]=applyBoundaries(postNeuron->weight[aff],0,1);
                postNeuron->weightNeg[aff]=applyBoundaries(postNeuron->weightNeg[aff],0,1);
            }
        }
    }
    *postNeuron->lastInhibition=time;
}

void
mexFunction(	int nlhs, mxArray *plhs[],
				int nrhs, const mxArray *prhs[] )
{
	if(nrhs==6) 
    {
        /* variables */
        mxArray *output = mxDuplicateArray(prhs[0]); 
        mxArray *outputInSpikes = mxDuplicateArray(prhs[1]); 
        NEURONS neuron = matlabToC_neurons(output);	
        INPUTSPIKES inputSpikes=matlabToC_inputSpikes(outputInSpikes);
        INPUTSPIKES dnSpikes=matlabToC_inputSpikes(prhs[2]);
        PARAM param = matlabToC_param(prhs[3]); 
        double start = mxGetScalar(prhs[4]);
        double stop = mxGetScalar(prhs[5]);
        double t, nextFiring = -1;
        int i, nNeuron = mxGetN(output), nextOneToFire;
        int s=0,dns=0;
        
        
		int wIdx=0;
		double nextWtime=0;
        int thIdx=0;
        double nextThtime=0;
        int potIdx=0;
        double nextPottime=0;
       
        mexPrintf("start\n");
        // affects output 
        plhs[0] = output;
        plhs[1] = outputInSpikes;
   

        while (s<inputSpikes.nSpikes&&inputSpikes.timeStamps[s]<start)
        {
            s++;
        }
        
        while (dns<dnSpikes.nSpikes&&dnSpikes.timeStamps[dns]<start)
        {
            dns++;
        }
        
        
        for(t=start;t<stop;t++)
        { // main loop 
            
            bool DNCancel=false;
            //presynaptic spikes from DN spiketrain
            while(dns<dnSpikes.nSpikes&&dnSpikes.timeStamps[dns]<=t)
            {
                DNCancel=true;
                dns++;
            }
            
            //presynaptic spikes from input spiketrain
            while(s<inputSpikes.nSpikes&&inputSpikes.timeStamps[s]<=t)
            {
                if(!DNCancel)
                {
                    for(i=0; i<nNeuron; i++)
                    {
                        integrate(&neuron[i],&inputSpikes,t,inputSpikes.afferents[s],&param);
                    }
                    inputSpikes.lastPreSpike[inputSpikes.afferents[s]]=t;
                }
                s++;
            }
            
            
            
            //compute winner
            nextOneToFire = -1;
            for(i=0; i<nNeuron; i++)
            {
                if(*(neuron[i].nextFiring)!=-1&&*(neuron[i].nextFiring)<=t)
                {
                    if(!param.wta_max)
                    {
                        nextOneToFire=i;
                        break;
                    }
                    else
                    {
                        updatePotential(&neuron[i],t,0,false,&param);
                        if(nextOneToFire==-1||*neuron[nextOneToFire].lastComputedPotential/(*neuron[nextOneToFire].threshold)<*neuron[i].lastComputedPotential/(*neuron[i].threshold))
                        {
                            nextOneToFire=i;
                        }
                    }
                }
                    
            }

            if(nextOneToFire!=-1)
            {
                // the winner fires
                fire(&neuron[nextOneToFire],&inputSpikes,t,&param);
                
                //inhibition
                
                for(i=0;i<nNeuron;i++)
                {
                    if(i!=nextOneToFire)
                    {
                        applyInhibition(&neuron[i],&inputSpikes,nextOneToFire,t,&param);
                    }
                }
                
            }
            
            //ThHist
			if (t>=nextThtime&&t<param.ThHistDuration)
			{
                int currentThIdx=(int)(t/param.ThHistStep);
                int i=0;
                
                while(thIdx<currentThIdx)
                {
                    if(thIdx>0)
                    {
                        for(i=0; i<nNeuron; i++)
                        {
                            neuron[i].ThHist[thIdx]=neuron[i].ThHist[thIdx-1];
                        }
                    }
                    thIdx++;
                }
                
				for(i=0; i<nNeuron; i++)
				{
					neuron[i].ThHist[thIdx]=*neuron[i].threshold;
				}
				thIdx++;
				nextThtime=thIdx*param.ThHistStep;
			}
            
            //PotHist
			if (t>=nextPottime&&t<param.PotHistDuration)
			{
                int currentPotIdx=(int)(t/param.PotHistStep);
                int i=0;
                
                while(potIdx<currentPotIdx)
                {
                    if(potIdx>0)
                    {
                        for(i=0; i<nNeuron; i++)
                        {
                            neuron[i].PotHist[potIdx]=neuron[i].PotHist[potIdx-1]*exp(-param.PotHistStep/param.tm);
                        }
                    }
                    else
                    {
                        for(i=0; i<nNeuron; i++)
                        {
                            neuron[i].PotHist[potIdx]=0;
                        }
                    }
                    potIdx++;
                }
				for(i=0; i<nNeuron; i++)
				{
                    updatePotential(&neuron[i],t,0,false,&param);
					neuron[i].PotHist[potIdx]=*neuron[i].lastComputedPotential;
				}
				potIdx++;
				nextPottime=potIdx*param.PotHistStep;
			}
            
            //WHist
            if (t>=nextWtime&&nextWtime<=param.WHistDuration)
            {
                int n=0;
                int aff=0;
                for(n=0; n<nNeuron; n++)
                {
                    for(aff=0;aff<inputSpikes.nAfferents;aff++)
                    {
                        neuron[n].WHist[wIdx][aff]=neuron[n].weight[aff];
                        neuron[n].WNegHist[wIdx][aff]=neuron[n].weightNeg[aff];
                    }

                }
                wIdx++;
                nextWtime=nextWtime+param.WHistStep;
            }

        } // main loop 
        
        mexPrintf("done\n");
        mxFree(neuron);
    }

}
