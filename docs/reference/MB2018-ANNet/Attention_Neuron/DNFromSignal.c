#include "mex.h"
#include <math.h>
#include "param.c"
#include "transpo.c"
#include "signal.c"
#include <stdlib.h>
#include <time.h>

const int nNeuronField = 8;

const	char *neuronField[8] = { 
   		"nextFiring",
		"firingTime",
		"nFiring",
        "lastComputedPotential",
        "lastComputationTime",
        "threshold",
        "resetPotential",
        "PotHist"
};

typedef struct tag_neuron {
        
        int  *nextFiring;
        int  *firingTime;
        unsigned int  *nFiring;
        unsigned int     NFIRING;

        double  *lastComputedPotential;
        int  *lastComputationTime;
        
        double *threshold;
        
        double *resetPotential;

        double *PotHist;
        
} NEURON,*NEURONS;

NEURON matlabToC_neuron_byIdx(mxArray *matlabNeuron, int idx) {
	NEURON neuron;
	int i;
	mxArray *field;

	for(i=0; i<nNeuronField; i++) {
		field = mxGetField(matlabNeuron,idx,neuronField[i]);

		if(field==NULL) { /* missing neuron field*/
			mexErrMsgIdAndTxt("missingNeuronField",neuronField[i]);
		}

		switch(i) {
			case 0:
				neuron.nextFiring =(int*) mxGetPr(field);
				break;
			case 1:
				neuron.firingTime =(int*) mxGetPr(field);
				neuron.NFIRING = (unsigned int) mxGetN(field);
				break;
			case 2:
				neuron.nFiring =  (unsigned int*)mxGetPr(field);
				break; 
            case 3:
				neuron.lastComputedPotential =  mxGetPr(field);
				break;
            case 4:
				neuron.lastComputationTime = (int*) mxGetPr(field);
				break;
            case 5:
				neuron.threshold =  mxGetPr(field);
				break;
            case 6:
				neuron.resetPotential = mxGetPr(field);
				break;
            case 7:
				neuron.PotHist = mxGetPr(field);
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
double maxArray(int n, double *x) {
	int i;
	double result = -mxGetInf();
	for(i=0; i<n; i++)
		if(x[i]>result)
			result = x[i];
	return result;
}
int i_round(double x) {
	return (int) (x+.5);
}

void updateAfferentSpikes(TRANSPO* transpo,SIGNAL* signal,int idx,int t)
{
    double value;
    int i,j;
    
    value=signal->data[signal->nSignals*(t-signal->startStep)+idx];

    for(i=transpo[idx].nAfferents*transpo[idx].stepSize-1;i>=transpo[idx].nCenters;i--)
    {
        transpo[idx].intermediateSpikes[i]=transpo[idx].intermediateSpikes[i-transpo[idx].nCenters];
    }
    
    for (i=0;i<transpo[idx].nCenters;i++)
    {
        if(value>transpo[idx].centers[i]-transpo[idx].DVm && value<=transpo[idx].centers[i]+transpo[idx].DVm)
        {
            transpo[idx].intermediateSpikes[i]=1;
        }
        else
        {
            transpo[idx].intermediateSpikes[i]=0;
        }
    }
    
    for(i=0;i<transpo[idx].twindow;i++)
    {
        for (j=0;j<transpo[idx].nCenters;j++)
        {
			transpo[idx].currentSpikes[i*transpo[idx].nCenters + j] = transpo[idx].intermediateSpikes[i*transpo[idx].nCenters*transpo[idx].stepSize + j];
        }
    }
}

void updatePotential(NEURON *neuron, int time, double value ,bool absolute,PARAM *param)
{
    int dt=time-(*neuron->lastComputationTime);
    
    /*update*/
    
    if(!absolute&&dt<param->tm*20)
    {
        (*neuron->lastComputedPotential)=(*neuron->lastComputedPotential)*param->expTm[dt]+value;
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
       *(neuron->nextFiring)=time; 
    }
}

void integrate(NEURON *neuron, int time,unsigned short sig ,unsigned short afferent, PARAM *param,TRANSPO *transpo) {

    double w;
    w=transpo[sig].pRel[afferent];

    /*update potential and next firing time*/
    updatePotential(neuron,time,w,false,param);
    
}


void fire(NEURON *neuron, int time, PARAM *param,TRANSPO *transpo,int nSig) {

	/* postsynaptique spike count */
	*(neuron->nFiring) += 1;

	/* postsynaptique spike times */
	(*neuron).firingTime[((int)*(*neuron).nFiring-1)%(*neuron).NFIRING] = time;

	/* update Potential */
    updatePotential(neuron,time,*(neuron->resetPotential),false,param);
   
}


void
mexFunction(	int nlhs, mxArray *plhs[],
				int nrhs, const mxArray *prhs[] )
{
	if(nrhs==6) 
    {
        /* variables */
        mxArray *output = mxDuplicateArray(prhs[0]); /* neurons */
        mxArray *outputTranspo = mxDuplicateArray(prhs[2]);
        NEURONS neuron = matlabToC_neurons(output);	/* get neurons from matlab structure */
        SIGNAL signal = matlabToC_signal(prhs[1]);/*signal*/
        TRANSPO* transpo=matlabToC_transpos(outputTranspo); /* transposition */
        PARAM param = matlabToC_param(prhs[3]); /* param */
        
        int t;
        unsigned short i;
        unsigned short nTranspos=mxGetN(prhs[2]);
        int start = (int)mxGetScalar(prhs[4]);
        int stop = (int)mxGetScalar(prhs[5]);

        int potIdx=0;
        int nextPottime=0;
        
         /* affects output */
        plhs[0] = output;
        plhs[1] = outputTranspo;
        
        
        //check time range
        if(start<signal.startStep||stop>signal.startStep+signal.nData)
        {
            mexErrMsgIdAndTxt("timeRangeError","outside data time range,%d-%d,%d-%d\n",start,stop,signal.startStep,signal.startStep+signal.nData);
        }
        
        // main loop (time)
        for (t=start;t<stop;t++)
        {
            int aff=0;
            int sig=0;
            
           //presynaptic spikes loop (integrate)
            for (sig=0;sig<signal.nSignals;sig++)
            {
                updateAfferentSpikes(transpo,&signal, sig,t);
                
                for(aff=0;aff<transpo[sig].nAfferents;aff++)
                {/*presynaptic spikes*/
                    int dt;
                    if(transpo[sig].currentSpikes[aff]==1)
                    {
                        /*Prel update 1st step*/
                        dt=t - transpo[sig].pRelLastComputation[aff];
                        if(dt<param.std_t*20)
                        {
                            transpo[sig].pRel[aff]=1-(1-transpo[sig].pRel[aff])*param.expTd[dt];
                        }
                        else
                        {
                            transpo[sig].pRel[aff]=1;
                        }
                        
                        transpo[sig].lastPreSpike[aff]=t;
                        
                        /* integrate this spike */
                        integrate(&neuron[0],t,sig,aff,&param,transpo);
              
                        
                        /*Prel update 2nd step*/
                        transpo[sig].pRel[aff]=transpo[sig].pRel[aff]*(1-param.std_fd);
                        transpo[sig].pRelLastComputation[aff]=t;
                    }
                }
            }

            //postsynaptic spikes loop
            if(*neuron[0].nextFiring == t)
            { /* next event is a postsynaptic spike */
                
                /* the winner fires */
                fire(&neuron[0],t,&param,transpo,signal.nSignals);
                
            }
            
            
            //PotHist
            if (t>=nextPottime&&t<param.PotHistDuration)
            {
                
                updatePotential(&neuron[0],t,0,false,&param);
                neuron[0].PotHist[potIdx]=*neuron[0].lastComputedPotential;

                potIdx++;
                nextPottime=nextPottime+param.PotHistStep;
            }
        }
        
        
        mxFree(neuron);
        mexPrintf("done\n");
    }
    
}
