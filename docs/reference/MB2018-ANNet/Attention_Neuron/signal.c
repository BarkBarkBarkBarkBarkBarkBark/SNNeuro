typedef struct tag_signal 
{
    double* data;
    unsigned int nData;
    unsigned short nSignals;
    int startStep;
} SIGNAL;

SIGNAL matlabToC_signal(const mxArray *matlabSignal) {
    SIGNAL signal;
    
    signal.data=mxGetPr(mxGetField(matlabSignal,0,"data"));
    signal.nSignals=(int) mxGetScalar(mxGetField(matlabSignal,0,"nSignals"));
    signal.nData=mxGetN(mxGetField(matlabSignal,0,"data"));
    signal.startStep=(int) (mxGetScalar(mxGetField(matlabSignal,0,"start"))/mxGetScalar(mxGetField(matlabSignal,0,"dt")));
    return signal;
}