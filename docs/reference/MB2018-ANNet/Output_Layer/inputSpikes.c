typedef struct tag_inputSpikes {
    unsigned short* afferents;
    double* timeStamps;
    unsigned int nSpikes;
    unsigned short nAfferents;
    double *lastPreSpike;
}INPUTSPIKES;

INPUTSPIKES matlabToC_inputSpikes(const mxArray *matlabInputSpikes) {
    INPUTSPIKES inputSpikes;
    
    inputSpikes.lastPreSpike=mxGetPr(mxGetField(matlabInputSpikes,0,"lastPreSpike"));
    inputSpikes.nAfferents=(unsigned short)mxGetN(mxGetField(matlabInputSpikes,0,"lastPreSpike"));
    inputSpikes.afferents=(unsigned short *) mxGetPr(mxGetField(matlabInputSpikes,0,"afferents"));
    inputSpikes.timeStamps= mxGetPr(mxGetField(matlabInputSpikes,0,"timeStamps"));
    inputSpikes.nSpikes=(unsigned int) mxGetN(mxGetField(matlabInputSpikes,0,"timeStamps"));
    
    return inputSpikes;
}