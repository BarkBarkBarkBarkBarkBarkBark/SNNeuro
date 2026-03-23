typedef struct tag_inputSpikes {
    unsigned short* afferents;
    int* timeStamps;
    unsigned int nSpikes;
    unsigned short nAfferents;
    int *lastPreSpike;
}INPUTSPIKES;

INPUTSPIKES matlabToC_inputSpikes(const mxArray *matlabInputSpikes) {
    INPUTSPIKES inputSpikes;
    
    inputSpikes.lastPreSpike=(int *)mxGetPr(mxGetField(matlabInputSpikes,0,"lastPreSpike"));
    inputSpikes.nAfferents=(unsigned short)mxGetN(mxGetField(matlabInputSpikes,0,"lastPreSpike"));
    inputSpikes.afferents=(unsigned short *) mxGetPr(mxGetField(matlabInputSpikes,0,"afferents"));
    inputSpikes.timeStamps=( int *) mxGetPr(mxGetField(matlabInputSpikes,0,"timeStamps"));
    inputSpikes.nSpikes=(unsigned int) mxGetN(mxGetField(matlabInputSpikes,0,"timeStamps"));
    
    return inputSpikes;
}