const int nTranspoField = 10;
const	char *transpoField[10] = {
               
          "centers",
          "DVm",
          "twindow",
          "stepSize",
          "currentSpikes",
          "intermediateSpikes",
          "Prel",
          "Fd",
          "PrelLastComputation",
          "lastPreSpike"
          
};

typedef struct tag_transpo {

double* centers;
unsigned short nCenters;
double DVm;
unsigned short twindow;
unsigned short stepSize;
bool* currentSpikes;
bool* intermediateSpikes;
unsigned short nAfferents;
double *pRel;
double *Fd;
int *pRelLastComputation;
int *lastPreSpike;

} TRANSPO;

TRANSPO matlabToC_transpo_byIdx(const mxArray *matlabTranspo,int idx) {
	TRANSPO transpo;
	int i;
	mxArray *field;

	for(i=0; i<nTranspoField; i++) {
		field = mxGetField(matlabTranspo,idx,transpoField[i]);

		if(field==NULL) { /* missing param field */
			mexPrintf("Missing parameter field:");
			mexErrMsgIdAndTxt("missingParameterField",transpoField[i]);
		}

		switch(i) {
			case 0:
				transpo.centers  = mxGetPr(field);
                transpo.nCenters  = mxGetN(field);
				break;
			case 1:
				transpo.DVm = mxGetScalar(field);
				break;
            case 2:
				transpo.twindow = mxGetScalar(field);
				break;
            case 3:
				transpo.stepSize = mxGetScalar(field);
				break;
            case 4:
				transpo.currentSpikes = (bool*)mxGetPr(field);
                transpo.nAfferents = mxGetNumberOfElements(field);
				break;
            case 5:
				transpo.intermediateSpikes =(bool*) mxGetPr(field);
				break;
            case 6:
				transpo.pRel = mxGetPr(field);
				break;
            case 7:
				transpo.Fd = mxGetPr(field);
				break;
            case 8:
				transpo.pRelLastComputation = (int*)mxGetPr(field);
				break;
            case 9:
				transpo.lastPreSpike =(int*) mxGetPr(field);
				break;
		} 
	}
	return transpo;
}

TRANSPO* matlabToC_transpos(mxArray *matlabTranspo) {
	int nTranspo = (int) mxGetN(matlabTranspo);
	TRANSPO* transpos = mxMalloc(nTranspo*sizeof(TRANSPO));
	int i;
	for(i=0; i<nTranspo; i++)
		transpos[i] = matlabToC_transpo_byIdx(matlabTranspo,i);

	return transpos;
}
