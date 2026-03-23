const int nParamField = 7;
const	char *paramField[7] = {
          "tm",
          "std_t",
          "std_fd",
          "expTm",
          "expTd",
          "PotHistDuration",
		  "PotHistStep"
};

typedef struct tag_param {

double tm;
double std_t;
double std_fd;

double *expTm;
double *expTd;

int PotHistDuration;
int PotHistStep;

} PARAM;

PARAM matlabToC_param(const mxArray *matlabParam) {
	PARAM param;
	int i;
	mxArray *field;

	for(i=0; i<nParamField; i++) {
		field = mxGetField(matlabParam,0,paramField[i]);

		if(field==NULL) { /* missing param field */
			mexErrMsgIdAndTxt("missingParamField",paramField[i]);
		}

		switch(i) {
            case 0:
				param.tm = mxGetScalar(field);
				break;
            case 1:
				param.std_t = mxGetScalar(field);
				break;
            case 2:
				param.std_fd = mxGetScalar(field);
				break;
            case 3:
				param.expTm = mxGetPr(field);
				break;
            case 4:
				param.expTd = mxGetPr(field);
				break;
            case 5:
				param.PotHistDuration = (int)mxGetScalar(field);
				break;
			case 6:
				param.PotHistStep = (int)mxGetScalar(field);
				break;
		} 
	}
	return param;
}
