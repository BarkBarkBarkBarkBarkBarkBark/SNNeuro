const int nParamField = 12;
const	char *paramField[12] = {
          "tm",
          "stdp_t_pos",
          "stdp_t_neg",
          "stdp_a_pos",
          "stdp_a_neg",
          "cte_ltd",
          "expTm",
          "freezeSTDP",
		  "WHistDuration",
		  "WHistStep",
          "PotHistDuration",
		  "PotHistStep"
};

typedef struct tag_param {

double tm;
int stdp_t_pos;
int stdp_t_neg;
double stdp_a_pos;
double stdp_a_neg;
double cte_ltd;

double *expTm;

bool freezeSTDP;

int WHistDuration;
int WHistStep;

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
			mexErrMsgIdAndTxt("missingParameterField",paramField[i]);
		}

		switch(i) {
            case 0:
				param.tm = mxGetScalar(field);
				break;
            case 1:
				param.stdp_t_pos = (int)mxGetScalar(field);
				break;
            case 2:
				param.stdp_t_neg = (int)mxGetScalar(field);
				break;
            case 3:
				param.stdp_a_pos = mxGetScalar(field);
				break;
            case 4:
				param.stdp_a_neg = mxGetScalar(field);
				break;
            case 5:
				param.cte_ltd = mxGetScalar(field);
				break;
            case 6:
				param.expTm = mxGetPr(field);
				break;
            case 7:
				param.freezeSTDP = (bool) mxGetScalar(field);
				break;
			case 8:
				param.WHistDuration = (int)mxGetScalar(field);
				break;
			case 9:
				param.WHistStep = (int)mxGetScalar(field);
				break;
            case 10:
				param.PotHistDuration = (int)mxGetScalar(field);
				break;
			case 11:
				param.PotHistStep = (int)mxGetScalar(field);
				break;
		} 
	}
	return param;
}
