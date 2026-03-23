const int nParamField = 33;
const	char *paramField[33] = {
          "tm",
          "refractoryPeriod",
          "resetPotentialFactor",
          "wiFactor",
          "inhibitoryPeriod",
          "stdp_t_pos",
          "stdp_t_neg",
          "stdp_t_lat_pos",
          "stdp_t_lat_neg",
          "stdp_a_pair",
          "stdp_a_post",
          "stdp_a_lat",
          "stdp_a_pre",
          "stdp_a_pair_neg",
          "stdp_a_post_neg",
          "stdp_a_lat_neg",
          "stdp_a_pre_neg",
          "WnegFactor",
          "DNWeight",
          "ip_t_pos",
          "ip_t_neg",
          "ip_dth_post",
          "ip_dth_pre",
          "maxTh",
          "minTh",
          "freezeSTDP",
          "ThHistDuration",
		  "ThHistStep",
          "PotHistDuration",
		  "PotHistStep",
          "WHistDuration",
		  "WHistStep",
          "wta_max"
};

typedef struct tag_param {

double tm;
double refractoryPeriod;
double resetPotentialFactor;
double wiFactor;
double inhibitoryPeriod;
double stdp_t_pos;
double stdp_t_neg;
double stdp_t_lat_pos;
double stdp_t_lat_neg;
double stdp_a_pair;
double stdp_a_post;
double stdp_a_lat;
double stdp_a_pre;
double stdp_a_pair_neg;
double stdp_a_post_neg;
double stdp_a_lat_neg;
double stdp_a_pre_neg;
double WnegFactor;
double DNWeight;

double ip_t_pos;
double ip_t_neg;
double ip_dth_post;
double ip_dth_pre;
double maxTh;
double minTh;

bool freezeSTDP;

double ThHistDuration;
double ThHistStep;

double PotHistDuration;
double PotHistStep;

double WHistDuration;
double WHistStep;

bool wta_max;

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
				param.refractoryPeriod = mxGetScalar(field);
				break;
            case 2:
				param.resetPotentialFactor = mxGetScalar(field);
				break;
            case 3:
				param.wiFactor = mxGetScalar(field);
				break;
            case 4:
				param.inhibitoryPeriod = mxGetScalar(field);
				break;
            case 5:
				param.stdp_t_pos = mxGetScalar(field);
				break;
            case 6:
				param.stdp_t_neg = mxGetScalar(field);
				break;
            case 7:
				param.stdp_t_lat_pos = mxGetScalar(field);
				break;
            case 8:
				param.stdp_t_lat_neg = mxGetScalar(field);
				break;
            case 9:
				param.stdp_a_pair = mxGetScalar(field);
				break;
            case 10:
				param.stdp_a_post = mxGetScalar(field);
				break;
            case 11:
				param.stdp_a_lat = mxGetScalar(field);
				break;
            case 12:
				param.stdp_a_pre = mxGetScalar(field);
				break;
            case 13:
				param.stdp_a_pair_neg = mxGetScalar(field);
				break;
            case 14:
				param.stdp_a_post_neg = mxGetScalar(field);
				break;
            case 15:
				param.stdp_a_lat_neg = mxGetScalar(field);
				break;
            case 16:
				param.stdp_a_pre_neg = mxGetScalar(field);
				break;
            case 17:
				param.WnegFactor = mxGetScalar(field);
				break;
            case 18:
				param.DNWeight = mxGetScalar(field);
				break;
            case 19:
				param.ip_t_pos = mxGetScalar(field);
				break;
            case 20:
				param.ip_t_neg = mxGetScalar(field);
				break;
            case 21:
				param.ip_dth_post = mxGetScalar(field);
				break;
            case 22:
				param.ip_dth_pre = mxGetScalar(field);
				break;
            case 23:
				param.maxTh = mxGetScalar(field);
				break;
            case 24:
				param.minTh = mxGetScalar(field);
				break;;
            case 25:
				param.freezeSTDP = (bool) mxGetScalar(field);
				break;
            case 26:
				param.ThHistDuration = mxGetScalar(field);
				break;
			case 27:
				param.ThHistStep =mxGetScalar(field);
				break;
            case 28:
				param.PotHistDuration = mxGetScalar(field);
				break;
			case 29:
				param.PotHistStep = mxGetScalar(field);
				break;
            case 30:
				param.WHistDuration = mxGetScalar(field);
				break;
			case 31:
				param.WHistStep = mxGetScalar(field);
				break;
            case 32:
				param.wta_max = (bool)mxGetScalar(field);
				break;
		} 
	}
	return param;
}
