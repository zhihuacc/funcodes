#include "../include/wavelets_toolbox.h"
#include "../include/denoising.h"

#include <opencv2/imgproc/imgproc.hpp>

int compose_thr_param(double mean, double stdev, double c, int wwidth, bool doNorm, const string &thr_opt, Thresholding_Param &thr_param)
{
	if (stdev < 0)
	{
		return -1;
	}
	if ((wwidth & 1) == 0)    // wwidth must be odd.
	{
		return -2;
	}
	if (thr_opt != "localsoft" && thr_opt != "localadapt" && thr_opt != "bishrink")
	{
		return -3;
	}

	Thresholding_Param param;
	param.mean = mean;
	param.stdev = stdev;
	param.c = c;
	param.wwidth = wwidth;
	param.doNormalization = doNorm;
	param.thr_method = thr_opt;
	thr_param = param;

	return 0;
}


