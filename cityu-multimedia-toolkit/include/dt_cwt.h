#ifndef DT_CWT_H
#define DT_CWT_H

#include "tensor.h"

struct DT_CWT_Param
{
	double cl;
	double epl;
	double cr;
	double epr;
	double lend;
	double rend;
	double T;
	int m;
	int n;

};

int construct_dt_cwt_filter(const DT_CWT_Param &param, Tensor &output, Tensor &hi);

#endif
