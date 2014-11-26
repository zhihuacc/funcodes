#ifndef _FILTER_BANKS_H
#define _FILTER_BANKS_H

#include "tensor.h"
#include <vector>

using namespace std;

typedef vector<Tensor> Coef_Set;

class Filter_Bank
{
public:
	Filter_Bank(int n, const Tensor *filters);
//	int decompose(const Tensor &sig, Coef_Set &coefs);
//	int reconstruct(const Coef_Set &coefs, const MDSize &pre_size, Tensor &rec);

	int decompose(const Tensor &sig, Coef_Set &coefs, int lvls = 1);
	int reconstruct(const Coef_Set &coefs, Tensor &sig, int lvls = 1);

	int tensor_product(Filter_Bank &ouput);

private:
	int conv_then_downsample(const Tensor &comp, const Tensor &filter, Tensor &coef);
	int upsample_then_conv(const Tensor &coef, const Tensor &filter, const SmartArray &cur_size, Tensor &comp);

	vector<Tensor> _dec_filters;
//	vector<Tensor> _rec_filters;

	vector<SmartArray> _restored_size;
	vector<Tensor> _multilevel_dec_filters;
};

#endif
