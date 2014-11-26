#include "../include/filter_banks.h"
#include "../include/loaders.h"
#include <sstream>

Filter_Bank::Filter_Bank(int n, const Tensor *filters):_dec_filters(filters, filters + n)
{

}

int Filter_Bank::conv_then_downsample(const Tensor &comp, const Tensor &filter, Tensor &coef)
{
	Tensor conv_res;

	comp.conv(filter, conv_res);
	conv_res.downsample_by2(coef);

	return 0;
}

int Filter_Bank::upsample_then_conv(const Tensor &coef, const Tensor &filter, const SmartArray &restored_size, Tensor &comp)
{
	Tensor dsample_res;
	coef.upsample_by2(restored_size, dsample_res);
	dsample_res.conv(filter, comp);

	return 0;
}

//int Filter_Bank::decompose(const Tensor &comp, Coef_Set &coef)
//{
//	int nf = _dec_filters.size();
//	coef.reserve(nf);
//	coef.resize(nf);
//	for (int i = 0; i < nf; i++)
//	{
//		conv_then_downsample(comp, _dec_filters[i], coef[i]);
//	}
//
//	return 0;
//}
//
//int Filter_Bank::reconstruct(const Coef_Set &coefs, const MDSize &size, Tensor &comp)
//{
//	vector<Tensor> sub_comps;
//	int nf = _rec_filters.size();
//
//	sub_comps.reserve(nf);
//	sub_comps.resize(nf);
//	for (int i = 0; i < 3; i++)
//	{
//		upsample_then_conv(coefs[i], _rec_filters[i], size, sub_comps[i]);
//	}
//
//	comp = Tensor(size.dims, (const int*)size);
//	for (int i = 0; i < 3; i++)
//	{
//		comp.pw_add(sub_comps[i], comp);
//	}
//
//	return 0;
//}

int Filter_Bank::decompose(const Tensor &sig, Coef_Set &coefs, int lvls)
{
	Coef_Set cur_coefs;
	cur_coefs.push_back(sig);
	_restored_size.clear();
	_multilevel_dec_filters.clear();
	_multilevel_dec_filters.assign(_dec_filters.begin(), _dec_filters.end());
	int basic_dec_filters_num = _dec_filters.size();
	for (int i = 0; i <  lvls; ++i)
	{
		Tensor comp = cur_coefs[cur_coefs.size() - 1];
		cur_coefs.pop_back();
		_restored_size.push_back(comp.size());

		if (i > 0)
		{
			Tensor next_level_filter;
			int total_filters_num = _multilevel_dec_filters.size();
			for (int j = 0; j < basic_dec_filters_num; ++j)
			{
				_multilevel_dec_filters[total_filters_num - basic_dec_filters_num + j].folding(next_level_filter);
				_multilevel_dec_filters.push_back(next_level_filter);
			}
		}

		for (int j = 0; j < basic_dec_filters_num; ++j)
		{
			Tensor coef;
			conv_then_downsample(comp, _multilevel_dec_filters[_multilevel_dec_filters.size() - basic_dec_filters_num + j], coef);
			cur_coefs.push_back(coef);

			stringstream ss;
			ss << "./test2-coef-" << i << "-" << j << ".jpg";

			Media_Format mfmt;
			save_as_media(ss.str(), coef, &mfmt);
		}
	}

	coefs = cur_coefs;

	return 0;
}

int Filter_Bank::reconstruct(const Coef_Set &coefs, Tensor &sig, int lvls)
{
	//TODO: check _restored_size

	Coef_Set cur_coefs = coefs;
	for (int i = 0; i < lvls; ++i)
	{
		SmartArray restored_size = _restored_size[_restored_size.size() - 1];
		_restored_size.pop_back();

		Tensor comp = Tensor(restored_size.dims, (const int*)restored_size);
		int nf = _dec_filters.size();
		vector<Tensor> subcomps;
		subcomps.reserve(nf);
		subcomps.resize(nf);

		for (int j = 0; j < nf; ++j)
		{
			Tensor coef = cur_coefs[cur_coefs.size() - 1];
			cur_coefs.pop_back();

			Tensor rec_filter;
			_multilevel_dec_filters[_multilevel_dec_filters.size() - 1].conjugate_reflection(rec_filter);
			_multilevel_dec_filters.pop_back();

			upsample_then_conv(coef, rec_filter, restored_size, subcomps[j]);
			comp.pw_add(subcomps[j], comp);
		}

		cur_coefs.push_back(comp);
	}

	sig = cur_coefs[cur_coefs.size() - 1];

	return 0;
}
