/*
 * inpaint.h
 *
 *  Created on: 11 Feb, 2015
 *      Author: harvey
 */

#ifndef INPAINT_H_
#define INPAINT_H_

#include "wavelets_toolbox.h"
#include "denoising.h"

struct Inpainting_Param
{
	int 	sigma_num;
	vector<double> sigmas;
	int 	phase1_end;
	vector<int> phase_ends;
	vector<double> epsilons;
	double 	epsilon1;
	double 	epsilon2;
	double 	noise_sigma;
	int 	max_iter;
	vector<ML_MD_FS_Param> fs_params;
	vector<Thresholding_Param> thr_params;
	int 	ext_size;
	string 	ext_method;
};

int figure_good_sigmas(double est_sigma, double ratio, int phase1_num, int phase2_num, vector<double> &sigmas);

template<typename _Tp>
int inpaint(const Mat_<Vec<_Tp, 2> > &mat, const Mat_<Vec<_Tp, 2> > &mask, const Inpainting_Param &param, Mat_<Vec<_Tp, 2> > &res)
{
	Mat_<Vec<_Tp, 2> > noisy_masked_mat, rmask;
	int ndims = mat.dims;

	noisy_masked_mat = mat.mul(mask);
	double py = l2norm<_Tp>(noisy_masked_mat);

//	save_as_media<_Tp>("Test-Data/output/masked.avi", noisy_masked_mat, NULL);

	SmartIntArray border(ndims, param.ext_size);
	mat_border_extension(noisy_masked_mat, border, param.ext_method, noisy_masked_mat);

	rmask = Scalar(1.0,0) - mask;
	mat_border_extension(rmask, border, param.ext_method, rmask);

	Mat_<Vec<_Tp, 2> > last_restored, merged_restored,
				       cur_restored(ndims, noisy_masked_mat.size, Vec<_Tp, 2>(0,0));

	int sigma_id = 0;
	merged_restored = rmask.mul(cur_restored) + noisy_masked_mat;
	last_restored = cur_restored;
	for (int iter = 0; iter < param.max_iter && sigma_id < param.sigma_num; ++iter)
	{
//		cout << endl << "Iter ********** " << iter << ", sigma_id " << sigma_id << ", sigma " << param.sigmas[sigma_id] << endl;

		thresholding_denoise<_Tp>(merged_restored, param.fs_params[sigma_id], param.thr_params[sigma_id], cur_restored);
		pw_abs<_Tp>(cur_restored, cur_restored);

		merged_restored = rmask.mul(cur_restored) + noisy_masked_mat;

		//-- test error

		double err = l2norm<_Tp>(rmask.mul(last_restored - cur_restored));
		last_restored = cur_restored;

//		cout << "py: " << py << ", err: " << err;
		err /= py;
//		cout << ", err/py: " << err << endl;

		int pn = 0;
		while (sigma_id >= param.phase_ends[pn])
		{
			++pn;
		}

		if (err < param.epsilons[pn])
		{
			++sigma_id;
		}
		//--
	}

	if (fabs(param.noise_sigma - 0) < numeric_limits<_Tp>::epsilon())
	{
		//
		cur_restored = rmask.mul(cur_restored);
		mat_border_extension<_Tp>(cur_restored, border, "cut", cur_restored);
		cur_restored += mat.mul(mask);
	}
	else
	{
		mat_border_extension<_Tp>(cur_restored, border, "cut", cur_restored);
	}

	res = cur_restored;
	return 0;
}

template<typename _Tp>
int inpaint2(const Mat_<Vec<_Tp, 2> > &mat, const Mat_<Vec<_Tp, 2> > &mask, const Inpainting_Param &param, Mat_<Vec<_Tp, 2> > &res)
{
	Mat_<Vec<_Tp, 2> > noisy_masked_mat, rmask;
	int ndims = mat.dims;

	noisy_masked_mat = mat.mul(mask);
	double py = l2norm<_Tp>(noisy_masked_mat);

//	save_as_media<_Tp>("Test-Data/output/masked.avi", noisy_masked_mat, NULL);

	SmartIntArray border(ndims, param.ext_size);
	mat_border_extension(noisy_masked_mat, border, param.ext_method, noisy_masked_mat);

	rmask = Scalar(1.0,0) - mask;
	mat_border_extension(rmask, border, param.ext_method, rmask);

	Mat_<Vec<_Tp, 2> > last_restored, merged_restored,
				       cur_restored(ndims, noisy_masked_mat.size, Vec<_Tp, 2>(0,0));

	int sigma_id = 0;
	merged_restored = rmask.mul(cur_restored) + noisy_masked_mat;

	//
	Thresholding_Param deno_param = param.thr_params[0];
	deno_param.stdev = param.noise_sigma;
	thresholding_denoise<_Tp>(merged_restored, param.fs_params[0], deno_param, merged_restored);
	noisy_masked_mat = merged_restored.mul(Scalar(1.0,0) - rmask);
	//

	last_restored = cur_restored;
	for (int iter = 0; iter < param.max_iter && sigma_id < param.sigma_num; ++iter)
	{
//		cout << endl << "Iter ********** " << iter << ", sigma_id " << sigma_id << ", sigma " << param.sigmas[sigma_id] << endl;

		thresholding_denoise<_Tp>(merged_restored, param.fs_params[sigma_id], param.thr_params[sigma_id], cur_restored);
		pw_abs<_Tp>(cur_restored, cur_restored);

		merged_restored = rmask.mul(cur_restored) + noisy_masked_mat;

		//-- test error

		double err = l2norm<_Tp>(rmask.mul(last_restored - cur_restored));
		last_restored = cur_restored;

//		cout << "py: " << py << ", err: " << err;
		err /= py;
//		cout << ", err/py: " << err << endl;

		int pn = 0;
		while (sigma_id >= param.phase_ends[pn])
		{
			++pn;
		}

		if (err < param.epsilons[pn])
		{
			++sigma_id;
		}
		//--
	}

//	if (fabs(param.noise_sigma - 0) < numeric_limits<_Tp>::epsilon())
//	{
//		//
//		cur_restored = rmask.mul(cur_restored);
//		mat_border_extension<_Tp>(cur_restored, border, "cut", cur_restored);
//		cur_restored += mat.mul(mask);
//	}
//	else
//	{
//		mat_border_extension<_Tp>(cur_restored, border, "cut", cur_restored);
//	}

	cur_restored = rmask.mul(cur_restored);
	cur_restored += noisy_masked_mat;
	mat_border_extension<_Tp>(cur_restored, border, "cut", cur_restored);

	res = cur_restored;
	return 0;
}

template<typename _Tp>
int batch_inpaint(const Configuration *cfg, const string &top_scope)
{
	Inpainting_Param param;

	int ret = 0;
//	int ndims = 0;
	const char **fnames, **fmasks;
	int fnum, fnum2;
//	param.ndims = cfg->lookupInt(top_scope.c_str(), "ndims");
	cfg->lookupList(top_scope.c_str(), "fnames", fnames, fnum);
	cfg->lookupList(top_scope.c_str(), "masks", fmasks, fnum2);
	if (fnum != fnum2)
	{
		cout << "Mat and Mask numbers don't math. " << endl;
		return 0;
	}


	const char **sigma_list, **param_list;
	int param_num;

	const char **phase_ends_list, **epsilons_list;
	int phases_num, epsilons_num;
	cfg->lookupList(top_scope.c_str(), "phase_ends", phase_ends_list, phases_num);
	cfg->lookupList(top_scope.c_str(), "epsilons", epsilons_list, epsilons_num);
	if (phases_num != epsilons_num)
	{
		cout << "phase_ends dont match epsilons." << endl;
		return 0;
	}

	param.phase_ends.reserve(phases_num);
	param.phase_ends.resize(phases_num);
	param.epsilons.reserve(epsilons_num);
	param.epsilons.resize(phases_num);

	for (int i = 0; i < phases_num; ++i)
	{
		stringstream ss;
		ss << "phase_ends[" << i + 1 << "]";
		param.phase_ends[i] = cfg->stringToInt("", ss.str().c_str(), phase_ends_list[i]);

		ss.str("");
		ss << "epsilons[" << i + 1 << "]";
		param.epsilons[i] = cfg->stringToFloat("", ss.str().c_str(), epsilons_list[i]);
	}
	param.sigma_num = param.phase_ends[phases_num - 1];
	param.sigmas.reserve(param.sigma_num);
	param.sigmas.resize(param.sigma_num);
	param.fs_params.reserve(param.sigma_num);
	param.fs_params.resize(param.sigma_num);
	param.thr_params.reserve(param.sigma_num);
	param.thr_params.resize(param.sigma_num);

	string sigma_method( cfg->lookupString(top_scope.c_str(), "sigma_method") );
	if (sigma_method == "manul")
	{
		int sigma_num;
		cfg->lookupList(top_scope.c_str(), "sigmas", sigma_list, sigma_num);
		if (sigma_num != param.sigma_num)
		{
			cout << "sigmas' length is wrong." << endl;
			return 0;
		}

		for (int i = 0; i < sigma_num; ++i)
		{
			stringstream ss;
			ss << "sigmas[" << i + 1 << "]";
			param.sigmas[i] = cfg->stringToFloat("", ss.str().c_str(), sigma_list[i]);
		}
	}

	param.max_iter = cfg->lookupInt(top_scope.c_str(), "max_iter");
	param.noise_sigma = cfg->lookupFloat(top_scope.c_str(), "noise_sigma");

	param.ext_size = cfg->lookupInt(top_scope.c_str(), "ext_size");
	param.ext_method = cfg->lookupString(top_scope.c_str(), "ext_method");

	cfg->lookupList(top_scope.c_str(), "params", param_list, param_num);
	if (param.sigma_num != param_num)
	{
		return 0;
	}

	for (int f = 0; f < fnum; ++f)
	{
		string fname(fnames[f]);
		string fmask(fmasks[f]);

		Media_Format mfmt;
		Mat_<Vec<_Tp, 2> > clean_mat, noisy_masked_mat, mask, res;
		load_as_tensor<_Tp>(fname, clean_mat, &mfmt);
		load_as_tensor<_Tp>(fmask, mask, &mfmt);

		pw_less<_Tp>(complex<_Tp>(numeric_limits<_Tp>::epsilon(), 0), mask, mask);

		if (sigma_method == "auto")
		{
			double ratio = mean(mask)(0);
			param.sigmas.clear();
//			figure_good_sigmas(param.noise_sigma, ratio, param.phase1_end, param.sigma_num - param.phase1_end, param.sigmas);
			figure_good_sigmas(param.noise_sigma, ratio, param.phase_ends[0], param.sigma_num - param.phase_ends[0], param.sigmas);
		}

		int ndims = clean_mat.dims;
		for (int i = 0; i < param_num; ++i)
		{
			string param_scope = top_scope + "." + param_list[i];

			int nlevels = cfg->lookupInt(param_scope.c_str(), "nlevels");
			string fs_opt( cfg->lookupString(param_scope.c_str(), "fs") );
			bool is_sym = cfg->lookupBoolean(param_scope.c_str(), "is_sym");

			ret = compose_fs_param(nlevels, ndims, fs_opt, 0, "none", is_sym, param.fs_params[i]);
			if (ret)
			{
				cout << "Error in FS param: " << i << endl;
				return 0;
			}

			double c = static_cast<double>(cfg->lookupFloat(param_scope.c_str(), "c"));
			int wwidth = cfg->lookupInt(param_scope.c_str(), "wwidth");
			bool doNorm = cfg->lookupBoolean(param_scope.c_str(), "doNorm");
			string thr_method( cfg->lookupString(param_scope.c_str(), "thr_method") );

			ret = compose_thr_param(0, param.sigmas[i], c, wwidth, doNorm, thr_method, param.thr_params[i]);
			if (ret)
			{
				cout << "Error in Thr param: " << i << endl;
				return 0;
			}
		}

		//-- Fake up noisy data.
		Mat_<Vec<_Tp, 1> > channels[2];
		channels[0] = Mat_<Vec<_Tp, 1> >(ndims, clean_mat.size);
		channels[1] = Mat_<Vec<_Tp, 1> >(ndims, clean_mat.size, Vec<_Tp, 1>((_Tp)0));
		randn(channels[0], 0, param.noise_sigma);
		merge(channels, 2, noisy_masked_mat);
		channels[0].release();
		channels[1].release();

//		write_mat_dat<_Tp, 2>(noisy_masked_mat, "Test-Data/output/noises5.dat");
		noisy_masked_mat += clean_mat;
		// --

//		inpaint<_Tp>(noisy_masked_mat, mask, param, res);
		inpaint2<_Tp>(noisy_masked_mat, mask, param, res);

		double score, msr;
		psnr<_Tp>(clean_mat, res, score, msr);
//		cout << endl << "Done " << fname << "," << fmask << " noise sigma: "  << param.noise_sigma << ", psnr: " << score << ", msr: " << msr << endl;
		cout << score << endl;

	}
	return 0;
}


#endif /* INPAINT_H_ */
