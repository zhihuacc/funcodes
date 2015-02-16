#ifndef WAVELETS_DENOISING_H
#define WAVELETS_DENOISING_H

#include "../include/wavelets_toolbox.h"
#include <config4cpp/Configuration.h>
#include <opencv2/imgproc/imgproc.hpp>

using namespace config4cpp;

struct Thresholding_Param
{
	double mean;
	double stdev;
	double c;
	int    wwidth;

	bool   doNormalization;
	string thr_method;
};

int compose_thr_param(double mean, double stdev, double c, int wwidth, bool doNorm, const string &thr_opt, Thresholding_Param &thr_param);

template <typename _Tp>
int normalize_coefs(const typename ML_MC_Coefs_Set<_Tp>::type &coefs_set, const typename ML_MC_Filter_Norms_Set<_Tp>::type &norms_set, bool forward, typename ML_MC_Coefs_Set<_Tp>::type &ncoefs_set)
{
	int levels = (int)coefs_set.size();
	typename ML_MC_Coefs_Set<_Tp>::type new_coefs_set;
	new_coefs_set.reserve(levels);
	new_coefs_set.resize(levels);
	for (int i = 0; i < levels; ++i)
	{
		int n = (int)coefs_set[i].size();
		new_coefs_set[i].reserve(n);
		new_coefs_set[i].resize(n);

		const typename MC_Filter_Norms_Set<_Tp>::type &this_level_norms_set = norms_set[i];
		for (int j = 0; j < n; ++j)
		{
			if ((i == levels - 1) && j == (n - 1))
			{
				new_coefs_set[i][j] = coefs_set[i][j];
			}
			else
			{
				if (forward)
				{
					new_coefs_set[i][j] = coefs_set[i][j] / this_level_norms_set[j];
				}
				else
				{
					new_coefs_set[i][j] = coefs_set[i][j] * this_level_norms_set[j];
				}
			}
		}
	}

	ncoefs_set = new_coefs_set;

	return 0;
}


template<typename _Tp>
int bishrink2(const typename ML_MC_Coefs_Set<_Tp>::type &coefs_set, int wwidth, double c, double sigma, typename ML_MC_Coefs_Set<_Tp>::type &thr_set)
{
	int nd = coefs_set[0][0].dims;
	int nlevels = static_cast<int>( coefs_set.size() );
	SmartIntArray winsize(nd, wwidth);
	Mat_<Vec<_Tp, 2> > avg_window(nd, winsize, Vec<_Tp, 2>(1.0 / pow(wwidth, nd),0));

	Mat_<_Tp> kern(2, (int[]){1, wwidth}, 1.0 / wwidth);
	SmartArray<Mat_<_Tp> > skerns(nd, kern);

	typename ML_MC_Coefs_Set<_Tp>::type new_coefs_set;
	new_coefs_set.reserve(coefs_set.size());
	new_coefs_set.resize(coefs_set.size());
	SmartIntArray anchor(nd, wwidth / 2);
	for (int cur_lvl = 0; cur_lvl < nlevels - 1; ++cur_lvl)
	{
		const typename MC_Coefs_Set<_Tp>::type &this_level_set = coefs_set[cur_lvl];
		const typename MC_Coefs_Set<_Tp>::type &lower_level_set = coefs_set[cur_lvl + 1];
		new_coefs_set[cur_lvl].reserve(this_level_set.size());
		new_coefs_set[cur_lvl].resize(this_level_set.size());
		for (int idx = 0; idx < (int)this_level_set.size(); ++idx)
		{
			const Mat_<Vec<_Tp, 2> > Y = this_level_set[idx];
			Mat_<Vec<_Tp, 2> > Y_par;
			Mat_<_Tp> Y2, Y_par2, T, R;
//			_Tp inf = numeric_limits<_Tp>::epsilon();
			_Tp inf = 2.220446049250313e-16;
			pw_l2square_cr<_Tp>(Y, Y2);
			T = Y2.clone();

//			md_filtering<_Tp>(T, avg_window, anchor, T);
			separable_conv<_Tp>(T, skerns);

			sqrt(max(T - sigma*sigma, inf), T);
			T = (c*sigma*sigma) / T;
			// T is ready and real matrix

			Y_par = lower_level_set[idx].clone();
			SmartIntArray times(Y.dims);
			for (int i = 0; i < Y.dims; ++i)
			{
				times[i] = Y.size[i] / Y_par.size[i];
			}
			interpolate<_Tp>(Y_par, times, Y_par);
			pw_l2square_cr<_Tp>(Y_par, Y_par2);
//			pw_sqrt<_Tp, 2>(Y2 + Y_par2, R, 0);
			sqrt(Y2 + Y_par2, R);
			Y_par.release();
			Y_par2.release();
			Y2.release();
//			R -= T;

			max(R - T, 0, R);
			R /= max(T + R, inf);

			pw_mul_crc<_Tp>(Y, R, new_coefs_set[cur_lvl][idx]);
		}
	}
	new_coefs_set[nlevels - 1] = coefs_set[nlevels - 1];
	thr_set = new_coefs_set;

	return 0;
}



template<typename _Tp>
int local_soft2(const typename ML_MC_Coefs_Set<_Tp>::type &coefs_set, int wwidth, double c, double sigma, typename ML_MC_Coefs_Set<_Tp>::type &thr_set)
{
	int nlevels = static_cast<int>( coefs_set.size() );
	int nd = coefs_set[0][0].dims;
	SmartIntArray winsize(nd, wwidth);
	Mat_<Vec<_Tp, 2> > avg_window(nd, winsize, Vec<_Tp, 2>(1.0 / pow(wwidth, nd),0));
	SmartIntArray anchor(nd, wwidth / 2);

	Mat_<_Tp> kern(2, (int[]){1, wwidth}, 1.0 / wwidth);
	SmartArray<Mat_<_Tp> > skerns(nd, kern);

	typename ML_MC_Coefs_Set<_Tp>::type new_coefs_set;
	new_coefs_set.reserve(coefs_set.size());
	new_coefs_set.resize(coefs_set.size());
	for (int cur_lvl = 0; cur_lvl < nlevels; ++cur_lvl)
	{
		const typename MC_Coefs_Set<_Tp>::type &this_level_set = coefs_set[cur_lvl];
		new_coefs_set[cur_lvl].reserve(this_level_set.size());
		new_coefs_set[cur_lvl].resize(this_level_set.size());
		for (int idx = 0; idx < (int)this_level_set.size(); ++idx)
		{
			// Keep old lowpass coef unchanged.
			if (cur_lvl == nlevels - 1 && idx == static_cast<int>(this_level_set.size() - 1))
			{
				new_coefs_set[cur_lvl][idx] = coefs_set[cur_lvl][idx];
				continue;
			}
			const Mat_<Vec<_Tp, 2> > Y = this_level_set[idx];
			Mat_<_Tp> Y_abs, T, T_abs;
			_Tp inf = 2.220446049250313e-16;
//			_Tp inf = numeric_limits<_Tp>::epsilon();

			pw_l2square_cr<_Tp>(Y, T);   // T is single channel.
			sqrt(T, Y_abs);

			separable_conv<_Tp>(T, skerns);

			sqrt(max(T - sigma*sigma, inf), T);
			T = (c*sigma*sigma) / T;
			// T is ready

			Mat_<_Tp> mask;
			pw_less_rrr(T, Y_abs, mask);

			T = 1.0 - T / max(Y_abs, inf);
//			T = 1.0 - T / Y_abs;
			// T is ratio and ratio is ready

			pw_mul_crc<_Tp>(Y, T, new_coefs_set[cur_lvl][idx], mask);
		}
	}

	thr_set = new_coefs_set;
	return 0;
}



template<typename _Tp>
int local_adapt2(const typename ML_MC_Coefs_Set<_Tp>::type &coefs_set, int wwidth, double c, double sigma, typename ML_MC_Coefs_Set<_Tp>::type &thr_set)
{
	int nlevels = static_cast<int>( coefs_set.size() );
	int nd = coefs_set[0][0].dims;
	SmartIntArray winsize(nd, wwidth);
	Mat_<Vec<_Tp, 2> > avg_window(nd, winsize, Vec<_Tp, 2>(1.0 / pow(wwidth, nd),0));
	SmartIntArray anchor(nd, wwidth / 2);

	Mat_<_Tp> kern(2, (int[]){1, wwidth}, 1.0 / wwidth);
	SmartArray<Mat_<_Tp> > skerns(nd, kern);

	typename ML_MC_Coefs_Set<_Tp>::type new_coefs_set;
	new_coefs_set.reserve(coefs_set.size());
	new_coefs_set.resize(coefs_set.size());
	for (int cur_lvl = 0; cur_lvl < nlevels; ++cur_lvl)
	{
		const typename MC_Coefs_Set<_Tp>::type &this_level_set = coefs_set[cur_lvl];
		new_coefs_set[cur_lvl].reserve(this_level_set.size());
		new_coefs_set[cur_lvl].resize(this_level_set.size());
		for (int idx = 0; idx < (int)this_level_set.size(); ++idx)
		{
			//keep old lowpass coef unchanged.
			if (cur_lvl == nlevels - 1 && idx == static_cast<int>(this_level_set.size() - 1))
			{
				new_coefs_set[cur_lvl][idx] = coefs_set[cur_lvl][idx];
				continue;
			}

			const Mat_<Vec<_Tp, 2> > Y = this_level_set[idx];
			Mat_<_Tp> Y_abs, T, T_abs;
//			_Tp inf = numeric_limits<_Tp>::epsilon();
			_Tp inf = 2.220446049250313e-16;

			pw_l2square_cr<_Tp>(Y, Y_abs);
			sqrt(Y_abs, Y_abs);
//			pw_abs<_Tp>(Y, Y_abs);

//			md_filtering<_Tp>(Y_abs, avg_window, anchor, T);
			T = Y_abs.clone();
			separable_conv<_Tp>(T, skerns);

			pow(T, 2, T);

			sqrt(max(T - sigma*sigma, inf), T);
			T = (c*sigma*sigma) / T;
			// T is ready and real matrix

			Mat_<_Tp> mask;
			pw_less_rrr<_Tp>(T, Y_abs, mask);

			T = 1.0 - T / max(Y_abs, inf);
//			T = 1.0 - T / Y_abs;
			// T is ratio is ready

			pw_mul_crc<_Tp>(Y, T, new_coefs_set[cur_lvl][idx], mask);
		}
	}

	thr_set = new_coefs_set;
	return 0;
}

template<typename _Tp>
int thresholding(const typename ML_MC_Coefs_Set<_Tp>::type &coefs_set, const string &opt, int wwidth, double c, double sigma, typename ML_MC_Coefs_Set<_Tp>::type &thr_set)
{
	int ret = 0;
	if (opt == "localsoft")
	{
		ret = local_soft2<_Tp>(coefs_set, wwidth, c, sigma, thr_set);
	}
	else if (opt == "localadapt")
	{
		ret = local_adapt2<_Tp>(coefs_set, wwidth, c, sigma, thr_set);
	}
	else if (opt == "bishrink")
	{
		ret = bishrink2<_Tp>(coefs_set, wwidth, c, sigma, thr_set);
	}
	else
	{
		return -1;
	}

	return ret;
}

template <typename _Tp>
int thresholding_denoise(const Mat_<Vec<_Tp, 2> > &noisy_input, const ML_MD_FS_Param &fs_param, const Thresholding_Param &thr_param, Mat_<Vec<_Tp, 2> > &output)
{
	int ndims = noisy_input.dims;
	SmartIntArray input_size(ndims, noisy_input.size);
	SmartIntArray better_ext_border(ndims);


	figure_good_mat_size(fs_param, input_size, fs_param.ext_border, better_ext_border);

	cout <<"Actual extension border: " << endl;
	cout << "  ";
	for (int i = 0; i < ndims; ++i)
	{
		cout << better_ext_border[i] << " ";
	}
	cout << endl;

	Mat_<Vec<_Tp, 2> > ext_input, rec;
	mat_border_extension<_Tp>(noisy_input, better_ext_border, fs_param.ext_method, ext_input);


	ML_MD_FSystem<_Tp> 					filter_system;
	typename ML_MC_Coefs_Set<_Tp>::type 			coefs_set, new_coefs_set;
	typename ML_MC_Filter_Norms_Set<_Tp>::type 	norms_set;

	clock_t t0 = tic();
	decompose_by_ml_md_filter_bank2<_Tp>(fs_param, ext_input, filter_system, norms_set, coefs_set);
	clock_t t1 = tic();
	string msg = show_elapse(t1 - t0);
//	cout << endl << "Dec Time: " << endl << msg << endl;
	ext_input.release();

//	cout << "Filter Norms: " << endl;
//	for (int i = 0; i < (int)norms_set.size(); ++i)
//	{
//		for (int j = 0; j < (int)norms_set[i].size(); ++j)
//		{
//			cout << norms_set[i][j] << " ";
//		}
//		cout << endl;
//	}

	t0 = tic();
	if (thr_param.doNormalization)
	{
		normalize_coefs<_Tp>(coefs_set, norms_set, true, coefs_set);
	}
	thresholding<_Tp>(coefs_set, thr_param.thr_method, thr_param.wwidth, thr_param.c, thr_param.stdev, new_coefs_set);
	if (thr_param.doNormalization)
	{
		normalize_coefs<_Tp>(new_coefs_set, norms_set, false, new_coefs_set);
	}
	t1 = tic();
	msg = show_elapse(t1 - t0);
//	cout << endl << "Denoising Time: " << endl << msg << endl;

//	for (int i = 0; i < new_coefs_set.size(); ++i)
//	{
//		for (int j = 0; j < new_coefs_set[i].size(); ++j)
//		{
//			stringstream ss;
//			ss << "Test-Data/output/coef_" << i << "_" << j << ".txt";
//			print_mat_details_g<_Tp>(new_coefs_set[i][j], 2, ss.str());
//		}
//	}

	t0 = tic();
	reconstruct_by_ml_md_filter_bank2<_Tp>(fs_param, filter_system, new_coefs_set, rec);
	t1 = tic();
	msg = show_elapse(t1 - t0);
//	cout << "Rec Time: " << endl << msg << endl;

	mat_border_extension<_Tp>(rec, better_ext_border, "cut", rec);
	output = rec;
//	pw_abs<_Tp>(rec, output);

	return 0;
}

int denoise_entry(const Configuration *cfg, const string noisy_file);
int denoising_demo(const string &fn);

#endif
