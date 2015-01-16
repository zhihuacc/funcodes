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

	ncoefs_set = new_coefs_set;

	return 0;
}

template<typename _Tp>
int bishrink(const typename ML_MC_Coefs_Set<_Tp>::type &coefs_set, int wwidth, double c, double sigma, typename ML_MC_Coefs_Set<_Tp>::type &thr_set)
{
	int nd = coefs_set[0][0].dims;
	int nlevels = coefs_set.size();
	SmartIntArray winsize(nd, wwidth);
	Mat_<Vec<_Tp, 2> > avg_window(nd, winsize, Vec<_Tp, 2>(1.0 / pow(wwidth, nd),0));

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
			Mat_<Vec<_Tp, 2> > Y = this_level_set[idx].clone();
			Mat_<Vec<_Tp, 2> > Y_par = lower_level_set[idx].clone();
			Mat_<Vec<_Tp, 2> > Y_abs, Y_par_abs;
			Mat_<Vec<_Tp, 2> > T;
			Mat_<Vec<_Tp, 2> > T_abs;
			Mat_<Vec<_Tp, 2> > R;
			complex<_Tp> infinitesimal(1e-16,0);
			pw_abs<_Tp>(Y, Y_abs);
			pw_pow<_Tp>(Y_abs, static_cast<_Tp>(2), T);

			SmartIntArray border(nd, wwidth - 1);
			mat_border_extension(T, border, "blk", T);
			md_filtering<_Tp>(T, avg_window, anchor, T);
			mat_border_extension(T, border, "cut", T);

			T -= Scalar(sigma * sigma, 0);
			pw_max<_Tp>(T, infinitesimal, T);
			pw_sqrt<_Tp>(T, T);
			pw_div<_Tp>(complex<_Tp>(c*sigma*sigma, 0), T, T);
			// T is ready and real matrix

			SmartIntArray times(Y.dims);
			for (int i = 0; i < Y.dims; ++i)
			{
				times[i] = Y.size[i] / Y_par.size[i];
			}
			interpolate<_Tp>(Y_par, times, Y_par);
			Mat_<Vec<_Tp, 2> > tmp0, tmp1;
			pw_pow<_Tp>(Y_abs, static_cast<_Tp>(2), tmp0);
			pw_abs<_Tp>(Y_par, tmp1);
			pw_pow<_Tp>(tmp1, static_cast<_Tp>(2), tmp1);
			pw_sqrt<_Tp>(tmp0 + tmp1, R);
			R -= T;
			tmp0.release();
			tmp1.release();

			Mat_<Vec<_Tp, 2> > mask, ratio;
			pw_max<_Tp>(R, complex<_Tp>(0,0), R);  // This equals R * (R > 0)
			// R is ready

			Mat_<Vec<_Tp, 2> > r_t = R + T;
			pw_max<_Tp>(r_t, infinitesimal, r_t);
			pw_div<_Tp>(R, r_t, ratio);

			pw_mul<_Tp>(Y, ratio, new_coefs_set[cur_lvl][idx]);
		}
	}
	new_coefs_set[nlevels - 1] = coefs_set[nlevels - 1];
	thr_set = new_coefs_set;

	return 0;
}

template<typename _Tp>
int local_soft(const typename ML_MC_Coefs_Set<_Tp>::type &coefs_set, int wwidth, double c, double sigma, typename ML_MC_Coefs_Set<_Tp>::type &thr_set)
{
	int nd = coefs_set[0][0].dims;
	SmartIntArray winsize(nd, wwidth);
	Mat_<Vec<_Tp, 2> > avg_window(nd, winsize, Vec<_Tp, 2>(1.0 / pow(wwidth, nd),0));

	typename ML_MC_Coefs_Set<_Tp>::type new_coefs_set;
	new_coefs_set.reserve(coefs_set.size());
	new_coefs_set.resize(coefs_set.size());
	SmartIntArray anchor(nd, wwidth / 2);
	for (int cur_lvl = 0; cur_lvl < (int)coefs_set.size(); ++cur_lvl)
	{
		const typename MC_Coefs_Set<_Tp>::type &this_level_set = coefs_set[cur_lvl];
		new_coefs_set[cur_lvl].reserve(this_level_set.size());
		new_coefs_set[cur_lvl].resize(this_level_set.size());
		for (int idx = 0; idx < (int)this_level_set.size(); ++idx)
		{
			const Mat_<Vec<_Tp, 2> > Y = this_level_set[idx];
			Mat_<Vec<_Tp, 2> > Y_abs;
			Mat_<Vec<_Tp, 2> > T;
			Mat_<Vec<_Tp, 2> > T_abs;
			complex<_Tp> infinitesimal(1e-16, 0);

			pw_abs<_Tp>(Y, Y_abs);
			pw_pow<_Tp>(Y_abs, static_cast<_Tp>(2), T);

			SmartIntArray border(nd, wwidth - 1);
			mat_border_extension(T, border, "blk", T);
			md_filtering<_Tp>(T, avg_window, anchor, T);
			mat_border_extension(T, border, "cut", T);

			T -= Scalar(sigma * sigma, 0);
			pw_max<_Tp>(T, infinitesimal, T);
			pw_sqrt<_Tp>(T, T);
			pw_div<_Tp>(complex<_Tp>(c*sigma*sigma, 0), T, T);
			// T is ready and real matrix


			Mat_<Vec<_Tp, 2> > mask;
			pw_less<_Tp>(T, Y_abs, mask);

			pw_max<_Tp>(Y_abs, infinitesimal, Y_abs);
			Mat_<Vec<_Tp, 2> > ratio;
			pw_div<_Tp>(T, Y_abs, ratio);

			T.release();

			ratio = Scalar(1.0,0) - ratio;
			// ratio is ready

			Mat_<Vec<_Tp, 2> > new_Y;
			pw_mul<_Tp>(Y, ratio, new_Y);
			pw_mul<_Tp>(mask, new_Y, new_Y);

			new_coefs_set[cur_lvl][idx] = new_Y;
		}
	}

	thr_set = new_coefs_set;
	return 0;
}


template<typename _Tp>
int local_adapt(const typename ML_MC_Coefs_Set<_Tp>::type &coefs_set, int wwidth, double c, double sigma, typename ML_MC_Coefs_Set<_Tp>::type &thr_set)
{
	int nd = coefs_set[0][0].dims;
	SmartIntArray winsize(nd, wwidth);
	Mat_<Vec<_Tp, 2> > avg_window(nd, winsize, Vec<_Tp, 2>(1.0 / pow(wwidth, nd),0));

	typename ML_MC_Coefs_Set<_Tp>::type new_coefs_set;
	new_coefs_set.reserve(coefs_set.size());
	new_coefs_set.resize(coefs_set.size());
	SmartIntArray anchor(nd, wwidth / 2);
	for (int cur_lvl = 0; cur_lvl < (int)coefs_set.size(); ++cur_lvl)
	{
		const typename MC_Coefs_Set<_Tp>::type &this_level_set = coefs_set[cur_lvl];
		new_coefs_set[cur_lvl].reserve(this_level_set.size());
		new_coefs_set[cur_lvl].resize(this_level_set.size());
		for (int idx = 0; idx < (int)this_level_set.size(); ++idx)
		{
			Mat_<Vec<_Tp, 2> > Y = this_level_set[idx].clone();
			Mat_<Vec<_Tp, 2> > Y_abs;
			Mat_<Vec<_Tp, 2> > T;
			Mat_<Vec<_Tp, 2> > T_abs;
			complex<_Tp> infinitesimal(1e-16, 0);
			pw_abs<_Tp>(Y, Y_abs);

			SmartIntArray border(nd, wwidth - 1);
			mat_border_extension(Y_abs, border, "blk", T);
			md_filtering<_Tp>(T, avg_window, anchor, T);
			mat_border_extension(T, border, "cut", T);

			pw_pow<_Tp>(T, static_cast<_Tp>(2), T);

			T -= Scalar(sigma * sigma, 0);
			pw_max<_Tp>(T, infinitesimal, T);
			pw_sqrt<_Tp>(T, T);
			pw_div<_Tp>(complex<_Tp>(c*sigma*sigma, 0), T, T);
			// T is ready and real matrix

			Mat_<Vec<_Tp, 2> > mask;
			pw_less<_Tp>(T, Y_abs, mask);

			pw_max<_Tp>(Y_abs, infinitesimal, Y_abs);
			Mat_<Vec<_Tp, 2> > ratio;
			pw_div<_Tp>(T, Y_abs, ratio);

			T.release();

			ratio = Scalar(1.0,0) - ratio;
			// ratio is ready

			Mat_<Vec<_Tp, 2> > new_Y;
			pw_mul<_Tp>(Y, ratio, new_Y);
			pw_mul<_Tp>(mask, new_Y, new_Y);

			new_coefs_set[cur_lvl][idx] = new_Y;

//			stringstream ss;
//			ss << "Test-Data/output/new-coef-" << cur_lvl << "-" << idx <<".txt";
//			print_mat_details_g<_Tp>(new_Y, 2, ss.str());
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
		ret = local_soft<_Tp>(coefs_set, wwidth, c, sigma, thr_set);
	}
	else if (opt == "localadapt")
	{
		ret = local_adapt<_Tp>(coefs_set, wwidth, c, sigma, thr_set);
	}
	else if (opt == "bishrink")
	{
		ret = bishrink<_Tp>(coefs_set, wwidth, c, sigma, thr_set);
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

	cout << endl << "Extension border: " << endl;
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
	cout << endl << "Dec Time: " << endl << msg << endl;
	ext_input.release();

//	cout << "Filter Norms: " << endl;
//	for (int i = 0; i < nlevels; ++i)
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
	cout << endl << "Denoising Time: " << endl << msg << endl;

	t0 = tic();
	reconstruct_by_ml_md_filter_bank2<_Tp>(fs_param, filter_system, new_coefs_set, rec);
	t1 = tic();
	msg = show_elapse(t1 - t0);
	cout << "Rec Time: " << endl << msg << endl;

	mat_border_extension<_Tp>(rec, better_ext_border, "cut", rec);
	output = rec;

	return 0;
}

int denoise_entry(const Configuration *cfg, const string noisy_file);

int psnr_entry(const string &left, const string &right);

#endif
