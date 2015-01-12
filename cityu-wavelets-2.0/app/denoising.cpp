#include "../include/wavelets_toolbox.h"
#include "../include/denoising.h"

#include <opencv2/imgproc/imgproc.hpp>
//
//template <typename _Tp>
//int normalize_coefs(const typename ML_MC_Coefs_Set<_Tp>::type &coefs_set, const typename ML_MC_Filter_Norms_Set<_Tp>::type &norms_set, bool forward, typename ML_MC_Coefs_Set<_Tp>::type &ncoefs_set)
//{
//	int levels = (int)coefs_set.size();
//	typename ML_MC_Coefs_Set<_Tp>::type new_coefs_set;
//	new_coefs_set.reserve(levels);
//	new_coefs_set.resize(levels);
//	for (int i = 0; i < levels; ++i)
//	{
//		int n = (int)coefs_set[i].size();
//		new_coefs_set[i].reserve(n);
//		new_coefs_set[i].resize(n);
//
//		const typename MC_Filter_Norms_Set<_Tp>::type &this_level_norms_set = norms_set[i];
//		for (int j = 0; j < n; ++j)
//		{
//			if (forward)
//			{
//				new_coefs_set[i][j] = coefs_set[i][j] / this_level_norms_set[j];
//			}
//			else
//			{
//				new_coefs_set[i][j] = coefs_set[i][j] * this_level_norms_set[j];
//			}
//		}
//	}
//
//	ncoefs_set = new_coefs_set;
//
//	return 0;
//}
//
//template<typename _Tp>
//int bishrink(const typename ML_MC_Coefs_Set<_Tp>::type &coefs_set, int wwidth, double c, double sigma, typename ML_MC_Coefs_Set<_Tp>::type &thr_set)
//{
//	int nd = coefs_set[0][0].dims;
//	int nlevels = coefs_set.size();
//	SmartIntArray winsize(nd, wwidth);
//	Mat avg_window(nd, winsize, CV_64FC2, Scalar(1.0 / pow(wwidth, nd),0));
//
//	typename ML_MC_Coefs_Set<_Tp>::type new_coefs_set;
//	new_coefs_set.reserve(coefs_set.size());
//	new_coefs_set.resize(coefs_set.size());
//	SmartIntArray anchor(nd, wwidth / 2);
//	for (int cur_lvl = 0; cur_lvl < (int)nlevels - 1; ++cur_lvl)
//	{
//		const typename MC_Coefs_Set<_Tp>::type &this_level_set = coefs_set[cur_lvl];
//		const typename MC_Coefs_Set<_Tp>::type &lower_level_set = coefs_set[cur_lvl + 1];
//		new_coefs_set[cur_lvl].reserve(this_level_set.size());
//		new_coefs_set[cur_lvl].resize(this_level_set.size());
//		for (int idx = 0; idx < (int)this_level_set.size(); ++idx)
//		{
//			Mat_<Vec<_Tp, 2> > Y = this_level_set[idx].clone();
//			Mat_<Vec<_Tp, 2> > Y_par = lower_level_set[idx].clone();
//			Mat_<Vec<_Tp, 2> > Y_abs, Y_par_abs;
//			Mat_<Vec<_Tp, 2> > T;
//			Mat_<Vec<_Tp, 2> > T_abs;
//			Mat_<Vec<_Tp, 2> > R;
//			Mat_<Vec<_Tp, 2> > good_to_div;
//			complex<_Tp> infinitesimal(1e-16,0);
//			pw_abs<_Tp>(Y, Y_abs);
//			pw_pow<_Tp>(Y_abs, 2, T);
//			md_filtering<_Tp>(T, avg_window, anchor, T);
//
//			T -= Scalar(sigma * sigma, 0);
//			pw_max<_Tp>(T, infinitesimal, T);
//			pw_sqrt<_Tp>(T, T);
//			pw_div<_Tp>(complex<_Tp>(c*sigma*sigma, 0), T, T);
//			// T is ready and real matrix
//
//			SmartIntArray times(Y.dims);
//			for (int i = 0; i < Y.dims; ++i)
//			{
//				times[i] = Y.size[i] / Y_par.size[i];
//			}
//			interpolate<_Tp>(Y_par, times, Y_par);
//			Mat_<Vec<_Tp, 2> > tmp0, tmp1;
//			pw_pow<_Tp>(Y_abs, 2, tmp0);
//			pw_abs<_Tp>(Y_par, tmp1);
//			pw_pow<_Tp>(tmp1, 2, tmp1);
//			pw_sqrt<_Tp>(tmp0 + tmp1, R);
//			R -= T;
//
//			Mat_<Vec<_Tp, 2> > mask, ratio;
//			pw_max<_Tp>(R, complex<_Tp>(0,0), R);
//			// R is ready
//
//			Mat_<Vec<_Tp, 2> > r_t = R + T;
//			pw_max<_Tp>(r_t, infinitesimal, r_t);
//			pw_div<_Tp>(R, r_t, ratio);
//
//			pw_mul<_Tp>(Y, ratio, new_coefs_set[cur_lvl][idx]);
//		}
//	}
//	new_coefs_set[nlevels - 1] = coefs_set[nlevels - 1];
//	thr_set = new_coefs_set;
//
//	return 0;
//}
//
//template<typename _Tp>
//int local_soft(const typename ML_MC_Coefs_Set<_Tp>::type &coefs_set, int wwidth, double c, double sigma, typename ML_MC_Coefs_Set<_Tp>::type &thr_set)
//{
//	int nd = coefs_set[0][0].dims;
//	SmartIntArray winsize(nd, wwidth);
//	Mat avg_window(nd, (const int *)winsize, CV_64FC2, Scalar(1.0 / pow(wwidth, nd),0));
//
//	typename ML_MC_Coefs_Set<_Tp>::type new_coefs_set;
//	new_coefs_set.reserve(coefs_set.size());
//	new_coefs_set.resize(coefs_set.size());
//	SmartIntArray anchor(nd, wwidth / 2);
//	for (int cur_lvl = 0; cur_lvl < (int)coefs_set.size(); ++cur_lvl)
//	{
//		const typename MC_Coefs_Set<_Tp>::type &this_level_set = coefs_set[cur_lvl];
//		new_coefs_set[cur_lvl].reserve(this_level_set.size());
//		new_coefs_set[cur_lvl].resize(this_level_set.size());
//		for (int idx = 0; idx < (int)this_level_set.size(); ++idx)
//		{
//			const Mat_<Vec<_Tp, 2> > Y = this_level_set[idx];
//			Mat_<Vec<_Tp, 2> > Y_abs;
//			Mat_<Vec<_Tp, 2> > T;
//			Mat_<Vec<_Tp, 2> > T_abs;
//			complex<_Tp> infinitesimal(1e-16, 0);
//
//			pw_abs<_Tp>(Y, Y_abs);
//			pw_pow<_Tp>(Y_abs, static_cast<_Tp>(2), T);
//			md_filtering<_Tp>(T, avg_window, anchor, T);
//
//			T -= Scalar(sigma * sigma, 0);
//			pw_max<_Tp>(T, infinitesimal, T);
//			pw_sqrt<_Tp>(T, T);
//			pw_div<_Tp>(complex<_Tp>(c*sigma*sigma, 0), T, T);
//			// T is ready and real matrix
//
//
//			Mat_<Vec<_Tp, 2> > mask;
//			pw_less<_Tp>(T, Y_abs, mask);
//
//			pw_max<_Tp>(Y_abs, infinitesimal, Y_abs);
//			Mat_<Vec<_Tp, 2> > ratio;
//			pw_div<_Tp>(T, Y_abs, ratio);
//
//			T.release();
//
//			ratio = Scalar(1.0,0) - ratio;
//			// ratio is ready
//
//			Mat_<Vec<_Tp, 2> > new_Y;
//			pw_mul<_Tp>(Y, ratio, new_Y);
//			pw_mul<_Tp>(mask, new_Y, new_Y);
//
//			new_coefs_set[cur_lvl][idx] = new_Y;
//		}
//	}
//
//	thr_set = new_coefs_set;
//	return 0;
//}
//
//
//template<typename _Tp>
//int local_adapt(const typename ML_MC_Coefs_Set<_Tp>::type &coefs_set, int wwidth, double c, double sigma, typename ML_MC_Coefs_Set<_Tp>::type &thr_set)
//{
//	int nd = coefs_set[0][0].dims;
//	SmartIntArray winsize(nd, wwidth);
//	Mat_<Vec<_Tp, 2> > avg_window(nd, winsize, Vec<_Tp, 2>(1.0 / pow(wwidth, nd),0));
//
//	typename ML_MC_Coefs_Set<_Tp>::type new_coefs_set;
//	new_coefs_set.reserve(coefs_set.size());
//	new_coefs_set.resize(coefs_set.size());
//	SmartIntArray anchor(nd, wwidth / 2);
//	for (int cur_lvl = 0; cur_lvl < (int)coefs_set.size(); ++cur_lvl)
//	{
//		const typename MC_Coefs_Set<_Tp>::type &this_level_set = coefs_set[cur_lvl];
//		new_coefs_set[cur_lvl].reserve(this_level_set.size());
//		new_coefs_set[cur_lvl].resize(this_level_set.size());
//		for (int idx = 0; idx < (int)this_level_set.size(); ++idx)
//		{
//			Mat_<Vec<_Tp, 2> > Y = this_level_set[idx].clone();
//			Mat_<Vec<_Tp, 2> > Y_abs;
//			Mat_<Vec<_Tp, 2> > T;
//			Mat_<Vec<_Tp, 2> > T_abs;
//			complex<_Tp> infinitesimal(1e-16, 0);
//			pw_abs<_Tp>(Y, Y_abs);
//			md_filtering<_Tp>(Y_abs, avg_window, anchor, T);
//			pw_pow<_Tp>(T, static_cast<_Tp>(2), T);
//
//			T -= Scalar(sigma * sigma, 0);
//			pw_max<_Tp>(T, infinitesimal, T);
//			pw_sqrt<_Tp>(T, T);
//			pw_div<_Tp>(complex<_Tp>(c*sigma*sigma, 0), T, T);
//			// T is ready and real matrix
//
//			Mat_<Vec<_Tp, 2> > mask;
//			pw_less<_Tp>(T, Y_abs, mask);
//
//			pw_max<_Tp>(Y_abs, infinitesimal, Y_abs);
//			Mat_<Vec<_Tp, 2> > ratio;
//			pw_div<_Tp>(T, Y_abs, ratio);
//
//			T.release();
//
//			ratio = Scalar(1.0,0) - ratio;
//			// ratio is ready
//
//			Mat_<Vec<_Tp, 2> > new_Y;
//			pw_mul<_Tp>(Y, ratio, new_Y);
//			pw_mul<_Tp>(mask, new_Y, new_Y);
//
//			new_coefs_set[cur_lvl][idx] = new_Y;
//
////			stringstream ss;
////			ss << "Test-Data/output/new-coef-" << cur_lvl << "-" << idx <<".txt";
////			print_mat_details_g<_Tp>(new_Y, 2, ss.str());
//		}
//	}
//
//	thr_set = new_coefs_set;
//	return 0;
//}
//
//template<typename _Tp>
//int thresholding(const typename ML_MC_Coefs_Set<_Tp>::type &coefs_set, const string &opt, int wwidth, double c, double sigma, typename ML_MC_Coefs_Set<_Tp>::type &thr_set)
//{
//	int ret = 0;
//	if (opt == "localsoft")
//	{
//		ret = local_soft<_Tp>(coefs_set, wwidth, c, sigma, thr_set);
//	}
//	else if (opt == "localadapt")
//	{
//		ret = local_adapt<_Tp>(coefs_set, wwidth, c, sigma, thr_set);
//	}
//	else if (opt == "bishrink")
//	{
//		ret = bishrink<_Tp>(coefs_set, wwidth, c, sigma, thr_set);
//	}
//	else
//	{
//		return -1;
//	}
//
//	return ret;
//}
//
//int thresholding_denoise(const Mat &noisy_input, const ML_MD_FS_Param &fs_param, const Thresholding_Param &thr_param, Mat &output)
//{
//	int ndims = noisy_input.dims;
//	int depth = noisy_input.depth();
//	SmartIntArray input_size(ndims, noisy_input.size);
//	SmartIntArray better_ext_border(ndims);
//
//	figure_good_mat_size(fs_param, input_size, fs_param.ext_border, better_ext_border);
//
//	Mat ext_input;
//	if (depth == 4)
//	{
//		mat_border_extension<float>(static_cast<const Mat_<Vec<float, 2>  > &>(noisy_input), better_ext_border, fs_param.ext_method, static_cast<Mat_<Vec<float, 2>  > &>(ext_input));
//	}
//	else if (depth == 8)
//	{
//		mat_border_extension<double>(static_cast<const Mat_<Vec<double, 2>  > &>(noisy_input), better_ext_border, fs_param.ext_method, static_cast<Mat_<Vec<double, 2>  > &>(ext_input));
//	}
//
////	for (int i = 0; i < fs_param.nlevels; ++i)
////	{
////		for (int j = 0; j < fs_param.ndims; ++j)
////		{
////			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].ctrl_points = Smart64FArray(3, (double[]){-33.0/32.0, 33.0/32.0, M_PI});
////			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].epsilons = Smart64FArray(3, (double[]){69.0/128.0, 69.0/128.0, 51.0/512.0});
////			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].degree = 1;
////			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].highpass_ds_folds = SmartIntArray(3, (int[]){2,2,2});
////			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].lowpass_ds_fold = 2;
////			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].opt = "sincos";
////
//////			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].ctrl_points = Smart64FArray(6, (double[]){-2, -1.145796, 0, 1.145796, 2, M_PI});
//////			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].epsilons = Smart64FArray(6, (double[]){0.35, 0.3, 0.125, 0.3, 0.35, 0.0778});
//////			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].degree = 1;
//////			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].highpass_ds_folds = SmartIntArray(6, (int[]){4,4,4,4,4,4});
//////			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].lowpass_ds_fold = 2;
//////			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].opt = "sincos";
////		}
////	}
//
//	ML_MD_FSystem<double> filter_system;
//	ML_MC_Coefs_Set<double>::type coefs_set;
//	ML_MC_Filter_Norms_Set<double>::type norms_set;
//
//	clock_t t0 = tic();
//	decompose_by_ml_md_filter_bank<double>(fs_param, mat_ext, filter_system, norms_set, coefs_set);
//	clock_t t1 = tic();
//	string msg = show_elapse(t1 - t0);
//	cout << msg << endl;
//	mat_ext.release();
//
//	cout << "Filter Norms: " << endl;
//	for (int i = 0; i < nlevels; ++i)
//	{
//		for (int j = 0; j < (int)norms_set[i].size(); ++j)
//		{
//			cout << norms_set[i][j] << " ";
//		}
//		cout << endl;
//	}
//
//
//	ML_MC_Coefs_Set<double>::type new_coefs_set;
//	normalize_coefs<double>(coefs_set, norms_set, true, coefs_set);
//	thresholding<double>(coefs_set, thr_opt, wwidth, c, stddev, new_coefs_set);
//	normalize_coefs<double>(new_coefs_set, norms_set, false, new_coefs_set);
////	new_coefs_set = coefs_set;
//
//	t0 = tic();
//	reconstruct_by_ml_md_filter_bank<double>(fs_param, filter_system, new_coefs_set, rec);
//	t1 = tic();
//	msg = show_elapse(t1 - t0);
//	cout << "Rec Time: " << endl << msg << endl;
//
//	mat_border_extension<double>(rec, border, "cut", mat_cut);
////		mat_cut = rec;
//
//
////	save_as_media<double>("Test-Data/output/Lena512-rec.png", mat_cut, &mfmt);
//
//	double score, msr;
//	psnr<double>(mat, mat_cut, score, msr);
//	cout << "PSNR: score: " << score << ", msr: " << msr << endl;
//
//	return 0;
//}


//int app_denoising()
//{
//	int nlevels = 2;
//	int nd = 0;
//	double mean = 0;
//	double stddev = 5;
//	double c = 1;
//	double wwidth = 7;
//	string thr_opt("localsoft");
//
//	string img_names("Test-Data/coastguard144.avi");
//
//	Media_Format mfmt;
//	Mat_<Vec<double, 2> > mat, noisy_img, mat_ext, mat_cut, rec;
//	load_as_tensor<double>(img_names, mat, &mfmt);
//	nd = mat.dims;
//
//	Mat_<Vec<double, 1> > channels[2];
//	channels[0] = Mat_<Vec<double, 1> >(mat.dims, mat.size);
//	channels[1] = Mat_<Vec<double, 1> >(mat.dims, mat.size, Vec<double, 1>((double)0));
//	randn(channels[0], mean, stddev);
//	merge(channels, 2, noisy_img);
//	channels[0].release();
//	channels[1].release();
//
//	noisy_img = mat + noisy_img;
//
//	ML_MD_FS_Param fs_param(nlevels, nd);
//
//	SmartIntArray border(mat.dims, mat.size);
//	for (int i = 0; i < nd; ++i)
//	{
//		border[i] = ((noisy_img.size[i] + 32) & (~((1 << nlevels) - 1))) - noisy_img.size[i];
//	}
//
////	border[0] = 12;
////	border[1] = 16;
////	border[2] = 16;
//
//	mat_border_extension<double>(noisy_img, border, "sym", mat_ext);
//
//	for (int i = 0; i < fs_param.nlevels; ++i)
//	{
//		for (int j = 0; j < fs_param.ndims; ++j)
//		{
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].ctrl_points = Smart64FArray(3, (double[]){-33.0/32.0, 33.0/32.0, M_PI});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].epsilons = Smart64FArray(3, (double[]){69.0/128.0, 69.0/128.0, 51.0/512.0});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].degree = 1;
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].highpass_ds_folds = SmartIntArray(3, (int[]){2,2,2});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].lowpass_ds_fold = 2;
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].opt = "sincos";
//
////			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].ctrl_points = Smart64FArray(6, (double[]){-2, -1.145796, 0, 1.145796, 2, M_PI});
////			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].epsilons = Smart64FArray(6, (double[]){0.35, 0.3, 0.125, 0.3, 0.35, 0.0778});
////			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].degree = 1;
////			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].highpass_ds_folds = SmartIntArray(6, (int[]){4,4,4,4,4,4});
////			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].lowpass_ds_fold = 2;
////			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].opt = "sincos";
//		}
//	}
//
//	ML_MD_FSystem<double> filter_system;
//	ML_MC_Coefs_Set<double>::type coefs_set;
//	ML_MC_Filter_Norms_Set<double>::type norms_set;
//
//	clock_t t0 = tic();
//	decompose_by_ml_md_filter_bank<double>(fs_param, mat_ext, filter_system, norms_set, coefs_set);
//	clock_t t1 = tic();
//	string msg = show_elapse(t1 - t0);
//	cout << msg << endl;
//	mat_ext.release();
//
//
//
//	cout << "Filter Norms: " << endl;
//	for (int i = 0; i < nlevels; ++i)
//	{
//		for (int j = 0; j < (int)norms_set[i].size(); ++j)
//		{
//			cout << norms_set[i][j] << " ";
//		}
//		cout << endl;
//	}
//
//
//	ML_MC_Coefs_Set<double>::type new_coefs_set;
//	normalize_coefs<double>(coefs_set, norms_set, true, coefs_set);
//	thresholding<double>(coefs_set, thr_opt, wwidth, c, stddev, new_coefs_set);
//	normalize_coefs<double>(new_coefs_set, norms_set, false, new_coefs_set);
////	new_coefs_set = coefs_set;
//
//	t0 = tic();
//	reconstruct_by_ml_md_filter_bank<double>(fs_param, filter_system, new_coefs_set, rec);
//	t1 = tic();
//	msg = show_elapse(t1 - t0);
//	cout << "Rec Time: " << endl << msg << endl;
//
//	mat_border_extension<double>(rec, border, "cut", mat_cut);
////		mat_cut = rec;
//
//
////	save_as_media<double>("Test-Data/output/Lena512-rec.png", mat_cut, &mfmt);
//
//	double score, msr;
//	psnr<double>(mat, mat_cut, score, msr);
//	cout << "PSNR: score: " << score << ", msr: " << msr << endl;
//
//	return 0;
//}
//
//int app_denoising()
//{
//	const int nnoises = 3;
//	const int nimgs = 3;
//
//	double mean[nnoises] = {0,0,0};
//	double stddev[nnoises] = {5, 10, 15};
//
//	Mat origin_imgs[nimgs];
//	string img_names[nimgs] = {"Test-Data/Lena512.png", "Test-Data/hill.png", "Test-Data/fingerprint.png"};
//
//	Media_Format mfmt;
////	load_as_tensor("Test-Data/Lena512.png", origin_imgs[0], &mfmt);
////	load_as_tensor("Test-Data/hill.png", origin_imgs[1], &mfmt);
////	load_as_tensor("Test-Data/fingerprint.png", origin_imgs[2], &mfmt);
//
//	SmartIntArray bder_size(origin_imgs[0].dims);
//	for (int i = 0; i < bder_size.len; ++i)
//	{
//		bder_size[i] = 64;
//	}
//
//	int nlevels = 2;
//	int nd = 2;
//	MLevel_MDFilter_System_Param ml_md_fs_param;
//	ml_md_fs_param.md_fs_param_for_each_level.reserve(nlevels);
//	ml_md_fs_param.lowpass_approx_ds_folds.reserve(nlevels);
//	for (int i = 0; i < ml_md_fs_param.md_fs_param_for_each_level.len; ++i)
//	{
//		ml_md_fs_param.md_fs_param_for_each_level[i].reserve(nd);
//		for (int j = 0; j < ml_md_fs_param.md_fs_param_for_each_level[i].len; ++j)
//		{
//			ml_md_fs_param.md_fs_param_for_each_level[i][j].ctrl_points = Smart64FArray(3, (double[]){-33.0/32.0, 33.0/32.0, M_PI});
//			ml_md_fs_param.md_fs_param_for_each_level[i][j].epsilons = Smart64FArray(3, (double[]){69.0/128.0, 69.0/128.0, 51.0/512.0});
//			ml_md_fs_param.md_fs_param_for_each_level[i][j].degree = 1;
//			ml_md_fs_param.md_fs_param_for_each_level[i][j].folds = SmartIntArray(3, (int[]){2,2,2});
//			ml_md_fs_param.md_fs_param_for_each_level[i][j].opt = "sincos";
//
////			ml_md_fs_param.md_fs_param_for_each_level[i][j].ctrl_points = Smart64FArray(6, (double[]){-2, -1.145796, 0, 1.145796, 2, M_PI});
////			ml_md_fs_param.md_fs_param_for_each_level[i][j].epsilons = Smart64FArray(6, (double[]){0.35, 0.3, 0.125, 0.3, 0.35, 0.0778});
////			ml_md_fs_param.md_fs_param_for_each_level[i][j].degree = 1;
////			ml_md_fs_param.md_fs_param_for_each_level[i][j].folds = SmartIntArray(6, (int[]){4,4,4,4,4,4});
////			ml_md_fs_param.md_fs_param_for_each_level[i][j].opt = "sincos";
//		}
//		ml_md_fs_param.lowpass_approx_ds_folds[i] = SmartIntArray(nd, (int[]){2,2,2});
//	}
//
//	for (int i = 0; i < nnoises; ++i)
//	{
//		for (int j = 0; j < nimgs - 2; ++j)
//		{
//			Mat this_origin_img;
//			load_as_tensor(img_names[j], this_origin_img, &mfmt);
//
//			Mat channels[2];
//			channels[0] = Mat(this_origin_img.dims, this_origin_img.size, CV_64FC1);
//			channels[1] = Mat(this_origin_img.dims, this_origin_img.size, CV_64FC1, Scalar(0,0));
//			randn(channels[0], mean[i], stddev[i]);
//			Mat noise;
//			merge(channels, 2, noise);
//
//			Mat noisy_img = this_origin_img + noise;
////			Mat noisy_img = this_origin_img;
//			Mat noisy_img_ext;
//			mat_border_extension(noisy_img, bder_size.len, (const int*)bder_size, "mir1001", noisy_img_ext);
//			noisy_img_ext = noisy_img.clone();
//
////			ML_MD_Filter_System ml_md_fs;
//			Each_Channel_Filter_Norms_Set norms_set;
//			ML_MChannel_Coefs_Set coefs_set;
//			decompose_by_ml_md_filter_bank(ml_md_fs_param, noisy_img_ext, norms_set, coefs_set);
//
//			cout << " Engery: " << endl;
//			for (int i = 0; i < norms_set.size(); ++i)
//			{
//				for (int j = 0; j < norms_set[i].size(); ++j)
//				{
//					cout << norms_set[i][j] << " ";
//				}
//				cout << endl;
//			}
//			cout << endl;
//
//
//			ML_MChannel_Coefs_Set new_coefs;
//			normalize_coefs(coefs_set, norms_set, true, coefs_set);
//			thresholding(coefs_set, 7, 1, stddev[i], new_coefs);
////			bishrink(coefs_set, 7, sqrt(3), stddev[i], new_coefs);
//			normalize_coefs(new_coefs, norms_set, false, new_coefs);
//
//			Mat rec;
//			reconstruct_by_ml_md_filter_bank(ml_md_fs_param, new_coefs, rec);
//
//			mat_border_cut(rec, bder_size.len, (const int*)bder_size, rec);
//
//
//			stringstream ss;
//			Media_Format mfm;
//			ss << img_names[j] << "-noisy-"  << i << ".png";
//			save_as_media(ss.str(), noisy_img_ext, &mfm);
//
//			stringstream ss1;
//			ss1 << img_names[j] << "-rec-" << i << ".png";
//			save_as_media(ss1.str(), rec, &mfm);
//
//			double score, msr;
//			psnr(this_origin_img, noisy_img, score, msr);
//			cout << "PSNR: score: "<<score << ", msr: " << msr << endl;
//			psnr(this_origin_img, rec, score, msr);
//			cout << "PSNR: score: "<<score << ", msr: " << msr << endl;
//
//
//		}
//	}
//
//	return 0;
//}


