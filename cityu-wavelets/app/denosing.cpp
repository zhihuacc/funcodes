#include "../include/wavelets_tools.h"
#include "../include/app.h"

#include <opencv2/imgproc/imgproc.hpp>

int thresholding(const ML_MChannel_Coefs_Set &coefs_set, int wwidth, double c, double sigma, ML_MChannel_Coefs_Set &thr_set)
{
	SmartIntArray winsize(2, (const int *)(int[]){wwidth, wwidth});
	Mat avg_window(winsize.len, (const int *)winsize, CV_64FC2, Scalar(1.0 / (wwidth * wwidth),0));


	thr_set.reserve(coefs_set.size());
	thr_set.resize(coefs_set.size());
	for (int cur_lvl = 0; cur_lvl < (int)coefs_set.size(); ++cur_lvl)
	{
		const vector<Mat> &this_level_set = coefs_set[cur_lvl];
		thr_set[cur_lvl].reserve(this_level_set.size());
		thr_set[cur_lvl].resize(this_level_set.size());
		for (int idx = 0; idx < (int)this_level_set.size(); ++idx)
		{
			Mat Y = this_level_set[idx].clone();
			Mat Y_abs;
			Mat T;
			Mat T_abs;
			Mat good_to_div;
			Mat mini_mat(Y.dims, Y.size, CV_64FC2, Scalar(1e-16, 0));
			pw_abs(Y, Y_abs);
			pw_pow(Y_abs, 2, T);
			blur(T, T, Size(wwidth,wwidth), Point(-1,-1), BORDER_REFLECT);

			T -= Scalar(sigma * sigma, 0);
			pw_max(T, mini_mat, good_to_div);
			pw_sqrt(good_to_div, T);
			pw_reciprocal(T, T, complex<double>(c * sigma * sigma, 0));
			// T is ready and real matrix

//			pw_abs(T, T_abs);
			Mat mask;
			pw_less(T, Y_abs, mask);

//			pw_max(Y_abs, mini_mat, good_to_div);
			pw_reciprocal(Y_abs, good_to_div);
			Mat ratio;
			pw_mul(T, good_to_div, ratio);
			ratio = Scalar(1.0,0) - ratio;
			// ratio is ready

			Mat new_Y;
			pw_mul(Y, ratio, new_Y);
			pw_mul(mask, new_Y, new_Y);

			thr_set[cur_lvl][idx] = new_Y;
		}
	}
	return 0;
}

int app_denoising()
{
	const int nnoises = 3;
	const int nimgs = 3;

	double mean[nnoises] = {0,0,0};
	double stddev[nnoises] = {5, 10, 15};

	Mat origin_imgs[nimgs];
	string img_names[nimgs] = {"Test-Data/Lena512.png", "Test-Data/hill.png", "Test-Data/fingerprint.png"};

	Media_Format mfmt;
	load_as_tensor("Test-Data/Lena512.png", origin_imgs[0], &mfmt);
	load_as_tensor("Test-Data/hill.png", origin_imgs[1], &mfmt);
	load_as_tensor("Test-Data/fingerprint.png", origin_imgs[2], &mfmt);

	SmartIntArray bder_size(origin_imgs[0].dims);
	for (int i = 0; i < bder_size.len; ++i)
	{
		bder_size[i] = 64;
	}

	int nlevels = 2;
	int nd = 2;
	MLevel_MDFilter_System_Param ml_md_fs_param;
	ml_md_fs_param.md_fs_param_for_each_level.reserve(nlevels);
//	ml_md_fs_param.md_fs_param_for_each_level[1].reserve(2);
	ml_md_fs_param.lowpass_approx_ds_folds.reserve(nlevels);
//	ml_md_fs_param.md_fs_param_for_each_level[2].reserve(2);
	for (int i = 0; i < ml_md_fs_param.md_fs_param_for_each_level.len; ++i)
	{
		ml_md_fs_param.md_fs_param_for_each_level[i].reserve(nd);
		for (int j = 0; j < ml_md_fs_param.md_fs_param_for_each_level[i].len; ++j)
		{
			ml_md_fs_param.md_fs_param_for_each_level[i][j].ctrl_points = Smart64FArray(3, (double[]){-33.0/32.0, 33.0/32.0, M_PI});
			ml_md_fs_param.md_fs_param_for_each_level[i][j].epsilons = Smart64FArray(3, (double[]){69.0/128.0, 69.0/128.0, 51.0/512.0});
			ml_md_fs_param.md_fs_param_for_each_level[i][j].degree = 1;
			ml_md_fs_param.md_fs_param_for_each_level[i][j].folds = SmartIntArray(3, (int[]){2,2,2});
			ml_md_fs_param.md_fs_param_for_each_level[i][j].opt = "sincos";
		}
		ml_md_fs_param.lowpass_approx_ds_folds[i] = SmartIntArray(2, (int[]){2,2});
	}

	for (int i = 0; i < nnoises; ++i)
	{
		for (int j = 0; j < nimgs - 2; ++j)
		{
			Mat &this_origin_img = origin_imgs[j];

			Mat channels[2];
			channels[0] = Mat(this_origin_img.dims, this_origin_img.size, CV_64FC1);
			channels[1] = Mat(this_origin_img.dims, this_origin_img.size, CV_64FC1, Scalar(0,0));
			randn(channels[0], mean[i], stddev[i]);
			Mat noise;
			merge(channels, 2, noise);

			Mat noisy_img = this_origin_img + noise;
//			Mat noisy_img = this_origin_img;
			Mat nosiy_img_ext;
			mat_border_extension(noisy_img, bder_size.len, (const int*)bder_size, "mir1001", nosiy_img_ext);


			ML_MD_Filter_System ml_md_fs;
			ML_MChannel_Coefs_Set coefs_set;
			decompose_by_ml_md_filter_bank(ml_md_fs_param, nosiy_img_ext, ml_md_fs, coefs_set);

			ML_MChannel_Coefs_Set new_coefs;
			thresholding(coefs_set, 7, 1, stddev[i], new_coefs);

			Mat rec;
			reconstruct_by_ml_md_filter_bank(ml_md_fs_param, new_coefs, rec);

			mat_border_cut(rec, bder_size.len, (const int*)bder_size, rec);


			stringstream ss;
			Media_Format mfm;
			ss << img_names[j] << "-noisy-"  << i << ".png";
			save_as_media(ss.str(), nosiy_img_ext, &mfm);

			stringstream ss1;
			ss1 << img_names[j] << "-rec-" << i << ".png";
			save_as_media(ss1.str(), rec, &mfm);

			double score, msr;
			psnr(this_origin_img, noisy_img, score, msr);
			cout << "PSNR: score: "<<score << ", msr: " << msr << endl;
			psnr(this_origin_img, rec, score, msr);
			cout << "PSNR: score: "<<score << ", msr: " << msr << endl;


		}
	}

	return 0;
}


