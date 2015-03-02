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

//int denoise_entry(const Configuration *cfg, const string &top_scope)
//{
//typedef double FLOAT_TYPE;
//
//	const char **fnames;
//	int fnum;
//	cfg->lookupList(top_scope.c_str(), "fnames", fnames, fnum);
//
//	int nlevels = cfg->lookupInt(top_scope.c_str(), "nlevels");
//	string fs_opt( cfg->lookupString(top_scope.c_str(), "fs") );
//	int ext_size = cfg->lookupInt(top_scope.c_str(), "ext_size");
//	string ext_method( cfg->lookupString(top_scope.c_str(), "ext_method") );
////	string mat_file ( cfg->lookupString(param_scope.c_str(), "f") );
//
//	bool is_sym = cfg->lookupBoolean(top_scope.c_str(), "is_sym");
//	int ndims = cfg->lookupInt(top_scope.c_str(), "ndims");
//
//	Mat_<Vec<FLOAT_TYPE, 2> > noisy_mat;
//	Media_Format mfmt;
//
//	ML_MD_FS_Param ml_md_fs_param;
//	int ret = compose_fs_param(nlevels, ndims, fs_opt, ext_size, ext_method, is_sym, ml_md_fs_param);
//	if (ret)
//	{
//		cout << "Error in FS param. " << endl;
//		return 0;
//	}
//
//	cout << "Dec-Rec Paramters: " << endl;
//	cout << "  nlevels: " << nlevels << endl;
//	cout << "  ndims: " << ndims << endl;
//	cout << "  fs_opt: " << fs_opt << endl;
//	cout << "  ext_size: " << ext_size << endl;
//	cout << "  ext_method: " << ext_method << endl;
//	cout << "  is_sym: " << is_sym << endl;
//
//	double mean = static_cast<double>(cfg->lookupFloat(top_scope.c_str(), "mean"));
//	double stdev = static_cast<double>(cfg->lookupFloat(top_scope.c_str(), "stdev"));
//	double c = static_cast<double>(cfg->lookupFloat(top_scope.c_str(), "c"));
//	int wwidth = cfg->lookupInt(top_scope.c_str(), "wwidth");
//	bool doNorm = cfg->lookupBoolean(top_scope.c_str(), "doNorm");
//	string thr_method( cfg->lookupString(top_scope.c_str(), "thr_method") );
//
//	Thresholding_Param thr_param;
//	ret = compose_thr_param(mean, stdev, c, wwidth, doNorm, thr_method, thr_param);
//	if (ret)
//	{
//		cout << "Error in Thr param. " << endl;
//		return 0;
//	}
//
//	cout << endl << "Thresholding Parameters: " << endl;
//	cout << "  mean: " << mean << endl;
//	cout << "  stdev: " << stdev << endl;
//	cout << "  c: " << c << endl;
//	cout << "  wwidth: " << wwidth << endl;
//	cout << "  doNorm: " << doNorm << endl;
//	cout << "  thr_method: " << thr_method << endl;
//
//
//	for (int f = 0; f < fnum; ++f)
//	{
//		string fname(fnames[f]);
//
//		Mat_<Vec<FLOAT_TYPE, 2> > clean_mat, noisy_mat;
//		Mat_<Vec<FLOAT_TYPE, 1> > channels[2];
//
//		load_as_tensor<FLOAT_TYPE>(fname, clean_mat, &mfmt);
//
//			// Fake up noises.
//		channels[0] = Mat_<Vec<FLOAT_TYPE, 1> >(clean_mat.dims, clean_mat.size);
//		channels[1] = Mat_<Vec<FLOAT_TYPE, 1> >(clean_mat.dims, clean_mat.size, Vec<FLOAT_TYPE, 1>((FLOAT_TYPE)0));
//		randn(channels[0], mean, stdev);
//		merge(channels, 2, noisy_mat);
//		channels[0].release();
//		channels[1].release();
//
////		load_as_tensor<FLOAT_TYPE>("Test-Data/nnoise90-512.png", noisy_mat, &mfmt);
////		write_mat_dat<FLOAT_TYPE, 2>(noisy_mat, "Test-Data/output/noises90.dat");
//		noisy_mat += clean_mat;
//
//
//		double score, msr;
//		psnr<FLOAT_TYPE>(noisy_mat, clean_mat, score, msr);
//		cout << "Denosing " << fname << " Start: psnr: " << score << ", msr: " << msr << endl;
//
//		Mat_<Vec<FLOAT_TYPE, 2> > denoised_mat;
//		thresholding_denoise<FLOAT_TYPE>(noisy_mat, ml_md_fs_param, thr_param, denoised_mat);
//
//		bool doSave = cfg->lookupBoolean(top_scope.c_str(), "doSave");
//		if (doSave)
//		{
//			string denoised_file = fname + "_denoised_" + mfmt.suffix;
//			save_as_media<FLOAT_TYPE>(denoised_file, denoised_mat, &mfmt);
//		}
//
//
//		psnr<FLOAT_TYPE>(denoised_mat, clean_mat, score, msr);
//		cout << "Denoising " << fname << " Done: psnr: " << score << ", msr: " << msr << endl;
//	}
//
//	return 0;
//}


//
//int denoising_demo(const string &fn)
//{
//typedef double FLOAT_TYPE;
//	int ret = 0;
//	Media_Format mfmt;
//	Mat_<Vec<FLOAT_TYPE, 2> > input, noisy_input, denoised_output;
//	load_as_tensor<FLOAT_TYPE>(fn, input, &mfmt);
//	int ndims = input.dims;
//
//	//-- Fake up noisy data.
//	double mean = 0;
//	double stdev = 5;
//	Mat_<Vec<FLOAT_TYPE, 1> > channels[2];
//	channels[0] = Mat_<Vec<FLOAT_TYPE, 1> >(input.dims, input.size);
//	channels[1] = Mat_<Vec<FLOAT_TYPE, 1> >(input.dims, input.size, Vec<FLOAT_TYPE, 1>((FLOAT_TYPE)0));
//	randn(channels[0], mean, stdev);
//	merge(channels, 2, noisy_input);
//	channels[0].release();
//	channels[1].release();
//	noisy_input = input + noisy_input;
////	noisy_input = input;
//	// --
//
//	double score, msr;
//	psnr<FLOAT_TYPE>(input, noisy_input, score, msr);
//	cout << "Noisy Input PSNR score: " << score << ", msr: " << msr << endl << endl;
//
//	// -- Prepare parameters;
//	int nlevels = 2;
//	string fs_param_opt = "CTF3";
//	int ext_size = 64;
//	string ext_opt = "mir1001";
//	ML_MD_FS_Param ml_md_fs_param;
//	ret = compose_fs_param(nlevels, ndims, fs_param_opt, ext_size, ext_opt, false, ml_md_fs_param);
//	if (ret)
//	{
//		cout << "Error in FS param. " << endl;
//		return 0;
//	}
//
//
//	Thresholding_Param thr_param;
//	ret = compose_thr_param(mean, stdev, sqrt(3), 7, true, "bishrink", thr_param);
//	if (ret)
//	{
//		cout << "Error in Thr param. " << endl;
//		return 0;
//	}
//	// --
//
//	thresholding_denoise<FLOAT_TYPE>(noisy_input, ml_md_fs_param, thr_param, denoised_output);
//	noisy_input.release();
//
//
//	psnr<FLOAT_TYPE>(input, denoised_output, score, msr);
//	cout << "Denoised PSNR score: " << score << ", msr: " << msr << endl;
//
//	return 0;
//}



