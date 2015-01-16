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

int denoise_entry(const Configuration *cfg, const string noisy_file)
{
typedef double FLOAT_TYPE;

	string param_scope("default");

	int nlevels = cfg->lookupInt(param_scope.c_str(), "nlevels");
	string fs_opt( cfg->lookupString(param_scope.c_str(), "fs") );
	int ext_size = cfg->lookupInt(param_scope.c_str(), "ext_size");
	string ext_method( cfg->lookupString(param_scope.c_str(), "ext_method") );
//	string mat_file ( cfg->lookupString(param_scope.c_str(), "f") );
	int ndims = 0;
	bool is_sym = cfg->lookupBoolean(param_scope.c_str(), "is_sym");

	Mat_<Vec<FLOAT_TYPE, 2> > noisy_mat;
	Media_Format mfmt;
	load_as_tensor<FLOAT_TYPE>(noisy_file, noisy_mat, &mfmt);
	ndims = noisy_mat.dims;

	ML_MD_FS_Param ml_md_fs_param;
	int ret = compose_fs_param(nlevels, ndims, fs_opt, ext_size, ext_method, is_sym, ml_md_fs_param);
	if (ret)
	{
		cout << "Error in FS param. " << endl;
		return 0;
	}

	cout << "Dec-Rec Paramters: " << endl;
	cout << "  nlevels: " << nlevels << endl;
	cout << "  ndims: " << ndims << endl;
	cout << "  fs_opt: " << fs_opt << endl;
	cout << "  ext_size: " << ext_size << endl;
	cout << "  ext_method: " << ext_method << endl;
	cout << "  is_sym: " << is_sym << endl;

	double mean = static_cast<double>(cfg->lookupFloat(param_scope.c_str(), "mean"));
	double stdev = static_cast<double>(cfg->lookupFloat(param_scope.c_str(), "stdev"));
	double c = static_cast<double>(cfg->lookupFloat(param_scope.c_str(), "c"));
	int wwidth = cfg->lookupInt(param_scope.c_str(), "wwidth");
	bool doNorm = cfg->lookupBoolean(param_scope.c_str(), "doNorm");
	string thr_method( cfg->lookupString(param_scope.c_str(), "thr_method") );

	Thresholding_Param thr_param;
	ret = compose_thr_param(mean, stdev, c, wwidth, doNorm, thr_method, thr_param);
	if (ret)
	{
		cout << "Error in Thr param. " << endl;
		return 0;
	}

	cout << endl << "Thresholding Parameters: " << endl;
	cout << "  mean: " << mean << endl;
	cout << "  stdev: " << stdev << endl;
	cout << "  c: " << c << endl;
	cout << "  wwidth: " << wwidth << endl;
	cout << "  doNorm: " << doNorm << endl;
	cout << "  thr_method: " << thr_method << endl;

	Mat_<Vec<FLOAT_TYPE, 2> > denoised_mat;
	thresholding_denoise<FLOAT_TYPE>(noisy_mat, ml_md_fs_param, thr_param, denoised_mat);

	bool doSave = cfg->lookupBoolean(param_scope.c_str(), "doSave");
	if (doSave)
	{
		string denoised_file = noisy_file + "-denoised.avi";
		save_as_media<FLOAT_TYPE>(denoised_file, denoised_mat, &mfmt);
	}

	return 0;
}

int psnr_entry(const string &left, const string &right)
{
	typedef double FLOAT_TYPE;

	Mat_<Vec<FLOAT_TYPE, 2> > left_mat, right_mat;
	Media_Format mfmt;
	load_as_tensor<FLOAT_TYPE>(left, left_mat, &mfmt);
	load_as_tensor<FLOAT_TYPE>(right, right_mat, &mfmt);

	double score, msr;
	psnr<FLOAT_TYPE>(left_mat, right_mat, score, msr);

	cout << "PSNR: " << score << ", MSR: " << msr << endl;

	return 0;
}


