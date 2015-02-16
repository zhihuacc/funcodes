#ifndef _COMMANDS_H
#define _COMMANDS_H

#include <string>
#include <config4cpp/Configuration.h>

#include "wavelets_toolbox.h"
#include "denoising.h"

using namespace std;
using namespace config4cpp;

//struct Inpainting_Param
//{
//	int 	sigma_num;
//	vector<double> sigmas;
//	int 	phase1_end;
//	double 	epsilon1;
//	double 	epsilon2;
//	double 	estimated_sigma;
//	int 	max_iter;
//	vector<ML_MD_FS_Param> fs_params;
//	vector<Thresholding_Param> thr_params;
//	int 	ext_size;
//	string 	ext_method;
//};

int cvtxml_entry(const string &fn);
int psnr_entry(const string &left, const string &right);
//int inpaint(const Configuration *cfg, const string &img_name, const string &mask_name);
//int batch_inpaint_entry(const Configuration *cfg, const string &top_scope);
//
//template<typename _Tp>
//int inpaint(const Mat_<Vec<_Tp, 2> > &mat, const Mat_<Vec<_Tp, 2> > &mask, const Inpainting_Param &param, Mat_<Vec<_Tp, 2> > &res)
//{
//	Mat_<Vec<_Tp, 2> > noisy_masked_mat, rmask;
//	int ndims = mat.dims;
//
//	noisy_masked_mat = mat.mul(mask);
//	double py = l2norm<_Tp>(noisy_masked_mat);
//
//	SmartIntArray border(ndims, param.ext_size);
//	mat_border_extension(noisy_masked_mat, border, param.ext_method, noisy_masked_mat);
//
//	rmask = Scalar(1.0,0) - mask;
//	mat_border_extension(rmask, border, param.ext_method, rmask);
//
//	Mat_<Vec<_Tp, 2> > last_restored, merged_restored,
//				       cur_restored(ndims, noisy_masked_mat.size, Vec<_Tp, 2>(0,0));
//
//	int sigma_id = 0;
//	merged_restored = rmask.mul(cur_restored) + noisy_masked_mat;
//	last_restored = cur_restored;
//	for (int iter = 0; iter < param.max_iter && sigma_id < param.sigma_num; ++iter)
//	{
//		cout << endl << "Iter ********** " << iter << ", sigma_id " << sigma_id << endl;
//
//		thresholding_denoise<_Tp>(merged_restored, param.fs_params[sigma_id], param.thr_params[sigma_id], cur_restored);
//		pw_abs<_Tp>(cur_restored, cur_restored);
//
////			stringstream ss;
////		ss << "Test-Data/output/merged_" << sigma_id << "-" << iter << ".txt";
////		print_mat_details_g<FLOAT_TYPE, 2>(merged_restored, 2, ss.str());
//
//		merged_restored = rmask.mul(cur_restored) + noisy_masked_mat;
//
////		psnr<_Tp>(clean_mat, merged_restored, score, msr);
////		cout << "sigma: " << stdev[sigma_id] << ", psnr: " << score << ", msr: " << msr << endl;
//
////			ss.str("");
////			ss << "Test-Data/output/merged_restored_" << sigma_id << "-" << iter << ".png";
////			save_as_media<FLOAT_TYPE>(ss.str(), merged_restored, &mfmt);
////			print_mat_details_g<FLOAT_TYPE, 2>(cur_restored, 2, ss.str());//		save_as_media<FLOAT_TYPE>(ss.str(), merged_restored, &mfmt);
//
//		//-- test error
//
//		double err = l2norm<_Tp>(rmask.mul(last_restored - cur_restored));
//		last_restored = cur_restored;
//
//		cout << "py: " << py << ", err: " << err;
//		err /= py;
//		cout << ", err/py: " << err << endl;
//		if (sigma_id < param.phase1_end)
//		{
//			if (err < param.epsilon1)
//			{
//				++sigma_id;
//			}
//		}
//		else if (param.phase1_end <= sigma_id && sigma_id < param.sigma_num)
//		{
//			if (err < param.epsilon2)
//			{
//				++sigma_id;
//			}
//		}
//		//--
//	}
//
//	if (fabs(param.estimated_sigma - 0) < numeric_limits<_Tp>::epsilon())
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
//
//	res = cur_restored;
//	return 0;
//}
//
//template<typename _Tp>
//int batch_inpaint(const Configuration *cfg, const string &top_scope)
//{
//	Inpainting_Param param;
//
//	int ret = 0;
////	int ndims = 0;
//	const char **fnames, **fmasks;
//	int fnum, fnum2;
////	param.ndims = cfg->lookupInt(top_scope.c_str(), "ndims");
//	cfg->lookupList(top_scope.c_str(), "fnames", fnames, fnum);
//	cfg->lookupList(top_scope.c_str(), "masks", fmasks, fnum2);
//	if (fnum != fnum2)
//	{
//		cout << "Mat and Mask numbers don't math. " << endl;
//		return 0;
//	}
//
//
//	const char **sigma_list, **param_list;
//	int sigma_num, param_num;
//	cfg->lookupList(top_scope.c_str(), "sigmas", sigma_list, sigma_num);
//	cfg->lookupList(top_scope.c_str(), "params", param_list, param_num);
//	if (sigma_num != param_num)
//	{
//		return 0;
//	}
//
//	param.sigma_num = sigma_num;
//	param.sigmas.reserve(sigma_num);
//	param.sigmas.resize(sigma_num);
//	param.fs_params.reserve(sigma_num);
//	param.fs_params.resize(sigma_num);
//	param.thr_params.reserve(sigma_num);
//	param.thr_params.resize(sigma_num);
//	for (int i = 0; i < param_num; ++i)
//	{
//		stringstream ss;
//		ss << "sigmas[" << i + 1 << "]";
//		param.sigmas[i] = cfg->stringToFloat("", ss.str().c_str(), sigma_list[i]);
//	}
//
//	param.max_iter = cfg->lookupInt(top_scope.c_str(), "max_iter");
//	param.phase1_end = cfg->lookupInt(top_scope.c_str(), "phase1_end");
//	param.epsilon1 = cfg->lookupFloat(top_scope.c_str(), "epsilon1");
//	param.epsilon2 = cfg->lookupFloat(top_scope.c_str(), "epsilon2");
//	param.estimated_sigma = cfg->lookupFloat(top_scope.c_str(), "noise_sigma");
//
//	param.ext_size = cfg->lookupInt(top_scope.c_str(), "ext_size");
//	param.ext_method = cfg->lookupString(top_scope.c_str(), "ext_method");
//
//	for (int f = 0; f < fnum; ++f)
//	{
//		string fname(fnames[f]);
//		string fmask(fmasks[f]);
//
//		Media_Format mfmt;
//		Mat_<Vec<_Tp, 2> > clean_mat, noisy_masked_mat, mask, res;
//		load_as_tensor<_Tp>(fname, clean_mat, &mfmt);
//		load_as_tensor<_Tp>(fmask, mask, &mfmt);
//		mask /= 255;
//
//		int ndims = clean_mat.dims;
//		for (int i = 0; i < param_num; ++i)
//		{
//			string param_scope = top_scope + "." + param_list[i];
//
//			int nlevels = cfg->lookupInt(param_scope.c_str(), "nlevels");
//			string fs_opt( cfg->lookupString(param_scope.c_str(), "fs") );
//			bool is_sym = cfg->lookupBoolean(param_scope.c_str(), "is_sym");
//
//			ret = compose_fs_param(nlevels, ndims, fs_opt, 0, "none", is_sym, param.fs_params[i]);
//			if (ret)
//			{
//				cout << "Error in FS param: " << i << endl;
//				return 0;
//			}
//
//			double c = static_cast<double>(cfg->lookupFloat(param_scope.c_str(), "c"));
//			int wwidth = cfg->lookupInt(param_scope.c_str(), "wwidth");
//			bool doNorm = cfg->lookupBoolean(param_scope.c_str(), "doNorm");
//			string thr_method( cfg->lookupString(param_scope.c_str(), "thr_method") );
//
//			ret = compose_thr_param(0, param.sigmas[i], c, wwidth, doNorm, thr_method, param.thr_params[i]);
//			if (ret)
//			{
//				cout << "Error in Thr param: " << i << endl;
//				return 0;
//			}
//		}
//
//		//-- Fake up noisy data.
//		Mat_<Vec<_Tp, 1> > channels[2];
//		channels[0] = Mat_<Vec<_Tp, 1> >(ndims, clean_mat.size);
//		channels[1] = Mat_<Vec<_Tp, 1> >(ndims, clean_mat.size, Vec<_Tp, 1>((_Tp)0));
//		randn(channels[0], 0, param.estimated_sigma);
//		merge(channels, 2, noisy_masked_mat);
//		channels[0].release();
//		channels[1].release();
//
////		load_as_tensor<FLOAT_TYPE>("Test-Data/nnoise90-512.png", noisy_masked_mat, &mfmt);
//		write_mat_dat<_Tp, 2>(noisy_masked_mat, "Test-Data/output/noises90.dat");
//		noisy_masked_mat += clean_mat;
//		// --
//
//		inpaint<_Tp>(noisy_masked_mat, mask, param, res);
//
//		save_as_media<_Tp>("Test-Data/output/restored.png", res, &mfmt);
//
//		double score, msr;
//		psnr<_Tp>(clean_mat, res, score, msr);
//		cout << endl << "Done sigma: " << ", psnr: " << score << ", msr: " << msr << endl;
//
//	}
//	return 0;
//}

#endif
