#include "../include/mat_toolbox.h"
#include "../include/commands.h"
#include "../include/denoising.h"

using namespace std;

int cvtxml_entry(const string &fn)
{
	typedef double FLOAT_TYPE;

	Media_Format mfmt;
	Mat_<Vec<FLOAT_TYPE, 2> > mat;
	load_as_tensor<FLOAT_TYPE>(fn, mat, &mfmt);

	string xml = fn + ".xml";
	save_as_media<FLOAT_TYPE>(xml, mat, &mfmt);

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
//
//int batch_inpaint_entry(const Configuration *cfg, const string &top_scope)
//{
//	typedef double FLOAT_TYPE;
//
//	int ret = 0;
//	int ndims = 0;
//	const char **fnames, **fmasks;
//	int fnum, fnum2;
//	ndims = cfg->lookupInt(top_scope.c_str(), "ndims");
//	cfg->lookupList(top_scope.c_str(), "fnames", fnames, fnum);
//	cfg->lookupList(top_scope.c_str(), "masks", fmasks, fnum2);
//	if (fnum != fnum2)
//	{
//		cout << "Mat and Mask numbers don't math. " << endl;
//		return 0;
//	}
//
////	double sigma2 = cfg->lookupFloat("", "stdev[3]");
//
//	const char **stdev_list, **param_list;
//	int stdev_num, param_num;
//	cfg->lookupList(top_scope.c_str(), "stdev", stdev_list, stdev_num);
//	cfg->lookupList(top_scope.c_str(), "param", param_list, param_num);
//	if (stdev_num != param_num)
//	{
//		return 0;
//	}
//
//	vector<double> stdev(stdev_num);
//	for (int i = 0; i < param_num; ++i)
//	{
//		stringstream ss;
//		ss << "stdev[" << i + 1 << "]";
//		stdev[i] = cfg->stringToFloat("", ss.str().c_str(), stdev_list[i]);
//	}
//
//	int phase1_end = cfg->lookupInt(top_scope.c_str(), "phase1_end");
//	double epsilon1 = cfg->lookupFloat(top_scope.c_str(), "epsilon1");
//	double epsilon2 = cfg->lookupFloat(top_scope.c_str(), "epsilon2");
//	double noise_stdev = cfg->lookupFloat(top_scope.c_str(), "noise_stdev");
//	vector<ML_MD_FS_Param> fs_params(param_num);
//	vector<Thresholding_Param> thr_params(param_num);
//	for (int i = 0; i < param_num; ++i)
//	{
//		string param_scope = top_scope + "." + param_list[i];
//
//		int nlevels = cfg->lookupInt(param_scope.c_str(), "nlevels");
//		string fs_opt( cfg->lookupString(param_scope.c_str(), "fs") );
////		int ext_size = cfg->lookupInt(param_scope.c_str(), "ext_size");
////		string ext_method( cfg->lookupString(param_scope.c_str(), "ext_method") );
//		bool is_sym = cfg->lookupBoolean(param_scope.c_str(), "is_sym");
//
//		ret = compose_fs_param(nlevels, ndims, fs_opt, 0, "none", is_sym, fs_params[i]);
//		if (ret)
//		{
//			cout << "Error in FS param: " << i << endl;
//			return 0;
//		}
//
//		double c = static_cast<double>(cfg->lookupFloat(param_scope.c_str(), "c"));
//		int wwidth = cfg->lookupInt(param_scope.c_str(), "wwidth");
//		bool doNorm = cfg->lookupBoolean(param_scope.c_str(), "doNorm");
//		string thr_method( cfg->lookupString(param_scope.c_str(), "thr_method") );
//
//		ret = compose_thr_param(0, stdev[i], c, wwidth, doNorm, thr_method, thr_params[i]);
//		if (ret)
//		{
//			cout << "Error in Thr param: " << i << endl;
//			return 0;
//		}
//
//	}
//
//	int g_ext_size = cfg->lookupInt(top_scope.c_str(), "ext_size");
//	string g_ext_method( cfg->lookupString(top_scope.c_str(), "ext_method"));
//
//	for (int f = 0; f < fnum; ++f)
//	{
//		string fname(fnames[f]);
//		string fmask(fmasks[f]);
//
//		Media_Format mfmt;
//		Mat_<Vec<FLOAT_TYPE, 2> > clean_mat, noisy_masked_mat, mask;
//		load_as_tensor<FLOAT_TYPE>(fname, clean_mat, &mfmt);
//		load_as_tensor<FLOAT_TYPE>(fmask, mask, &mfmt);
//
//
//		mask /= 255;
//
//		//-- Fake up noisy data.
//		Mat_<Vec<FLOAT_TYPE, 1> > channels[2];
//		channels[0] = Mat_<Vec<FLOAT_TYPE, 1> >(clean_mat.dims, clean_mat.size);
//		channels[1] = Mat_<Vec<FLOAT_TYPE, 1> >(clean_mat.dims, clean_mat.size, Vec<FLOAT_TYPE, 1>((FLOAT_TYPE)0));
//		randn(channels[0], 0, noise_stdev);
//		merge(channels, 2, noisy_masked_mat);
//		channels[0].release();
//		channels[1].release();
//
////		load_as_tensor<FLOAT_TYPE>("Test-Data/nnoise90-512.png", noisy_masked_mat, &mfmt);
//		write_mat_dat<FLOAT_TYPE, 2>(noisy_masked_mat, "Test-Data/output/noises90.dat");
//
//		noisy_masked_mat += clean_mat;
//
//		// --
//
//		noisy_masked_mat = noisy_masked_mat.mul(mask);
////		clean_masked = clean_mat.mul(mask);
//		double py = l2norm<FLOAT_TYPE>(noisy_masked_mat);
//
//		SmartIntArray border(ndims, g_ext_size);
//		mat_border_extension(clean_mat, border, g_ext_method, clean_mat);
//		mat_border_extension(noisy_masked_mat, border, g_ext_method, noisy_masked_mat);
//		mat_border_extension(mask, border, g_ext_method, mask);
//		mask = Scalar(1.0,0) - mask;
//		// From now on, mask is reverse mask.
//
////		save_as_media<FLOAT_TYPE>("Test-Data/output/ext_clean_mat.png", clean_mat, &mfmt);
//
//		Mat_<Vec<FLOAT_TYPE, 2> > last_restored,
//					              merged_restored,
//					              cur_restored(ndims, noisy_masked_mat.size, Vec<FLOAT_TYPE, 2>(0,0));
//		double score, msr;
//		int sigma_id = 0;
//		merged_restored = mask.mul(cur_restored) + noisy_masked_mat;
//		last_restored = cur_restored;
//		for (int iter = 0; iter < 1000 && sigma_id < stdev_num; ++iter)
//		{
//			cout << endl << "Iter ********** " << iter << ", sigma_id " << sigma_id << endl;
//
//			thresholding_denoise<FLOAT_TYPE>(merged_restored, fs_params[sigma_id], thr_params[sigma_id], cur_restored);
//			pw_abs<FLOAT_TYPE>(cur_restored, cur_restored);
//
////			stringstream ss;
//	//		ss << "Test-Data/output/merged_" << sigma_id << "-" << iter << ".txt";
//	//		print_mat_details_g<FLOAT_TYPE, 2>(merged_restored, 2, ss.str());
//
//			merged_restored = mask.mul(cur_restored) + noisy_masked_mat;
//
//			psnr<FLOAT_TYPE>(clean_mat, merged_restored, score, msr);
//			cout << "sigma: " << stdev[sigma_id] << ", psnr: " << score << ", msr: " << msr << endl;
//
////			ss.str("");
////			ss << "Test-Data/output/merged_restored_" << sigma_id << "-" << iter << ".png";
////			save_as_media<FLOAT_TYPE>(ss.str(), merged_restored, &mfmt);
////			print_mat_details_g<FLOAT_TYPE, 2>(cur_restored, 2, ss.str());//		save_as_media<FLOAT_TYPE>(ss.str(), merged_restored, &mfmt);
//
//			//-- test error
//
//			double err = l2norm<FLOAT_TYPE>(mask.mul(last_restored - cur_restored));
//			last_restored = cur_restored;
//
//			cout << "py: " << py << ", err: " << err;
//			err /= py;
//			cout << ", err/py: " << err << endl;
//			if (sigma_id < phase1_end)
//			{
//				if (err < epsilon1)
//				{
//					++sigma_id;
//				}
//			}
//			else if (phase1_end <= sigma_id && sigma_id < stdev_num)
//			{
//				if (err < epsilon2)
//				{
//					++sigma_id;
//				}
//			}
//			//--
//		}
//
////		mat_border_extension<FLOAT_TYPE>(merged_restored, border, "cut", merged_restored);
//
//		mat_border_extension<FLOAT_TYPE>(clean_mat, border, "cut", clean_mat);
//		if (fabs(noise_stdev - 0) < numeric_limits<FLOAT_TYPE>::epsilon())
//		{
//			mat_border_extension<FLOAT_TYPE>(cur_restored, border, "cut", cur_restored);
//			mat_border_extension<FLOAT_TYPE>(mask, border, "cut", mask);
//			cur_restored = mask.mul(cur_restored);
//			mask = Scalar(1.0,0) - mask;
//			cur_restored += clean_mat.mul(mask);
//		}
//		else
//		{
//			mat_border_extension<FLOAT_TYPE>(cur_restored, border, "cut", cur_restored);
//		}
//
////		save_as_media<FLOAT_TYPE>("Test-Data/output/Lena512-restored.png", cur_restored, &mfmt);
//		psnr<FLOAT_TYPE>(clean_mat, cur_restored, score, msr);
//		cout << endl << "Done sigma: " << stdev[sigma_id - 1] << ", psnr: " << score << ", msr: " << msr << endl;
//
//	}
//	return 0;
//}
//
//
//
//
//int inpaint(const Configuration *cfg, const string &img_name, const string &mask_name)
//{
//	typedef double FLOAT_TYPE;
//
//	int ret = 0;
//	Media_Format mfmt;
//	Mat_<Vec<FLOAT_TYPE, 2> > clean_mat, noisy_masked_mat, noisy_masked_mat2, mask, masked_mat, ext_rmask, ext_clean_mat, ext_noisy_masked_mat;
//	load_as_tensor<FLOAT_TYPE>(img_name, clean_mat, &mfmt);
//	load_as_tensor<FLOAT_TYPE>(mask_name, mask, &mfmt);
//
//
//	mask /= 255;
//	int ndims = clean_mat.dims;
//
//	//-- Fake up noisy data.
//	double mean = 0;
//	double noise_stdev = 5;
//	Mat_<Vec<FLOAT_TYPE, 1> > channels[2];
//	channels[0] = Mat_<Vec<FLOAT_TYPE, 1> >(clean_mat.dims, clean_mat.size);
//	channels[1] = Mat_<Vec<FLOAT_TYPE, 1> >(clean_mat.dims, clean_mat.size, Vec<FLOAT_TYPE, 1>((FLOAT_TYPE)0));
//	randn(channels[0], mean, noise_stdev);
//	merge(channels, 2, noisy_masked_mat);
//	channels[0].release();
//	channels[1].release();
////	noisy_masked_mat = clean_mat + noisy_masked_mat;
//	noisy_masked_mat = clean_mat.clone();
//	// --
//
//
//    // noisy_masked_mat is real-world input.
//	noisy_masked_mat = noisy_masked_mat.mul(mask);
//
//	double py = l2norm<FLOAT_TYPE>(noisy_masked_mat);
//
//	SmartIntArray border(2, 16);
//	mat_border_extension(clean_mat, border, "mir1001", ext_clean_mat);
//	mat_border_extension(noisy_masked_mat, border, "mir1001", ext_noisy_masked_mat);
//	mat_border_extension(mask, border, "mir1001", ext_rmask);
//	ext_rmask = Scalar(1.0,0) - ext_rmask;
//	mask.release();
//
////	print_mat_details_g<FLOAT_TYPE>(ext_noisy_masked_mat, 2, "Test-Data/output/ext_noisy_masked.txt");
//
//	double ep1 =5e-3, ep2 =  1e-4;
//	int phase1_end = 5;
////	double stdev[] = {100, 92, 84, 76, 68, 60, 52.63, 44.36, 36.53, 28.35,
////			           24.613, 20.513, 16.413, 12.343, 10.243, 8.143, 6, 4, 2, 1};
//	double stdev[] = {512.0000,  227.6198,  101.1929,   44.9873,   20.0000,
//			 15.0000,   10.1877,    6.9193,    4.6995,    3.1918,    2.1678,    1.4724,    1.0000};
////	string fs[] = {"CTF3", "CTF3", "CTF3", "CTF3", "CTF3", "CTF3", "CTF3", "CTF3", "CTF3", "CTF3",
////			       "CTF3", "CTF3", "CTF3", "CTF3", "CTF3", "CTF3", "CTF3", "CTF3", "CTF3", "CTF3"};
//	string fs[] = {"CTF3", "CTF3", "CTF3", "CTF3", "CTF3",
//			       "CTF3", "CTF3", "CTF3", "CTF3", "CTF3", "CTF3", "CTF3", "CTF3"};
//	int sigma_num = sizeof(stdev) / sizeof(double);
//
//	Mat_<Vec<FLOAT_TYPE, 2> > last_restored,
//			                  merged_restored,
//			                  cur_restored(ndims, ext_noisy_masked_mat.size, Vec<FLOAT_TYPE, 2>(0,0));
//	double score, msr;
//	int sigma_id = 0;
//	merged_restored = ext_rmask.mul(cur_restored) + ext_noisy_masked_mat;
//	last_restored = cur_restored;
//	for (int iter = 0; iter < 1000 && sigma_id < sigma_num; ++iter)
//	{
//		cout << endl << "Iter ********** " << iter << ", sigma_id " << sigma_id << endl;
//
//		// -- Prepare parameters;
//		int nlevels = 2;
//		int ext_size = 0;
//		string ext_opt = "none";
//		ML_MD_FS_Param ml_md_fs_param;
//		ret = compose_fs_param(nlevels, ndims, fs[sigma_id], ext_size, ext_opt, true, ml_md_fs_param);
//		if (ret)
//		{
//			cout << "Error in FS param. " << endl;
//			return 0;
//		}
//
//
//		Thresholding_Param thr_param;
//		ret = compose_thr_param(0, stdev[sigma_id], sqrt(3), 7, true, "bishrink", thr_param);
//		if (ret)
//		{
//			cout << "Error in Thr param. " << endl;
//			return 0;
//		}
//		// --
//
//		thresholding_denoise<FLOAT_TYPE>(merged_restored, ml_md_fs_param, thr_param, cur_restored);
//
//		pw_abs<FLOAT_TYPE>(cur_restored, cur_restored);
//
////		stringstream ss;
////		ss << "Test-Data/output/merged_" << sigma_id << "-" << iter << ".txt";
////		print_mat_details_g<FLOAT_TYPE, 2>(merged_restored, 2, ss.str());
//
//		merged_restored = ext_rmask.mul(cur_restored) + ext_noisy_masked_mat;
//
//		psnr<FLOAT_TYPE>(ext_clean_mat, merged_restored, score, msr);
//		cout << "sigma: " << stdev[sigma_id] << ", psnr: " << score << ", msr: " << msr << endl;
//
////		ss.str("");
////		ss << "Test-Data/output/cur_restored_" << sigma_id << "-" << iter << ".txt";
////////		save_as_media<FLOAT_TYPE>(ss.str(), cur_restored, &mfmt);
////		print_mat_details_g<FLOAT_TYPE, 2>(cur_restored, 2, ss.str());//		save_as_media<FLOAT_TYPE>(ss.str(), merged_restored, &mfmt);
//
//		//-- test error
//
//		double err = l2norm<FLOAT_TYPE>(ext_rmask.mul(last_restored - cur_restored));
//		last_restored = cur_restored;
//
//
//		cout << "py: " << py << ", err: " << err;
//		err /= py;
//		cout << ", err/py: " << err << endl;
//		if (sigma_id < phase1_end)
//		{
//			if (err < ep1)
//			{
//				++sigma_id;
//			}
//		}
//		else if (phase1_end <= sigma_id && sigma_id < sigma_num)
//		{
//			if (err < ep2)
//			{
//				++sigma_id;
//			}
//		}
//
//		if (sigma_id >= sigma_num)
//		{
//			break;
//		}
//		//--
//
//	}
//
////	merged_restored = ext_rmask.mul(cur_restored) + ext_masked_img;
//	mat_border_extension<FLOAT_TYPE>(merged_restored, border, "cut", merged_restored);
//
//	save_as_media<FLOAT_TYPE>("Test-Data/output/Lena512-restored.png", merged_restored, &mfmt);
//	psnr<FLOAT_TYPE>(clean_mat, merged_restored, score, msr);
//	cout << endl << "Done sigma: " << stdev[sigma_id - 1] << ", psnr: " << score << ", msr: " << msr << endl;
//
//
//	return 0;
//}
