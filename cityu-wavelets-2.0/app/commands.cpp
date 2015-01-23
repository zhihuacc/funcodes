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


int inpaint(const Configuration *cfg, const string &img_name, const string &mask_name)
{
	typedef double FLOAT_TYPE;

	int ret = 0;
	Media_Format mfmt;
	Mat_<Vec<FLOAT_TYPE, 2> > clean_img, mask, masked_img, rmask;
	load_as_tensor<FLOAT_TYPE>(img_name, clean_img, &mfmt);
	load_as_tensor<FLOAT_TYPE>(mask_name, mask, &mfmt);
	mask /= 255;
	rmask = 1.0 - mask;
	int ndims = clean_img.dims;

	double ep1 = 1e-5, ep2 = 5e-5;
	int phase1_end = 9;
	double stdev[] = {84, 76, 68, 60, 52, 44, 36, 28, 20,
			           18, 16, 14, 12, 10, 8, 6, 4, 2};
	string fs[] = {"CTF3", "CTF3", "CTF3", "CTF3", "CTF3", "CTF3", "CTF3", "CTF3", "CTF3",
			       "CTF6D4", "CTF6D4", "CTF6D4", "CTF6D4", "CTF6D4", "CTF6D4", "CTF6D4", "CTF6D4", "CTF6D4"};
	int sigma_num = sizeof(stdev) / sizeof(double);
	masked_img = clean_img.mul(mask);
	save_as_media<FLOAT_TYPE>("Test-Data/output/masked.png", masked_img, &mfmt);

	Mat_<Vec<FLOAT_TYPE, 2> > last_restored,
			                  merged_restored,
			                  cur_restored(ndims, clean_img.size, Vec<FLOAT_TYPE, 2>(0,0));
	int sigma_id = 0;
	for (int iter = 0; iter < 1000; ++iter)
	{
		merged_restored = rmask.mul(cur_restored) + masked_img;
		last_restored = cur_restored;

		// -- Prepare parameters;
		int nlevels = 2;
//		string fs_param_opt = "CTF3";
		int ext_size = 12;
		string ext_opt = "mir1001";
		ML_MD_FS_Param ml_md_fs_param;
		ret = compose_fs_param(nlevels, ndims, fs[sigma_id], ext_size, ext_opt, true, ml_md_fs_param);
		if (ret)
		{
			cout << "Error in FS param. " << endl;
			return 0;
		}


		Thresholding_Param thr_param;
		ret = compose_thr_param(0, stdev[sigma_id], 1, 7, true, "localsoft", thr_param);
		if (ret)
		{
			cout << "Error in Thr param. " << endl;
			return 0;
		}
		// --

		thresholding_denoise<FLOAT_TYPE>(merged_restored, ml_md_fs_param, thr_param, cur_restored);

		stringstream ss;
		ss << "Test-Data/output/cur_restored_" << sigma_id << "-" << iter << ".png";
		save_as_media<FLOAT_TYPE>(ss.str(), cur_restored, &mfmt);
		ss.str("");
		ss << "Test-Data/output/merged_" << sigma_id << "-" << iter << ".png";
		save_as_media<FLOAT_TYPE>(ss.str(), merged_restored, &mfmt);

		//-- test error
		double err = mat_error(last_restored, cur_restored, rmask);
		if (sigma_id < phase1_end && err < ep1)
		{
			++sigma_id;
		}
		else if (sigma_id < sigma_num && err < ep2)
		{
			++sigma_id;
		}

		if (sigma_id >= sigma_num)
		{
			break;
		}
		//--

	}

	merged_restored = rmask.mul(cur_restored) + masked_img;
	save_as_media<FLOAT_TYPE>("Test-Data/output/Lena512-restored.png", merged_restored, &mfmt);
	return 0;
}
