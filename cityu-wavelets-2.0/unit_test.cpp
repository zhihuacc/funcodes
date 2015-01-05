#include "unit_test.h"

#include "include/wavelets_toolbox.h"
#include "include/app.h"

int Unit_Test::decomposition_test(int argc, char **argv)
{
//#define FLOAT_TYPE double
	Media_Format mfmt;
	Mat_<Vec<double, 2> > mat;
	load_as_tensor<double>("Test-Data/Lena512.png", mat, &mfmt);

	int nlevels = 2;
	int nd = mat.dims;
	ML_MD_FS_Param fs_param(nlevels, nd);

	for (int i = 0; i < fs_param.nlevels; ++i)
	{
		for (int j = 0; j < fs_param.ndims; ++j)
		{
			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].ctrl_points = Smart64FArray(3, (double[]){-33.0/32.0, 33.0/32.0, M_PI});
			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].epsilons = Smart64FArray(3, (double[]){69.0/128.0, 69.0/128.0, 51.0/512.0});
			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].degree = 1;
			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].highpass_ds_folds = SmartIntArray(3, (int[]){2,2,2});
			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].lowpass_ds_fold = 2;
			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].opt = "sincos";

//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].ctrl_points = Smart64FArray(6, (double[]){-2, -1.145796, 0, 1.145796, 2, M_PI});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].epsilons = Smart64FArray(6, (double[]){0.35, 0.3, 0.125, 0.3, 0.35, 0.0778});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].degree = 1;
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].highpass_ds_folds = SmartIntArray(6, (int[]){4,4,4,4,4,4});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].lowpass_ds_fold = 2;
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].opt = "sincos";
		}
	}

	ML_MD_FSystem<double> filter_system;
//	typename ML_MC_Coefs_Set<double>::type coefs_set;
//	typename ML_MC_Filter_Norms_Set<double>::type norms_set;

	ML_MC_Coefs_Set<double>::type coefs_set;
	ML_MC_Filter_Norms_Set<double>::type norms_set;

	clock_t t0 = tic();
	decompose_by_ml_md_filter_bank<double>(fs_param, mat, filter_system, norms_set, coefs_set);
	clock_t t1 = tic();
	string msg = show_elapse(t1 - t0);
	cout << msg << endl;

	for (int i = 0; i < (int)coefs_set.size(); ++i)
	{
		for (int j = 0; j < (int)coefs_set[i].size(); ++j)
		{
			stringstream ss;
			ss <<  "Test-Data/output/coef-" << i << "-" << j << ".jpg";
			save_as_media<double>(ss.str(), coefs_set[i][j], &mfmt);
		}
	}

	return 0;
}

int Unit_Test::reconstruction_test(int argc, char **argv)
{
#define RECONSTRUCT_FLOAT_TYPE double
	string filename("Test-Data/coastguard144.avi");
//	string output_filename("Test-Data/output/gflower.avi");
	Media_Format mfmt;
	Mat_<Vec<RECONSTRUCT_FLOAT_TYPE, 2> > mat, mat_ext, mat_cut;
	load_as_tensor<RECONSTRUCT_FLOAT_TYPE>(filename, mat, &mfmt);
//	save_as_media<double>("Test-Data/output/Lena512-origin.png", mat, &mfmt);

	int nlevels = 2;
	int nd = mat.dims;
	ML_MD_FS_Param fs_param(nlevels, nd);

	for (int i = 0; i < fs_param.nlevels; ++i)
	{
		for (int j = 0; j < fs_param.ndims; ++j)
		{
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].ctrl_points = Smart64FArray(3, (double[]){-33.0/32.0, 33.0/32.0, M_PI});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].epsilons = Smart64FArray(3, (double[]){69.0/128.0, 69.0/128.0, 51.0/512.0});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].degree = 1;
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].highpass_ds_folds = SmartIntArray(3, (int[]){2,2,2});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].lowpass_ds_fold = 2;
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].opt = "sincos";

			//CTF6d4
			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].ctrl_points = Smart64FArray(6, (double[]){-2, -1.145796, 0, 1.145796, 2, M_PI});
			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].epsilons = Smart64FArray(6, (double[]){0.35, 0.3, 0.125, 0.3, 0.35, 0.0778});
			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].degree = 1;
			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].highpass_ds_folds = SmartIntArray(6, (int[]){4,4,4,4,4,4});
			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].lowpass_ds_fold = 2;
			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].opt = "sincos";

//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].ctrl_points = Smart64FArray(6, (double[]){-2, -1.15, 0, 1.15, 2, M_PI});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].epsilons = Smart64FArray(6, (double[]){0.35, 0.3, 0.125, 0.3, 0.35, 0.0778});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].degree = 1;
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].highpass_ds_folds = SmartIntArray(6, (int[]){2,2,2,2,2,2});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].lowpass_ds_fold = 2;
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].opt = "sincos";

//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].ctrl_points = Smart64FArray(6, (double[]){-2, -1.15, 0, 1.15, 2, M_PI});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].epsilons = Smart64FArray(6, (double[]){0.35, 0.3, 0.125, 0.3, 0.35, 0.0778});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].degree = 1;
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].highpass_ds_folds = SmartIntArray(6, (int[]){4,4,4,4,4,4});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].lowpass_ds_fold = 2;
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].opt = "sincos";

			//CTF6d2
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].ctrl_points = Smart64FArray(6, (double[]){-(M_PI + 119.0/128.0)/2.0, -119.0/128.0, 0, 119.0/128.0, (M_PI + 119.0/128.0) / 2.0, M_PI});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].epsilons = Smart64FArray(6, (double[]){115.0/256.0, 81.0/128.0, 35.0/128.0, 81.0/128.0, 115.0/256.0, 115.0/256.0});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].degree = 1;
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].highpass_ds_folds = SmartIntArray(6, (int[]){2,2,2,2,2,2});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].lowpass_ds_fold = 2;
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].opt = "sincos";
		}
	}

	SmartIntArray border(mat.dims, 32);

	SmartIntArray mat_size(mat.dims, mat.size);
	figure_good_mat_size(fs_param, mat_size, border);
	border[0] = 12;
	border[1] = 16;
	border[2] = 16;

	mat_border_extension<RECONSTRUCT_FLOAT_TYPE>(mat, border, "sym", mat_ext);
//	mat_ext = mat;
//	save_as_media<double>("Test-Data/output/Lena512-ext.png", mat_ext, &mfmt);


	ML_MD_FSystem<RECONSTRUCT_FLOAT_TYPE> filter_system;
	ML_MC_Coefs_Set<RECONSTRUCT_FLOAT_TYPE>::type coefs_set;
	ML_MC_Filter_Norms_Set<RECONSTRUCT_FLOAT_TYPE>::type norms_set;

	int check = check_mat_to_decompose<RECONSTRUCT_FLOAT_TYPE>(fs_param, mat_ext);
	if (check)
	{
		cout << "Mat is NOT in good shape to decompose. ret = " << check << endl;
		return 0;
	}
	clock_t t0 = tic();
	decompose_by_ml_md_filter_bank<RECONSTRUCT_FLOAT_TYPE>(fs_param, mat_ext, filter_system, norms_set, coefs_set);
	clock_t t1 = tic();
	string msg = show_elapse(t1 - t0);
	cout << "Dec Time: " << endl << msg << endl;
	mat_ext.release();


	Mat_<Vec<RECONSTRUCT_FLOAT_TYPE, 2> > rec;
	t0 = tic();
	reconstruct_by_ml_md_filter_bank<RECONSTRUCT_FLOAT_TYPE>(fs_param, filter_system, coefs_set, rec);
	t1 = tic();
	msg = show_elapse(t1 - t0);
	cout << "Rec Time: " << endl << msg << endl;


//	save_as_media<double>("Test-Data/output/Lena512-ext-rec.png", rec, &mfmt);
	mat_border_extension<RECONSTRUCT_FLOAT_TYPE>(rec, border, "cut", mat_cut);
//	mat_cut = rec;
	save_as_media<RECONSTRUCT_FLOAT_TYPE>("Test-Data/output/mobile2-rec.avi", mat_cut, &mfmt);

	double score, msr;
	psnr<RECONSTRUCT_FLOAT_TYPE>(mat, mat_cut, score, msr);
	cout << "PSNR: score: " << score << ", msr: " << msr << endl;

	return 0;
}

int Unit_Test::construct_1d_filter_test(int argc, char **argv)
{
	int nlevels = 2;
	int nd = 2;
	ML_MD_FS_Param fs_param(nlevels, nd);

	for (int i = 0; i < fs_param.nlevels; ++i)
	{
		for (int j = 0; j < fs_param.ndims; ++j)
		{
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].ctrl_points = Smart64FArray(3, (double[]){-33.0/32.0, 33.0/32.0, M_PI});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].epsilons = Smart64FArray(3, (double[]){69.0/128.0, 69.0/128.0, 51.0/512.0});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].degree = 1;
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].highpass_ds_folds = SmartIntArray(3, (int[]){2,2,2});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].lowpass_ds_fold = 2;
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].opt = "sincos";

			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].ctrl_points = Smart64FArray(6, (double[]){-2, -1.145796, 0, 1.145796, 2, M_PI});
			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].epsilons = Smart64FArray(6, (double[]){0.35, 0.3, 0.125, 0.3, 0.35, 0.0778});
			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].degree = 1;
			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].highpass_ds_folds = SmartIntArray(6, (int[]){4,4,4,4,4,4});
			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].lowpass_ds_fold = 2;
			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].opt = "sincos";

//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].ctrl_points = Smart64FArray(6, (double[]){-2, -1.15, 0, 1.15, 2, M_PI});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].epsilons = Smart64FArray(6, (double[]){0.35, 0.3, 0.125, 0.3, 0.35, 0.0778});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].degree = 1;
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].highpass_ds_folds = SmartIntArray(6, (int[]){4,4,4,4,4,4});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].lowpass_ds_fold = 2;
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].opt = "sincos";
		}
	}

	ML_MD_FSystem<double> fs(nlevels, nd);
	Mat_<Vec<double, 2> > x_pts;
	linspace<double>(complex<double>(-M_PI, 0), complex<double>(M_PI, 0), 150, x_pts);
	cout << "X: " << endl;
	print_mat_details_g<float, 2>(x_pts, 0);

	for (int i = 0; i < fs_param.ndims; ++i)
	{
		construct_1d_filter_system<double>(x_pts, fs_param.md_fs_param_at_level[0].oned_fs_param_at_dim[i],
								   fs.md_fs_at_level[0].oned_fs_at_dim[i]);
	}

	const OneD_FSystem<double> &oned_fs = fs.md_fs_at_level[0].oned_fs_at_dim[0];
	Mat_<Vec<double, 2> > sum(oned_fs.filters[0].coefs.dims, oned_fs.filters[0].coefs.size, Vec<double, 2>(0,0));
	for (int i = 0; i < (int)oned_fs.filters.len; ++i)
	{
		Mat_<Vec<double, 2> > square;
		pw_abs<double>(oned_fs.filters[i].coefs, square);
		pw_pow<double>(square, 2, square);
		sum += square;
		cout << "Filter " << i << endl;
		print_mat_details_g<double, 2>(oned_fs.filters[i].coefs, 0, "Test-Data/output/log.txt");
		const SmartIntArray &supp = oned_fs.filters[i].support_after_ds;
		cout << "Support: " << supp.len << endl;
		for (int j = 0; j < supp.len; ++j)
		{
			cout << supp[j] << " ";
		}
		cout << endl << endl;
	}

	cout << "Sum: " << endl;
	print_mat_details_g<float, 2>(sum, 0);

	return 0;

	SmartArray<Mat_<Vec<double, 2> > > comps_at_dim(nd);
	comps_at_dim[0] = fs.md_fs_at_level[0].oned_fs_at_dim[0].filters[0].coefs;
	comps_at_dim[1] = fs.md_fs_at_level[0].oned_fs_at_dim[1].filters[0].coefs;

	cout << "Tensor: " << endl;
	Mat_<Vec<double, 2> > md_filter;
	tensor_product<double>(comps_at_dim, md_filter);
	print_mat_details_g<double, 2>(md_filter, 0, "Test-Data/output/log.txt");
	save_as_media<double>("Test-Data/output/tensor0.png", md_filter, NULL);


	comps_at_dim[0] = fs.md_fs_at_level[0].oned_fs_at_dim[0].filters[0].coefs;
	comps_at_dim[1] = fs.md_fs_at_level[0].oned_fs_at_dim[1].filters[1].coefs;
	tensor_product<double>(comps_at_dim, md_filter);
	save_as_media<double>("Test-Data/output/tensor1.png", md_filter, NULL);

	comps_at_dim[0] = fs.md_fs_at_level[0].oned_fs_at_dim[0].filters[0].coefs;
	comps_at_dim[1] = fs.md_fs_at_level[0].oned_fs_at_dim[1].filters[2].coefs;
	tensor_product<double>(comps_at_dim, md_filter);
	save_as_media<double>("Test-Data/output/tensor2.png", md_filter, NULL);

	comps_at_dim[0] = fs.md_fs_at_level[0].oned_fs_at_dim[0].filters[1].coefs;
	comps_at_dim[1] = fs.md_fs_at_level[0].oned_fs_at_dim[1].filters[1].coefs;
	tensor_product<double>(comps_at_dim, md_filter);
	save_as_media<double>("Test-Data/output/tensor3.png", md_filter, NULL);

	comps_at_dim[0] = fs.md_fs_at_level[0].oned_fs_at_dim[0].filters[1].coefs;
	comps_at_dim[1] = fs.md_fs_at_level[0].oned_fs_at_dim[1].filters[2].coefs;
	tensor_product<double>(comps_at_dim, md_filter);
	save_as_media<double>("Test-Data/output/tensor4.png", md_filter, NULL);

	comps_at_dim[0] = fs.md_fs_at_level[0].oned_fs_at_dim[0].filters[2].coefs;
	comps_at_dim[1] = fs.md_fs_at_level[0].oned_fs_at_dim[1].filters[2].coefs;
	tensor_product<double>(comps_at_dim, md_filter);
	save_as_media<double>("Test-Data/output/tensor5.png", md_filter, NULL);

	return 0;
}

int Unit_Test::fft_center_shift_test(int argc, char **argv)
{
	string filename("Test-Data/gflower.avi");
	string output_filename("Test-Data/output/Lena512.png");
	Media_Format mfmt;
	Mat_<Vec<double, 2> > mat, fd, td;
	load_as_tensor<double>(filename, mat, &mfmt);
	save_as_media<double>("Test-Data/output/gflower-origin.avi", mat, &mfmt);

	normalized_fft<double>(mat, fd);
	save_as_media<double>("Test-Data/output/gflower-fd.avi", fd, &mfmt);

	center_shift<double>(fd, fd);
	save_as_media<double>("Test-Data/output/gflower-fd-shift.avi", fd, &mfmt);

	icenter_shift<double>(fd, fd);
	save_as_media<double>("Test-Data/output/gflower-fd-shiftback.avi", fd, &mfmt);

	normalized_ifft<double>(fd, td);
	save_as_media<double>("Test-Data/output/gflower-td.avi", td, &mfmt);

	return 0;
}

int Unit_Test::mat_select_test(int argc, char **argv)
{

#define mat_select_test_FLOAT_TYPE float
	Mat_<Vec<mat_select_test_FLOAT_TYPE, 2> > mat(2, (int[]){5,8}), sub_mat, zeros(2, (int[]){5,8}, Vec<mat_select_test_FLOAT_TYPE,2>(0,0));
	for (int i = 0; i < mat.size[0]; ++i)
	{
		for (int j = 0; j < mat.size[1]; ++j)
		{
			mat(i,j) = Vec<mat_select_test_FLOAT_TYPE, 2>(i,j);
		}
	}

	SmartArray<SmartIntArray> index_at_dim(mat.dims);
	index_at_dim[0].reserve(3);
	index_at_dim[1].reserve(5);
	/*
	 *   4 5 6 2 3
	 * 4
	 * 1
	 * 2
	 */
	index_at_dim[0][0] = 4;
	index_at_dim[0][1] = 1;
	index_at_dim[0][2] = 2;
	index_at_dim[1][0] = 4;
	index_at_dim[1][1] = 5;
	index_at_dim[1][2] = 6;
	index_at_dim[1][3] = 2;
	index_at_dim[1][4] = 3;

	mat_select<mat_select_test_FLOAT_TYPE>(mat, index_at_dim, sub_mat);
	mat_subfill<mat_select_test_FLOAT_TYPE>(zeros, index_at_dim, sub_mat);

	cout << "Origin: " << endl;
	print_mat_details_g<mat_select_test_FLOAT_TYPE,2>(mat, 2);
	cout << "Sub: " << endl;
	print_mat_details_g<mat_select_test_FLOAT_TYPE,2>(sub_mat, 2);
	cout << "Filled: " << endl;
	print_mat_details_g<mat_select_test_FLOAT_TYPE,2>(zeros, 2);

	return 0;
}

int Unit_Test::denoising(int argc, char **argv)
{
	app_denoising();

	return 0;
}

int Unit_Test::psnr_test(int argc, char **argv)
{
#define PSNR_FLOAT_TYPE double
	string filename1("Test-Data/gflower.avi");
	string filename2("Test-Data/output/gflower-rec.avi");
	Media_Format mfmt1, mfmt2;
	Mat_<Vec<PSNR_FLOAT_TYPE, 2> > mat1, mat2;
	load_as_tensor<PSNR_FLOAT_TYPE>(filename1, mat1, &mfmt1);
	load_as_tensor<PSNR_FLOAT_TYPE>(filename2, mat2, &mfmt2);
	double score = -1, msr = -1;
	psnr<PSNR_FLOAT_TYPE>(mat1, mat2, score, msr);
	cout << "PSNR: score: " << score << ", msr: " << msr << endl;
	return 0;
}
