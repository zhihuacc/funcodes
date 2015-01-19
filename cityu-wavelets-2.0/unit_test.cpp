#include "unit_test.h"

#include "include/wavelets_toolbox.h"
#include "include/denoising.h"

int Unit_Test::decomposition_test(int argc, char **argv)
{
//#define FLOAT_TYPE double
	Media_Format mfmt;
	Mat_<Vec<double, 2> > mat;
	load_as_tensor<double>("Test-Data/Lena512.png", mat, &mfmt);

	mat = Mat_<Vec<double, 2> >(2, (int[]){16,16});
	int rnd = 137;
//	for (int i = 0; i < mat.size[0]; ++i)
//	{
//		for (int j = 0; j < mat.size[1]; ++j)
//		{
//			mat(i,j) = Vec<double, 2>(rnd / (i + 1) % (j + 1), j * (10-i) % 5);
//		}
//	}

	for (int i = 0; i < mat.size[0]; ++i)
	{
		for (int j = 0; j < mat.size[1]; ++j)
		{
			mat(i,j) = Vec<double, 2>(rnd / (i + 1) % (j + 1), 0);
		}
	}

	int nlevels = 1;
	int nd = mat.dims;
	ML_MD_FS_Param fs_param(nlevels, nd);
	fs_param.isSym = true;
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

	ML_MC_Coefs_Set<double>::type coefs_set;
	ML_MC_Filter_Norms_Set<double>::type norms_set;

	clock_t t0 = tic();
	decompose_by_ml_md_filter_bank2<double>(fs_param, mat, filter_system, norms_set, coefs_set);
	clock_t t1 = tic();
	string msg = show_elapse(t1 - t0);
	cout << msg << endl;

	for (int i = 0; i < (int)coefs_set.size(); ++i)
	{
		for (int j = 0; j < (int)coefs_set[i].size(); ++j)
		{
			stringstream ss;
			ss <<  "Test-Data/output/coef-" << i << "-" << j << ".txt";
//			save_as_media<double>(ss.str(), coefs_set[i][j], &mfmt);

			Mat_<Vec<double, 2> > fd;
			normalized_fft<double>(coefs_set[i][j], fd);
			center_shift<double>(fd, fd);
//			cout << "Coef-" << i << "-" << j << endl;
			print_mat_details_g<double,2>(fd, 2, ss.str());
			cout << endl;
		}
	}

//	cout << "Origin: " << endl;
//	print_mat_details_g<double, 2>(mat, 2);

	Mat_<Vec<double, 2> > rec;
	reconstruct_by_ml_md_filter_bank2<double>(fs_param, filter_system, coefs_set, rec);
	cout << "Rec: " << endl;
//	print_mat_details_g<double,2>(rec, 2);

	double score, msr;

	psnr(mat, rec, score, msr);
	cout << "PSNR: score: " << score << ", msr: " << msr << endl;

	return 0;
}

int Unit_Test::reconstruction_test(int argc, char **argv)
{
#define RECONSTRUCT_FLOAT_TYPE double
	string filename("Test-Data/coastguard144.avi");
	Media_Format mfmt;
	Mat_<Vec<RECONSTRUCT_FLOAT_TYPE, 2> > mat, mat_ext, mat_cut;
	load_as_tensor<RECONSTRUCT_FLOAT_TYPE>(filename, mat, &mfmt);

	int nlevels = 2;
	int nd = mat.dims;
	ML_MD_FS_Param fs_param(nlevels, nd);
	fs_param.isSym = true;
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

			//CTF6d4
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].ctrl_points = Smart64FArray(6, (double[]){-2, -1.145796, 0, 1.145796, 2, M_PI});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].epsilons = Smart64FArray(6, (double[]){0.35, 0.3, 0.125, 0.3, 0.35, 0.0778});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].degree = 1;
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].highpass_ds_folds = SmartIntArray(6, (int[]){4,4,4,4,4,4});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].lowpass_ds_fold = 2;
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].opt = "sincos";

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

	SmartIntArray border(mat.dims, 12);

	SmartIntArray mat_size(mat.dims, mat.size);
	figure_good_mat_size(fs_param, mat_size, border, border);
//	border[0] = 32;
//	border[1] = 36;
//	border[2] = 16;

	mat_border_extension<RECONSTRUCT_FLOAT_TYPE>(mat, border, "mir1001", mat_ext);
//	mat_ext = mat;

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
	decompose_by_ml_md_filter_bank2<RECONSTRUCT_FLOAT_TYPE>(fs_param, mat_ext, filter_system, norms_set, coefs_set);
	clock_t t1 = tic();
	string msg = show_elapse(t1 - t0);
	cout << "Dec Time: " << endl << msg << endl;
	mat_ext.release();

//	for (int i = 0; i < filter_system.nlevels; ++i)
//	{
//		for (int j = 0; j < filter_system.ndims; ++j)
//		{
//
//			const OneD_FSystem<RECONSTRUCT_FLOAT_TYPE> &filters = filter_system.md_fs_at_level[i].oned_fs_at_dim[j];
//			for (int k = 0; k < filters.filters.len; ++k)
//			{
//				cout << "Supp: " << i << " " << j << " " << k << endl;
//				cout << "    Origin: " << endl;
//				for (int n = 0; n < filters.filters[k].support_after_ds.len; ++n)
//				{
//					cout << filters.filters[k].support_after_ds[n] << " ";
//				}
//				cout << endl;
//			}
//			cout << endl;
//		}
//	}
//
	for (int i = 0; i < (int)coefs_set.size(); ++i)
	{
		for (int j = 0; j < (int)coefs_set[i].size(); ++j)
		{
			stringstream ss;
			ss <<  "Test-Data/output/coef-" << i << "-" << j << ".txt";
//			save_as_media<double>(ss.str(), coefs_set[i][j], &mfmt);

//			Mat_<Vec<double, 2> > fd;
//			normalized_fft<double>(coefs_set[i][j], fd);
//			center_shift<double>(fd, fd);
			print_mat_details_g<double,2>(coefs_set[i][j], 2, ss.str());

		}
	}


	Mat_<Vec<RECONSTRUCT_FLOAT_TYPE, 2> > rec;
	t0 = tic();
	reconstruct_by_ml_md_filter_bank2<RECONSTRUCT_FLOAT_TYPE>(fs_param, filter_system, coefs_set, rec);
	t1 = tic();
	msg = show_elapse(t1 - t0);
	cout << "Rec Time: " << endl << msg << endl;

	mat_border_extension<RECONSTRUCT_FLOAT_TYPE>(rec, border, "cut", mat_cut);
//	mat_cut = rec;
	double score, msr;
	psnr<RECONSTRUCT_FLOAT_TYPE>(mat, mat_cut, score, msr);
	cout << "PSNR: score: " << score << ", msr: " << msr << endl;

	save_as_media<RECONSTRUCT_FLOAT_TYPE>("Test-Data/output/coastguard144-rec.avi", mat_cut, &mfmt);


	return 0;
}

int Unit_Test::reconstruction_test2(int argc, char **argv)
{
#define RECONSTRUCT_FLOAT_TYPE double
	string filename("Test-Data/Lena512-300.png");
	Media_Format mfmt;
	Mat_<Vec<RECONSTRUCT_FLOAT_TYPE, 2> > mat, mat1, mat_ext, mat_ext1, mat_cut, mat_cut1;
	load_as_tensor<RECONSTRUCT_FLOAT_TYPE>(filename, mat, &mfmt);

	int nlevels = 1;
	int nd = mat.dims;
	ML_MD_FS_Param fs_param(nlevels, nd);
	fs_param.isSym = true;
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
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].ctrl_points = Smart64FArray(6, (double[]){-2, -1.145796, 0, 1.145796, 2, M_PI});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].epsilons = Smart64FArray(6, (double[]){0.35, 0.3, 0.125, 0.3, 0.35, 0.0778});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].degree = 1;
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].highpass_ds_folds = SmartIntArray(6, (int[]){4,4,4,4,4,4});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].lowpass_ds_fold = 2;
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].opt = "sincos";

			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].ctrl_points = Smart64FArray(6, (double[]){-2, -1.15, 0, 1.15, 2, M_PI});
			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].epsilons = Smart64FArray(6, (double[]){0.35, 0.3, 0.125, 0.3, 0.35, 0.0778});
			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].degree = 1;
			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].highpass_ds_folds = SmartIntArray(6, (int[]){2,2,2,2,2,2});
			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].lowpass_ds_fold = 2;
			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].opt = "sincos";

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

	SmartIntArray border(mat.dims, 12);

	SmartIntArray mat_size(mat.dims, mat.size);
	figure_good_mat_size(fs_param, mat_size, border, border);
//	border[0] = 12;
//	border[1] = 16;
//	border[2] = 16;

	mat_border_extension<RECONSTRUCT_FLOAT_TYPE>(mat, border, "sym", mat_ext);
	mat_border_extension<RECONSTRUCT_FLOAT_TYPE>(mat1, border, "sym", mat_ext1);
//	mat_ext = mat;

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
	decompose_by_ml_md_filter_bank2<RECONSTRUCT_FLOAT_TYPE>(fs_param, mat_ext, filter_system, norms_set, coefs_set);
	clock_t t1 = tic();
	string msg = show_elapse(t1 - t0);
	cout << "Dec Time: " << endl << msg << endl;


	ML_MD_FS_Param fs_param1 = fs_param;
	fs_param1.isSym = !fs_param.isSym;
	ML_MD_FSystem<RECONSTRUCT_FLOAT_TYPE> filter_system1;
	ML_MC_Coefs_Set<RECONSTRUCT_FLOAT_TYPE>::type coefs_set1;
	ML_MC_Filter_Norms_Set<RECONSTRUCT_FLOAT_TYPE>::type norms_set1;
	decompose_by_ml_md_filter_bank2<RECONSTRUCT_FLOAT_TYPE>(fs_param1, mat_ext, filter_system1, norms_set1, coefs_set1);

	mat_ext.release();

	cout << "PSNR level 1 lowpass: " << endl;
	double score1, msr1;
	psnr<RECONSTRUCT_FLOAT_TYPE>(coefs_set[0][coefs_set[0].size()  -1], coefs_set1[0][coefs_set1[0].size()  -1], score1, msr1);
	cout <<"   score: " << score1 << ", msr: " << msr1 << endl;
	cout << "PSNR level 2 lowpass: " << endl;
	psnr<RECONSTRUCT_FLOAT_TYPE>(coefs_set[1][coefs_set[1].size()  -1], coefs_set1[1][coefs_set1[1].size()  -1], score1, msr1);
	cout <<"   score: " << score1 << ", msr: " << msr1 << endl;

	psnr<RECONSTRUCT_FLOAT_TYPE>(coefs_set[1][0], coefs_set1[1][coefs_set1[1].size() - 2], score1, msr1);
	cout <<"   score: " << score1 << ", msr: " << msr1 << endl;

//
//	for (int i = 0; i < filter_system.nlevels; ++i)
//	{
//		for (int j = 0; j < filter_system.ndims; ++j)
//		{
//
//			const OneD_FSystem<RECONSTRUCT_FLOAT_TYPE> &filters = filter_system.md_fs_at_level[i].oned_fs_at_dim[j];
//			for (int k = 0; k < filters.filters.len; ++k)
//			{
//				cout << "Supp: " << i << " " << j << " " << k << endl;
//				cout << "    Origin: " << endl;
//				for (int n = 0; n < filters.filters[k].support_after_ds.len; ++n)
//				{
//					cout << filters.filters[k].support_after_ds[n] << " ";
//				}
//				cout << endl;
//			}
//			cout << endl;
//		}
//	}

//	for (int i = 0; i < (int)coefs_set.size(); ++i)
//	{
//		for (int j = 0; j < (int)coefs_set[i].size(); ++j)
//		{
//			stringstream ss;
//			ss <<  "Test-Data/output/coef-" << i << "-" << j << ".txt";
////			save_as_media<double>(ss.str(), coefs_set[i][j], &mfmt);
//
//			Mat_<Vec<double, 2> > fd;
//			normalized_fft<double>(coefs_set[i][j], fd);
//			center_shift<double>(fd, fd);
////			cout << "Coef-" << i << "-" << j << endl;
//			print_mat_details_g<double,2>(fd, 2, ss.str());
//			cout << endl;
//		}
//	}


	Mat_<Vec<RECONSTRUCT_FLOAT_TYPE, 2> > rec;
	t0 = tic();
	reconstruct_by_ml_md_filter_bank2<RECONSTRUCT_FLOAT_TYPE>(fs_param, filter_system, coefs_set, rec);
	t1 = tic();
	msg = show_elapse(t1 - t0);
	cout << "Rec Time: " << endl << msg << endl;

	mat_border_extension<RECONSTRUCT_FLOAT_TYPE>(rec, border, "cut", mat_cut);
//	mat_cut = rec;
	double score, msr;
	psnr<RECONSTRUCT_FLOAT_TYPE>(mat, mat_cut, score, msr);
	cout << "PSNR: score: " << score << ", msr: " << msr << endl;

	save_as_media<RECONSTRUCT_FLOAT_TYPE>("Test-Data/output/Lena512-300-rec.png", mat_cut, &mfmt);


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

//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].ctrl_points = Smart64FArray(6, (double[]){-2, -1.145796, 0, 1.145796, 2, M_PI});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].epsilons = Smart64FArray(6, (double[]){0.35, 0.3, 0.125, 0.3, 0.35, 0.0778});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].degree = 1;
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].highpass_ds_folds = SmartIntArray(6, (int[]){4,4,4,4,4,4});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].lowpass_ds_fold = 2;
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].opt = "sincos";

//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].ctrl_points = Smart64FArray(6, (double[]){-2, -1.15, 0, 1.15, 2, M_PI});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].epsilons = Smart64FArray(6, (double[]){0.35, 0.3, 0.125, 0.3, 0.35, 0.0778});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].degree = 1;
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].highpass_ds_folds = SmartIntArray(6, (int[]){4,4,4,4,4,4});
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].lowpass_ds_fold = 2;
//			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].opt = "sincos";

			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].ctrl_points = Smart64FArray(6, (double[]){-2, -1.15, 0, 1.15, 2, M_PI});
			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].epsilons = Smart64FArray(6, (double[]){0.35, 0.3, 0.125, 0.3, 0.35, 0.0778});
			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].degree = 1;
			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].highpass_ds_folds = SmartIntArray(6, (int[]){2,2,2,2,2,2});
			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].lowpass_ds_fold = 2;
			fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].opt = "sincos";
		}
	}

	ML_MD_FSystem<double> fs(nlevels, nd);
	Mat_<Vec<double, 2> > x_pts;
	linspace<double>(complex<double>(-M_PI, 0), complex<double>(M_PI, 0), 332, x_pts);
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
		const Mat_<Vec<double, 2> > coefs = oned_fs.filters[i].coefs;
		pw_abs<double>(coefs, square);
		pw_pow<double>(square, 2, square);
		sum += square;
		cout << "Filter " << i << endl;
//		print_mat_details_g<double, 2>(coefs, 0, "Test-Data/output/log.txt");
		for (int p = 0; p < (int)coefs.total(); ++p)
		{
			if (coefs.at<complex<double> >(0, p).real() > 0)
			{
				cout << p << "-" << coefs.at<complex<double> >(0,p).real() << " ";
			}
		}
		cout << endl;
		const SmartIntArray &supp = oned_fs.filters[i].support_after_ds;
		const SmartIntArray &sym_supp = oned_fs.filters[i].sym_support_after_ds;
		cout << "Support: " << supp.len << endl;
		for (int j = 0; j < supp.len; ++j)
		{
			cout << supp[j] << " ";
		}
		cout << endl;
		cout << "   sym: " << endl;
		for (int j = 0; j < supp.len; ++j)
		{
			cout << sym_supp[j] << " ";
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
	cout << "Added: " << endl;
	mat_subadd<mat_select_test_FLOAT_TYPE>(zeros, index_at_dim, sub_mat);
	print_mat_details_g<mat_select_test_FLOAT_TYPE,2>(zeros, 2);

	return 0;
}

int Unit_Test::denoising(int argc, char **argv)
{
#define DENOISING_FLOAT_TYPE double

	string img_names("Test-Data/coastguard144.xml");

	int ret = 0;
	Media_Format mfmt;
	Mat_<Vec<DENOISING_FLOAT_TYPE, 2> > input, noisy_input, denoised_output;
	load_as_tensor<DENOISING_FLOAT_TYPE>(img_names, input, &mfmt);
	int ndims = input.dims;

	//-- Fake up noisy data.
	double mean = 0;
	double stdev = 5;
	Mat_<Vec<DENOISING_FLOAT_TYPE, 1> > channels[2];
	channels[0] = Mat_<Vec<DENOISING_FLOAT_TYPE, 1> >(input.dims, input.size);
	channels[1] = Mat_<Vec<DENOISING_FLOAT_TYPE, 1> >(input.dims, input.size, Vec<DENOISING_FLOAT_TYPE, 1>((DENOISING_FLOAT_TYPE)0));
	randn(channels[0], mean, stdev);
	merge(channels, 2, noisy_input);
	channels[0].release();
	channels[1].release();
	noisy_input = input + noisy_input;
//	noisy_input = input;
	// --

	double score, msr;
	psnr<DENOISING_FLOAT_TYPE>(input, noisy_input, score, msr);
	cout << "Noisy Input PSNR score: " << score << ", msr: " << msr << endl << endl;

	// -- Prepare parameters;
	int nlevels = 2;
	string fs_param_opt = "CTF3";
	int ext_size = 12;
	string ext_opt = "mir1001";
	ML_MD_FS_Param ml_md_fs_param;
	ret = compose_fs_param(nlevels, ndims, fs_param_opt, ext_size, ext_opt, true, ml_md_fs_param);
	if (ret)
	{
		cout << "Error in FS param. " << endl;
		return 0;
	}

//	Thresholding_Param thr_param;
//	thr_param.c = 1;
//	thr_param.mean = mean;
//	thr_param.stdev = stdev;
//	thr_param.doNormalization = true;
//	thr_param.wwidth = 7;        //Should be odd.
//	thr_param.thr_method = "bishrink";

	Thresholding_Param thr_param;
	ret = compose_thr_param(mean, stdev, 1, 7, true, "localsoft", thr_param);
	if (ret)
	{
		cout << "Error in Thr param. " << endl;
		return 0;
	}
	// --

	thresholding_denoise<DENOISING_FLOAT_TYPE>(noisy_input, ml_md_fs_param, thr_param, denoised_output);
	noisy_input.release();


	psnr<DENOISING_FLOAT_TYPE>(input, denoised_output, score, msr);
	cout << "Denoised PSNR score: " << score << ", msr: " << msr << endl;

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

int Unit_Test::test_any(int argc, char **argv)
{

	Media_Format mfmt2;
	Mat_<Vec<double, 2> > mat2;
	load_as_tensor<double>("Test-Data/Lena512.png", mat2, &mfmt2);

	int nd = 2;
	int wwidth = 7;
	SmartIntArray winsize(nd, wwidth);
	Mat_<Vec<double, 2> > avg_window(nd, winsize, Vec<double, 2>(1.0 / pow(wwidth, nd),0));
	SmartIntArray anchor(nd, wwidth / 2);
	SmartIntArray border(nd, wwidth - 1);

	mat_border_extension<double>(mat2, border, "blk", mat2);
	md_filtering<double>(mat2, avg_window, anchor, mat2);
	mat_border_extension<double>(mat2, border, "cut", mat2);

	clock_t t0 = tic();
	for (int i = 0; i < 10; ++i)
	{
		mat_border_extension<double>(mat2, border, "blk", mat2);
		md_filtering<double>(mat2, avg_window, anchor, mat2);
		mat_border_extension<double>(mat2, border, "cut", mat2);
	}
	clock_t t1 = tic();
	string msg = show_elapse(t1 - t0);
	cout << "md_filtering" << endl << msg << endl;

	save_as_media<double>("Test-Data/output/Lena272-168.png", mat2, &mfmt2);

	return 0;

	Mat_<Vec<double, 2> > mat(3, (int[]){10, 10, 10}, Vec<double, 2>(0,0));
	for (int i = 0; i < mat.size[0]; ++i)
	{
		for (int j = 0; j < mat.size[1]; ++j)
		{
			for (int k = 0; k < mat.size[2]; ++k)
			{
				mat(i,j,k)[0] = 1;
			}
		}
	}


	print_mat_details_g<double, 2>(mat, 2, "Test-Data/output/origin_mat.txt");
	cout << endl << endl;

	{
	SmartIntArray border(3, 5);
	mat_border_extension(mat, border, "blk", mat);
	print_mat_details_g<double, 2>(mat, 2, "Test-Data/output/ext_mat.txt");
	}

//    Mat_<Vec<double, 2> > filter(3, (int[]){3,3,3}, Vec<double, 2>(1.0/27.0,0));
//    SmartIntArray achor(3, 1);
//    md_filtering<double>(mat, filter, achor, mat);
//    print_mat_details_g<double, 2>(mat, 2, "Test-Data/output/conv.txt");

//    pw_abs<double>(mat, mat);
//    cout << endl << "Abs: " << endl;
//    print_mat_details_g<double,2>(mat, 2);

	return 0;
}
