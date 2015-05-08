#include "unit_test.h"

#include "include/wavelets_toolbox.h"
#include "include/denoising.h"
#include "include/inpaint.h"

#include "include/compact_support_wavelets.h"

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

int Unit_Test::conv_and_ds_test(int argc, char **argv)
{
	Media_Format mfmt1, mfmt2;
	Mat_<Vec<double, 2> > mat1, mat2, mat3;
	load_as_tensor<double>("Test-Data/Lena512.png", mat1, &mfmt1);

	Mat_<Vec<double, 2> > td_filter1(2, (int[]){1, 2}, Vec<double, 2>(0,0));
	Mat_<Vec<double, 2> > td_filter2(2, (int[]){1, 2}, Vec<double, 2>(0,0));
	td_filter1(0,0)[0] = 1.0 / 1.4142;
	td_filter1(0,1)[0] = 1.0 / 1.4142;
//	td_filter(0,2)[0] = 1.0 / 5.0;
//	td_filter(0,3)[0] = 1.0 / 5.0;
//	td_filter(0,4)[0] = 1.0 / 5.0;

	td_filter2(0,0)[0] = 1.0 / 1.4142;
	td_filter2(0,1)[0] = -1.0 / 1.4142;

	SmartArray<OneD_TD_Filter<double> > skerns(2);
	skerns[0].coefs = td_filter1;
	skerns[0].anchor = 0;
	skerns[1].coefs = td_filter2;
	skerns[1].anchor = 0;
//	skerns[3].coefs = td_filter;
//	skerns[3].anchor = 0;

//	clock_t s0, s1;
//	string msg;
//	conv_by_separable_kernels<double>(mat1, skerns, true, mat2);
////	save_as_media<double>("Test-Data/Barbara512-conv.png", mat1, NULL);
//	s0 = tic();
//	conv_by_separable_kernels<double>(mat1, skerns, false, mat2);
//	s1 = tic();
//	msg = show_elapse(s1 - s0);
//	cout << msg << endl;

	SmartIntArray step_size(2, (int[]){2,2});
	conv_by_separable_kernels_and_ds<double>(mat1, skerns, step_size, true, mat2);
//	clock_t s2 = tic();
//	msg = show_elapse(s2 - s1);
//	cout << msg << endl;

	save_as_media<double>("Test-Data/output/Barbara512-conv.png", mat2, NULL);

	SmartIntArray ds_steps(2, (int[]){2,2});
	downsample2(mat2, ds_steps, mat3);
	save_as_media<double>("Test-Data/output/Barbara512-conv-ds.png", mat3, NULL);

//	ds_steps[0] = 1;
//	upsample(mat3, ds_steps, mat3);
//	save_as_media<double>("Test-Data/output/Barbara512-conv-us.png", mat3, NULL);

	return 0;
}

int Unit_Test::conv_and_us_test(int argc, char **argv)
{
	Media_Format mfmt1, mfmt2;
	Mat_<Vec<double, 2> > mat1, mat2, mat3;
	load_as_tensor<double>("Test-Data/Lena512.png", mat1, &mfmt1);

	Mat_<Vec<double, 2> > td_filter1(2, (int[]){1, 2}, Vec<double, 2>(0,0));
	Mat_<Vec<double, 2> > td_filter2(2, (int[]){1, 2}, Vec<double, 2>(0,0));
	td_filter1(0,0)[0] = 1.0 / 1.4142;
	td_filter1(0,1)[0] = 1.0 / 1.4142;
//	td_filter(0,2)[0] = 1.0 / 5.0;
//	td_filter(0,3)[0] = 1.0 / 5.0;
//	td_filter(0,4)[0] = 1.0 / 5.0;

	td_filter2(0,0)[0] = -1.0 / 1.4142;
	td_filter2(0,1)[0] = 1.0 / 1.4142;

	SmartArray<OneD_TD_Filter<double> > skerns(2);
	skerns[0].coefs = td_filter1;
	skerns[0].anchor = 1;
	skerns[1].coefs = td_filter2;
	skerns[1].anchor = 1;
//	skerns[3].coefs = td_filter;
//	skerns[3].anchor = 0;

//	clock_t s0, s1;
//	string msg;
//	conv_by_separable_kernels<double>(mat1, skerns, true, mat2);
////	save_as_media<double>("Test-Data/Barbara512-conv.png", mat1, NULL);
//	s0 = tic();
//	conv_by_separable_kernels<double>(mat1, skerns, false, mat2);
//	s1 = tic();
//	msg = show_elapse(s1 - s0);
//	cout << msg << endl;

	SmartIntArray step_size(2, (int[]){2,2});
	conv_by_separable_kernels_and_us<double>(mat1, skerns, step_size, true, mat2);

	upsample<double>(mat1, step_size, mat3);
	conv_by_separable_kernels2<double>(mat3, skerns, true, mat3);
//	clock_t s2 = tic();
//	msg = show_elapse(s2 - s1);
//	cout << msg << endl;

	double score, msr;
	psnr<double>(mat2, mat3, score, msr);
	cout << "PSNR: score: " << score << " msr: " << msr << endl;

	save_as_media<double>("Test-Data/output/conv1.png", mat2, NULL);
//
//	SmartIntArray ds_steps(2, (int[]){2,2});
//	downsample(mat2, ds_steps, mat3);
	save_as_media<double>("Test-Data/output/conv2.png", mat3, NULL);

//	ds_steps[0] = 1;
//	upsample(mat3, ds_steps, mat3);
//	save_as_media<double>("Test-Data/output/Barbara512-conv-us.png", mat3, NULL);

	return 0;
}

int Unit_Test::test_any(int argc, char **argv)
{


//	Media_Format mfmt1, mfmt2;
//	Mat_<Vec<double, 2> > mat1, mat2;
//	load_as_tensor<double>("Test-Data/Lena512.png", mat1, &mfmt1);
//
//	normalized_fft<double>(mat1, mat1);
//	center_shift<double>(mat1, mat1);
//	frequency_domain_density<double>(mat1, 25.0, SmartIntArray(1, (int[]){36}), mat2);
//
//	print_mat_details_g<double, 2>(mat2, 2, "Test-Data/density.txt");

	Media_Format mfmt1, mfmt2;
	Mat_<Vec<double, 2> > mat1, mat2;
	load_as_tensor<double>("Test-Data/Barbara512.png", mat1, &mfmt1);

//	conv_by_separable_kernels_and_ds<double>(mat1, SmartArray<OneD_TD_Filter<double> >(), false, mat1);

	normalized_fft<double>(mat1, mat1);
	center_shift<double>(mat1, mat1);
	save_as_media<double>("Test-Data/Barbara-fft.png", mat1, NULL);

	frequency_domain_density<double>(mat1, 30, SmartIntArray(1, (int[]){90}), mat2);

	print_mat_details_g<double, 2>(mat2, 0, "Test-Data/Barbara-density.txt");

	 double d0, d1;
	 minMaxLoc(mat2, &d0, &d1);
	 mat2.convertTo(mat2, -1, 255.0 / (d1 - d0), -255.0 * d0 / (d1 - d0));
	 mat2 = Scalar(255,0) - mat2;

	save_as_media<double>("Test-Data/Barbara-density.png", mat2, NULL);

	return 0;
}

int Unit_Test::comp_supp_test(int argc, char **argv)
{
	Media_Format mfmt1, mfmt2;
	Mat_<Vec<double, 2> > mat1, mat2, mat3;
	load_as_tensor<double>("Test-Data/benchmark/tennis.avi", mat1, &mfmt1);

	Mat_<Vec<double, 2> > w1(2, (int[]){1, 2}, Vec<double, 2>(0,0));
	Mat_<Vec<double, 2> > w2(2, (int[]){1, 2}, Vec<double, 2>(0,0));
	Mat_<Vec<double, 2> > w3(2, (int[]){1, 2}, Vec<double, 2>(0,0));
	Mat_<Vec<double, 2> > w4(2, (int[]){1, 2}, Vec<double, 2>(0,0));
	double r = sqrt(2);
	w1(0,0)[0] = 1.0 / r;
	w1(0,1)[0] = 1.0 / r;
	w2(0,0)[0] = 1.0 / r;
	w2(0,1)[0] = -1.0 / r;

	w3(0,0)[0] = 1.0 / r;
	w3(0,1)[0] = 1.0 / r;
	w4(0,0)[0] = -1.0 / r;
	w4(0,1)[0] = 1.0 / r;

//	Mat_<Vec<double, 2> > w1(2, (int[]){1, 4}, Vec<double, 2>(0,0));
//	Mat_<Vec<double, 2> > w2(2, (int[]){1, 4}, Vec<double, 2>(0,0));
//	Mat_<Vec<double, 2> > w3(2, (int[]){1, 4}, Vec<double, 2>(0,0));
//	Mat_<Vec<double, 2> > w4(2, (int[]){1, 4}, Vec<double, 2>(0,0));
//	w1(0,0)[0] = -0.129409522550921;
//	w1(0,1)[0] = 0.224143868041857;
//	w1(0,2)[0] = 0.836516303737469;
//	w1(0,3)[0] =  0.482962913144690;
//	w2(0,0)[0] = -0.482962913144690;
//	w2(0,1)[0] =  0.836516303737469;
//	w2(0,2)[0] = -0.224143868041857;
//	w2(0,3)[0] =  -0.129409522550921;
//
//	w3(0,0)[0] = 0.482962913144690;
//	w3(0,1)[0] = 0.836516303737469;
//	w3(0,2)[0] =  0.224143868041857;
//	w3(0,3)[0] =  -0.129409522550921;
//	w4(0,0)[0] = -0.129409522550921;
//	w4(0,1)[0] =  -0.224143868041857;
//	w4(0,2)[0] =  0.836516303737469;
//	w4(0,3)[0] = -0.482962913144690;


	int nlevels = 5;
	int ndims = 3;
	ML_MD_TD_FSystem<double> ml_md_fs(nlevels,ndims);
	ML_MD_TD_FSystem<double> ml_md_fs2(nlevels,ndims);

	for (int i = 0; i < nlevels; i++)
	{
		for (int j = 0; j < ndims; j++)
		{
			ml_md_fs.md_fs_at_level[i].oned_fs_at_dim[j].filters = SmartArray<OneD_TD_Filter<double> >(3);
			ml_md_fs.md_fs_at_level[i].oned_fs_at_dim[j].ds_folds = SmartIntArray(3, 2);
			ml_md_fs.md_fs_at_level[i].oned_fs_at_dim[j].flags = SmartArray<unsigned int>(2);
			ml_md_fs.md_fs_at_level[i].oned_fs_at_dim[j].flags[0] = LOWPASS_FILTER;
			ml_md_fs.md_fs_at_level[i].oned_fs_at_dim[j].flags[1] = HIGHPASS_FILTER;
			ml_md_fs.md_fs_at_level[i].oned_fs_at_dim[j].flags[2] = LOWPASS_FILTER2;
			ml_md_fs.md_fs_at_level[i].oned_fs_at_dim[j].filters[0].coefs = w1;
			ml_md_fs.md_fs_at_level[i].oned_fs_at_dim[j].filters[0].anchor = 0;
			ml_md_fs.md_fs_at_level[i].oned_fs_at_dim[j].filters[1].coefs = w2;
			ml_md_fs.md_fs_at_level[i].oned_fs_at_dim[j].filters[1].anchor = 0;
			ml_md_fs.md_fs_at_level[i].oned_fs_at_dim[j].filters[2].coefs = w1;
			ml_md_fs.md_fs_at_level[i].oned_fs_at_dim[j].filters[2].anchor = 0;

			ml_md_fs2.md_fs_at_level[i].oned_fs_at_dim[j].filters = SmartArray<OneD_TD_Filter<double> >(3);
			ml_md_fs2.md_fs_at_level[i].oned_fs_at_dim[j].ds_folds = SmartIntArray(3, 2);
			ml_md_fs2.md_fs_at_level[i].oned_fs_at_dim[j].flags = SmartArray<unsigned int>(2);
			ml_md_fs2.md_fs_at_level[i].oned_fs_at_dim[j].flags[0] = LOWPASS_FILTER;
			ml_md_fs2.md_fs_at_level[i].oned_fs_at_dim[j].flags[1] = HIGHPASS_FILTER;
			ml_md_fs2.md_fs_at_level[i].oned_fs_at_dim[j].flags[2] = LOWPASS_FILTER2;
			ml_md_fs2.md_fs_at_level[i].oned_fs_at_dim[j].filters[0].coefs = w3;
			ml_md_fs2.md_fs_at_level[i].oned_fs_at_dim[j].filters[0].anchor = 1;
			ml_md_fs2.md_fs_at_level[i].oned_fs_at_dim[j].filters[1].coefs = w4;
			ml_md_fs2.md_fs_at_level[i].oned_fs_at_dim[j].filters[1].anchor = 1;
			ml_md_fs2.md_fs_at_level[i].oned_fs_at_dim[j].filters[2].coefs = w3;
			ml_md_fs2.md_fs_at_level[i].oned_fs_at_dim[j].filters[2].anchor = 1;


		}
	}


	ML_MC_TD_Coefs_Set<double>::type coefs_set;
	ML_MC_TD_Filter_Norms_Set<double>::type norms_set;

	string msg;
	clock_t s0 = tic(), s1;
	decompose_in_time_domain2<double>(ml_md_fs, mat1, true, norms_set, coefs_set);
	s1 = tic();
	msg = show_elapse(s1 - s0);
	cout << "Dec: " << endl << msg << endl;
//	for (int i = 0; i < (int)coefs_set.size(); ++i)
//	{
//		for (int j = 0; j < (int)coefs_set[i].size(); ++j)
//		{
//			stringstream ss;
//			ss << "Test-Data/output/coef-" << i <<"-" <<j << ".png";
//			save_as_media<double>(ss.str(), coefs_set[i][j].coefs, NULL);
//		}
//	}

	Mat_<Vec<double, 2> > rec;
	reconstruct_in_time_domain<double>(ml_md_fs2, coefs_set, true, rec);
	s0 = tic();
	msg = show_elapse(s0 - s1);
	cout << "Rec: " << endl << msg << endl;
	double score, msr;
	psnr<double>(mat1, rec, score, msr);
	cout << "PSNR: score: " << score << ", msr: " << msr << endl;

//	save_as_media<double>("Test-Data/output/comp-supp-rec.png", rec, NULL);
	return 0;
}

int Unit_Test::performance_test(int argc, char **argv)
{


	const int M = 3258, N = 3258;
	const int step = N*2;
	double r = sqrt(2);
	double skerns[8*2] = {1/r,0, 1/r,0,1/r,0,1/r,0,1/r,0, 1/r,0,1/r,0,1/r,0};
//	double mat[M][N*2], mat2[M][N*2], mat3[M][N*2];
//	double kernels[M][N*2];
	double *mat, *mat2, *mat3;
	double *kernels;

	mat = new double[M*N*2];
	mat2 = new double[M*N*2];
	mat3 = new double[M*N*2];
	kernels = new double[M*N*2];
	memset(kernels, 0, sizeof(double) * 2 * M * N);
	kernels[0] = 1/r;
	kernels[1] = 0;
	kernels[2] = 1 / r;
	kernels[3] = 0;
	kernels[4] = 1 / r;
	kernels[5] = 0;
	kernels[6] = 1 / r;
	kernels[7] = 0;

	for (int i = 0; i < M; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			mat[i*step + j*2] = rand();
			mat[i*step + j*2 + 1] = 0;
		}
	}

	clock_t t0, t1;
	string msg;
	t0 = tic();
	for (int i = 0; i < M; ++i)
	{
		for (int j = 0; j < N; ++j)
		{

			double sum = 0.0;
			for (int k = 0; k < 8; ++k)
			{
				int good_j = j - k;
				if (good_j < 0)
				{
					good_j += N;
				}
				else if (good_j >= N)
				{
					good_j -= N;
				}
				sum += mat[i*step + good_j*2] * skerns[k*2];
			}
			mat2[i*step + j*2] = sum;
			mat2[i*step + j*2+1] = 0;
		}
	}
	for (int j = 0; j < N; ++j)
	{
		for (int i = 0; i < M; ++i)
		{

			double sum = 0.0;
			for (int k = 0; k < 8; ++k)
			{
				int good_i = i - k;
				if (good_i < 0)
				{
					good_i += M;
				}
				else if (good_i >= M)
				{
					good_i -= M;
				}
				sum += mat2[good_i * step + j*2] * skerns[k*2];
			}
			mat3[i*step + j*2] = sum;
			mat3[i*step + j*2+1] = 0;
		}
	}
	t1 = tic();
	msg = show_elapse(t1 - t0);
	cout << "Plain: " << endl << msg << endl;

	//////////////////

	fftw_complex *before;
	fftw_complex *after;
	fftw_plan plan;

	t0 = tic();


	before = reinterpret_cast<fftw_complex *>(mat);
	after = reinterpret_cast<fftw_complex *>(mat2);
	// Here we can only use 'FFTW_ESTIMATE', because 'FFTW_MEASURE' would touch 'before'.
	plan = fftw_plan_dft(2, (int[]){M,N}, before, after, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(plan);
	fftw_destroy_plan(plan);

//	before = reinterpret_cast<fftw_complex *>(kernels);
//	after = reinterpret_cast<fftw_complex *>(mat3);
//	// Here we can only use 'FFTW_ESTIMATE', because 'FFTW_MEASURE' would touch 'before'.
//	plan = fftw_plan_dft(2, (int[]){M,N}, before, after, FFTW_FORWARD, FFTW_ESTIMATE);
//	fftw_execute(plan);
//	fftw_destroy_plan(plan);

//	for (int i = 0; i < M; ++i)
//	{
//		for (int j = 0; j < N; ++j)
//		{
//			mat3[i*step + j*2] = mat3[i*step + j*2] * kernels[i*step + j*2] - mat3[i*step + j*2+1]*kernels[i*step + j*2+1];
//			mat3[i*step + j*2+1] = mat3[i*step + j*2] * kernels[i*step + j*2+1] + mat3[i*step + j*2+1]*kernels[i*step + j*2];
//		}
//	}

//	before = reinterpret_cast<fftw_complex *>(mat3);
//	after = reinterpret_cast<fftw_complex *>(mat3);
//	// Here we can only use 'FFTW_ESTIMATE', because 'FFTW_MEASURE' would touch 'before'.
//	plan = fftw_plan_dft(2, (int[]){M,N}, before, after, FFTW_BACKWARD, FFTW_ESTIMATE);
//	fftw_execute(plan);
//	fftw_destroy_plan(plan);

	t1 = tic();
	msg = show_elapse(t1 - t0);
	cout << "FFT: " << endl << msg << endl;

	return 0;
}
