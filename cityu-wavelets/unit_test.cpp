#include "unit_test.h"

#include "include/wavelets_tools.h"

#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

int Unit_Test::mat_extension(int argc, char **argv)
{
	Mat mat(2, (int[]){5,5}, CV_64FC2, Scalar(0));
//	mat.at<Vec2d>(0,0)[0] = 1;
//	mat.at<Vec2d>(0,0)[1] = 1.2;
//	mat.at<Vec2d>(0,1)[0] = 2;
//	mat.at<Vec2d>(0,1)[1] = 2.2;
//	mat.at<Vec2d>(0,3)[0] = 3;
//	mat.at<Vec2d>(0,3)[1] = 3.2;
//	mat.at<Vec2d>(1,0)[0] = 4;
//	mat.at<Vec2d>(1,0)[1] = 4.2;
//	mat.at<Vec2d>(1,2)[0] = 13;
//	mat.at<Vec2d>(1,2)[1] = 13.2;
//	mat.at<Vec2d>(1,4)[0] = 14;
//	mat.at<Vec2d>(1,4)[1] = 14.2;
//	mat.at<Vec2d>(3,0)[0] = 24;
//	mat.at<Vec2d>(3,0)[1] = 24.2;
//	mat.at<Vec2d>(3,2)[0] = 23;
//	mat.at<Vec2d>(3,2)[1] = 23.2;
//	mat.at<Vec2d>(3,4)[0] = 24;
//	mat.at<Vec2d>(3,4)[1] = 24.2;
//	mat.at<Vec2d>(4,0)[0] = 34;
//	mat.at<Vec2d>(4,0)[1] = 34.2;
//	mat.at<Vec2d>(4,2)[0] = 33;
//	mat.at<Vec2d>(4,2)[1] = 33.2;
//	mat.at<Vec2d>(4,4)[0] = 34;
//	mat.at<Vec2d>(4,4)[1] = 34.2;


	Media_Format mfmt;
	load_as_tensor("./test2.jpg", mat, &mfmt);

	Mat ext, cut;
	mat_border_extension(mat, 2, (int[]){100,100}, "mir1", ext);

//	print_mat_details(mat);

	mat_border_cut(ext, 2, (int[]){5,5}, cut);
//	cout << endl << "Mir1 ext: " << endl;
//	print_mat_details(ext);
//	cout << endl;
//	print_mat_details(cut);
//	cout << endl;

	save_as_media("./test2-mir1.jpg", ext, &mfmt);
	save_as_media("./test2-mir1-cut.jpg", cut, &mfmt);

	mat_border_extension(mat, 2, (int[]){100,100}, "mir0", ext);
	mat_border_cut(ext, 2, (int[]){100,100}, cut);

//	cout << "Mir0 ext: " << endl;
//	print_mat_details(ext);
//	cout << endl;
//	print_mat_details(cut);

	save_as_media("./test2-mir0.jpg", ext, &mfmt);
	save_as_media("./test2-mir0-cut.jpg", cut, &mfmt);


	return 0;
}

int Unit_Test::fchi_test(int argc, char **argv)
{
	Mat mat(2, (int[]){2, 10}, CV_64FC2, Scalar(0,0));

	Chi_Ctrl_Param param;
	param.cL = -M_PI / 3;
	param.epL = M_PI / 8;
	param.cR = M_PI / 3;
	param.epR = M_PI / 8;
	param.degree = 1;

	mat.at<Vec2d>(0,0)[0] = -11 * M_PI / 24;
	mat.at<Vec2d>(0,1)[0] = -10 * M_PI / 24;
	mat.at<Vec2d>(0,2)[0] = -8 * M_PI / 24;
	mat.at<Vec2d>(0,3)[0] = -7 * M_PI / 24;
	mat.at<Vec2d>(0,4)[0] = -6 * M_PI / 24;
	mat.at<Vec2d>(0,5)[0] = 5 * M_PI / 24;
	mat.at<Vec2d>(0,6)[0] = 6 * M_PI / 24;
	mat.at<Vec2d>(0,7)[0] = 8 * M_PI / 24;
	mat.at<Vec2d>(0,8)[0] = 9 * M_PI / 24;
	mat.at<Vec2d>(0,9)[0] = 10 * M_PI / 24;

	mat.at<Vec2d>(1,0)[0] = -11 * M_PI / 24;
	mat.at<Vec2d>(1,1)[0] = -10 * M_PI / 24;
	mat.at<Vec2d>(1,2)[0] = -8 * M_PI / 24;
	mat.at<Vec2d>(1,3)[0] = -7 * M_PI / 24;
	mat.at<Vec2d>(1,4)[0] = -6 * M_PI / 24;
	mat.at<Vec2d>(1,5)[0] = 5 * M_PI / 24;
	mat.at<Vec2d>(1,6)[0] = 6 * M_PI / 24;
	mat.at<Vec2d>(1,7)[0] = 8 * M_PI / 24;
	mat.at<Vec2d>(1,8)[0] = 9 * M_PI / 24;
	mat.at<Vec2d>(1,9)[0] = 10 * M_PI / 24;


	Mat y1, y2;
	fchi(mat, param, "sincos", y1);
	fchi(mat, param, "sqrt", y2);


	mat = Mat(2, (int[]){2, 10}, CV_64FC2, Scalar(0,0));


	mat.at<Vec2d>(0,0)[0] = 2 * M_PI -11 * M_PI / 24;
	mat.at<Vec2d>(0,1)[0] = 2 * M_PI -10 * M_PI / 24;
	mat.at<Vec2d>(0,2)[0] = 2 * M_PI -8 * M_PI / 24;
	mat.at<Vec2d>(0,3)[0] = 2 * M_PI -7 * M_PI / 24;
	mat.at<Vec2d>(0,4)[0] = 2 * M_PI -6 * M_PI / 24;
	mat.at<Vec2d>(0,5)[0] = 5 * M_PI / 24;
	mat.at<Vec2d>(0,6)[0] = 6 * M_PI / 24;
	mat.at<Vec2d>(0,7)[0] = 8 * M_PI / 24;
	mat.at<Vec2d>(0,8)[0] = 9 * M_PI / 24;
	mat.at<Vec2d>(0,9)[0] = 10 * M_PI / 24;

	mat.at<Vec2d>(1,0)[0] = 2 * M_PI -11 * M_PI / 24;
	mat.at<Vec2d>(1,1)[0] = 2 * M_PI -10 * M_PI / 24;
	mat.at<Vec2d>(1,2)[0] = 2 * M_PI -8 * M_PI / 24;
	mat.at<Vec2d>(1,3)[0] = 2 * M_PI -7 * M_PI / 24;
	mat.at<Vec2d>(1,4)[0] = 2 * M_PI -6 * M_PI / 24;
	mat.at<Vec2d>(1,5)[0] = 5 * M_PI / 24;
	mat.at<Vec2d>(1,6)[0] = 6 * M_PI / 24;
	mat.at<Vec2d>(1,7)[0] = 8 * M_PI / 24;
	mat.at<Vec2d>(1,8)[0] = 9 * M_PI / 24;
	mat.at<Vec2d>(1,9)[0] = 10 * M_PI / 24;

	Chi_Ctrl_Param param2;
	param2.cL = M_PI / 3;
	param2.epL = M_PI / 8;
	param2.cR = 5 * M_PI / 3;
	param2.epR = M_PI / 8;
	param2.degree = 1;

	Mat y3, y4;
	fchi(mat, param2, "sincos", y3);
	fchi(mat, param2, "sqrt", y4);


	cout << "Sin Cos: " << endl;
	cout << "Low: " << endl;
	print_mat_details(y1);
	cout << endl << "High: " << endl;
	print_mat_details(y3);

	cout << endl << "Sqrt: " << endl;
	cout << "Low: " << endl;
	print_mat_details(y2);
	cout << endl << "High: " << endl;
	print_mat_details(y4);



	return 0;
}

int Unit_Test::fft_test(int argc, char **argv)
{
	Mat mat(2, (int[]){2, 10}, CV_64FC2, Scalar(0,0));

	mat.at<Vec2d>(0,0)[0] = -11 * M_PI / 24;
	mat.at<Vec2d>(0,1)[0] = -10 * M_PI / 24;
	mat.at<Vec2d>(0,2)[0] = -8 * M_PI / 24;
	mat.at<Vec2d>(0,3)[0] = -7 * M_PI / 24;
	mat.at<Vec2d>(0,4)[0] = -6 * M_PI / 24;
	mat.at<Vec2d>(0,5)[0] = 5 * M_PI / 24;
	mat.at<Vec2d>(0,6)[0] = 6 * M_PI / 24;
	mat.at<Vec2d>(0,7)[0] = 8 * M_PI / 24;
	mat.at<Vec2d>(0,8)[0] = 9 * M_PI / 24;
	mat.at<Vec2d>(0,9)[0] = 10 * M_PI / 24;

	mat.at<Vec2d>(1,0)[0] = 11 * M_PI / 24;
	mat.at<Vec2d>(1,1)[0] = 10 * M_PI / 24;
	mat.at<Vec2d>(1,2)[0] = -8 * M_PI / 24;
	mat.at<Vec2d>(1,3)[0] = -7 * M_PI / 24;
	mat.at<Vec2d>(1,4)[0] = -6 * M_PI / 24;
	mat.at<Vec2d>(1,5)[0] = 5 * M_PI / 24;
	mat.at<Vec2d>(1,6)[0] = 6 * M_PI / 24;
	mat.at<Vec2d>(1,7)[0] = 8 * M_PI / 24;
	mat.at<Vec2d>(1,8)[0] = -9 * M_PI / 24;
	mat.at<Vec2d>(1,9)[0] = -10 * M_PI / 24;

	Mat freq, mat2;
	normalized_fft(mat, freq);
	normalized_ifft(freq, mat2);

	cout << "Time-Domain: " << endl;
	print_mat_details(mat);
	cout << endl;

	cout << "Freq-Domain: " << endl;
	print_mat_details(freq);
	cout << endl;

	cout << "Time-Domain: " << endl;
	print_mat_details(mat2);
	cout << endl;

	return 0;
}

int Unit_Test::construct_filter_test(int argc, char **argv)
{
	int nf = 3;
	OneD_Filter_System_Param fs_param;
	fs_param.ctrl_points = Smart64FArray(nf, (double[]){-33.0/32.0, 33.0/32.0, M_PI});
	fs_param.epsilons = Smart64FArray(nf, (double[]){69.0/128.0, 69.0/128.0,51.0/512.0});
	fs_param.degree = 1;
	fs_param.folds = SmartIntArray(nf, (int[]){2,2,2});
	fs_param.opt = "sincos";
	Mat x_pts;
	int size_of_filter = 100;
	linspace(-M_PI, 0, M_PI, 0, size_of_filter, x_pts);

	cout << "x_pts: " << endl;
	print_mat_details(x_pts);

//	construct_1d_filter_system(x_pts, n, ctrl_param, 1, "sqrt", filters);
	int nd = 2;
	OneD_Filter_System fs[nd];
	for (int i = 0; i < nd; ++i)
	{
	    construct_1d_filter_system(x_pts, fs_param, fs[i]);
	}

	cout << "Filter Bank: " << endl;

//	for (int i = 0; i < nd; ++i)
//	{
		for (int j = 0; j < fs_param.ctrl_points.len; ++j)
		{
			cout << "Filter " << j << ":" << endl;
			print_mat_details(fs[0][j].coefs);
		}
//	}


		Mat sum(nd, (int[]){1, size_of_filter}, CV_64FC2, Scalar(0,0));
		for (int i = 0; i < nf; ++i)
		{
			Mat sqr;
			pw_pow(fs[0][i].coefs, 2, sqr);
			sum += sqr;
		}

		print_mat_details_g<Vec2d>(sum, 0, "Test-Data/oned-sum.txt");

		int total = nf * nf;
		Mat md_filters[total];
		int k = 0;
		for (int i = 0; i < nf; ++i)
		{
			for (int j = 0; j < nf; ++j)
			{
				SmartArray<Mat> comps(2);
				comps[0] = fs[0][i].coefs;
				comps[1] = fs[1][j].coefs;
				tensor_product(comps, md_filters[k++]);
			}
		}


		SmartIntArray size(nd);
		size[0] = size_of_filter;
		size[1] = size_of_filter;
		Mat mat(nd, (int[]){size_of_filter, size_of_filter}, CV_64FC2, Scalar(0,0));
		for (int j = 0; j < (int)total; ++j)
		{
			Mat pow2;
			pw_pow(md_filters[j], 2, pow2);
			mat = mat + pow2;
		}

		double d0, d1, d2, d3;
		Point p0, p1, p2, p3;

		Mat channels[2];
		split(mat, channels);

		minMaxLoc(channels[0], &d0, &d1, &p0, &p1);
		minMaxLoc(channels[1], &d2, &d3, &p2, &p3);

		cout << "Real min-" << d0 << "    Real max-" << d1 << endl;
		cout << p0.y << "," << p0.x << "  " << p1.y << "," << p1.x << endl;
		cout << "Image min-" << d2 << "   Image max-" << d3 << endl;
		cout << p2.y << "," << p2.x << "  " << p3.y << "," << p3.x << endl << endl;

		print_mat_details_g<Vec<double, 2> >(mat, 0, "Test-Data/squar-sum.txt");

	return 0;
}

int Unit_Test::downsample_fd_by2_test(int argc, char **argv)
{
//	Mat filter(2, (int[]){8,8}, CV_64FC2, Scalar(0,0));
//	filter.at<Vec2d>(0,0)[0] = 1;
//	filter.at<Vec2d>(0,4)[0] = 2;
//	filter.at<Vec2d>(0,7)[0] = 1;
//	filter.at<Vec2d>(7,0)[0] = 3;
//	filter.at<Vec2d>(7,1)[0] = 2;
//	filter.at<Vec2d>(7,4)[0] = 1;
//	filter.at<Vec2d>(7,7)[0] = 4;
//	filter.at<Vec2d>(4,4)[0] = 3;
//	SmartIntArray folds(2);
//	folds[0] = 2;
//	folds[1] = 2;

	Mat filter(2, (int[]){1,8}, CV_64FC2, Scalar(0,0));
	filter.at<Vec2d>(0,3)[0] = 2;
	filter.at<Vec2d>(0,4)[0] = 1;
	SmartIntArray folds(2);
	folds[0] = 1;
	folds[1] = 2;

	Mat folded_filter;
	SmartArray<SmartIntArray> support;
	downsample_in_fd_by2(filter, folds, folded_filter, support);

	cout << "Filter: " << endl;
	print_mat_details(filter);
	cout << "Folded: " << endl;
	print_mat_details(folded_filter);

	cout << "Support Set: " << endl;
	for (int i = 0; i < (int)support.len; ++i)
	{
		cout << "[" << support[i][0] << "," << support[i][1] << "] ";
	}
	cout << endl;

	return 0;
}

int Unit_Test::tensor_product_test(int argc, char **argv)
{
	SmartArray<Mat> comps(3);

	comps[0] = Mat(2, (int[]){1,4}, CV_64FC2, (double[]){1,0,2,0,3,0,4,0});
	comps[1] = Mat(2, (int[]){1,3}, CV_64FC2, (double[]){2,0,3,0,4,0});
	comps[2] = Mat(2, (int[]){1,3}, CV_64FC2, (double[]){2,0,3,0,4,0});

	Mat product;
	tensor_product(comps, product);
//	print_mat_details(product);

	return 0;
}

int Unit_Test::mat_select_test(int argc, char ** argv)
{
	Mat_<Vec2d> test(2, (int[]){10,10}, Vec2d(0,0));
	test(0,1) = Vec2d(1,0);
	test(0,2) = Vec2d(2,1);
	test(0,3) = Vec2d(3,0.5);
	test(0,6) = Vec2d(6,0.5);
	test(0,7) = Vec2d(7,0.5);

	test(4,5) = Vec2d(4,5);
	test(4,6) = Vec2d(4,6);
	test(4,7) = Vec2d(4,7);
	test(6,5) = Vec2d(6,5);
	test(6,6) = Vec2d(6,6);
	test(6,7) = Vec2d(6,7);
	test(7,5) = Vec2d(7,5);
	test(7,6) = Vec2d(7,6);

	SmartArray<SmartIntArray> index_set_for_each_dim(2);
	index_set_for_each_dim[0] = SmartIntArray(3, (int[]){6,7,4});
	index_set_for_each_dim[1] = SmartIntArray(3, (int[]){5,6,7});

	Mat sel;
	mat_select(test, index_set_for_each_dim, sel);
	print_mat_details(sel);

	Scalar s;

	return 0;
}

int Unit_Test::mat_subfill_test(int argc, char ** argv)
{
	Mat_<Vec2d> test(2, (int[]){10,10}, Vec2d(0,0));
	test(0,1) = Vec2d(0,1);
	test(0,2) = Vec2d(0,2);
	test(0,3) = Vec2d(0,3);
	test(0,6) = Vec2d(0,6);
	test(0,7) = Vec2d(0,7);

	test(4,5) = Vec2d(4,5);
	test(4,6) = Vec2d(4,6);
	test(4,7) = Vec2d(4,7);
	test(6,5) = Vec2d(6,5);
	test(6,6) = Vec2d(6,6);
	test(6,7) = Vec2d(6,7);
	test(7,5) = Vec2d(7,5);
	test(7,6) = Vec2d(7,6);

	SmartArray<SmartIntArray> index_set_for_each_dim(2);
	index_set_for_each_dim[0] = SmartIntArray(3, (int[]){6,7,4});
	index_set_for_each_dim[1] = SmartIntArray(3, (int[]){5,6,7});

	Mat sel;
	mat_select(test, index_set_for_each_dim, sel);
	cout << "Origin mat: " << endl;
	print_mat_details(test);
	cout << "Selected submat: " << endl;
	print_mat_details(sel);

	Mat rec_mat(test.dims, test.size, test.type(), Scalar(0,0));
	Mat filled;
	mat_subfill(rec_mat, index_set_for_each_dim, sel, filled);
	cout << "Sub fill: " << endl;
	print_mat_details(filled);

	return 0;
}



int Unit_Test::decompose_test(int argc, char **argv)
{
	Mat mat;
	Media_Format mfm;
	load_as_tensor("Test-Data/Lena512.png", mat, &mfm);

//	MLevel_MDFilter_System_Param ml_md_fs_param;
//	ml_md_fs_param.md_fs_param_for_each_level.reserve(2);
//	ml_md_fs_param.md_fs_param_for_each_level[0].reserve(2);
//	ml_md_fs_param.md_fs_param_for_each_level[1].reserve(2);
//	for (int i = 0; i < ml_md_fs_param.md_fs_param_for_each_level.len; ++i)
//	{
//		for (int j = 0; j < ml_md_fs_param.md_fs_param_for_each_level[i].len; ++j)
//		{
//			ml_md_fs_param.md_fs_param_for_each_level[i][j].ctrl_points = Smart64FArray(6, (double[]){-2, -1.145796, 0, 1.145796, 2, M_PI});
//			ml_md_fs_param.md_fs_param_for_each_level[i][j].epsilons = Smart64FArray(6, (double[]){0.35, 0.3, 0.125, 0.3, 0.35, 0.0778});
//			ml_md_fs_param.md_fs_param_for_each_level[i][j].degree = 1;
//			ml_md_fs_param.md_fs_param_for_each_level[i][j].folds = SmartIntArray(6, (int[]){4,4,4,4,4,4});
//			ml_md_fs_param.md_fs_param_for_each_level[i][j].opt = "sincos";
//		}
//	}

	MLevel_MDFilter_System_Param ml_md_fs_param;
	ml_md_fs_param.md_fs_param_for_each_level.reserve(2);
	ml_md_fs_param.md_fs_param_for_each_level[0].reserve(2);
	ml_md_fs_param.md_fs_param_for_each_level[1].reserve(2);
//	ml_md_fs_param.md_fs_param_for_each_level[2].reserve(2);
	for (int i = 0; i < ml_md_fs_param.md_fs_param_for_each_level.len; ++i)
	{
		for (int j = 0; j < ml_md_fs_param.md_fs_param_for_each_level[i].len; ++j)
		{
			ml_md_fs_param.md_fs_param_for_each_level[i][j].ctrl_points = Smart64FArray(3, (double[]){-33.0/32.0, 33.0/32.0, M_PI});
			ml_md_fs_param.md_fs_param_for_each_level[i][j].epsilons = Smart64FArray(3, (double[]){69.0/128.0, 69.0/128.0, 51.0/512.0});
			ml_md_fs_param.md_fs_param_for_each_level[i][j].degree = 1;
			ml_md_fs_param.md_fs_param_for_each_level[i][j].folds = SmartIntArray(3, (int[]){2,2,2});
			ml_md_fs_param.md_fs_param_for_each_level[i][j].opt = "sincos";
		}
	}

	ML_MD_Filter_System ml_md_fs;
	ML_MChannel_Coefs_Set ml_mc_coefs_set;
	decompose_by_ml_md_filter_bank(ml_md_fs_param, mat, ml_md_fs, ml_mc_coefs_set);


	for (int i = 0; i < ml_md_fs_param.md_fs_param_for_each_level.len; ++i)
	{
		for (int j = 0; j < (int)ml_mc_coefs_set[i].size(); ++j)
		{
			stringstream ss;
			ss <<  "Test-Data/coef-" << i << "-" << j << ".jpg";
			save_as_media(ss.str(), ml_mc_coefs_set[i][j], &mfm);

			Mat mat;
			normalized_fft(ml_mc_coefs_set[i][j], mat);
			center_shift(mat, mat);
			stringstream ss2;
			ss2 << "Test-Data/fd-coef-" << i << "-" << j << ".jpg";
			save_as_media(ss2.str(), mat, &mfm);
		}
	}

//	for (int i = 0; i < ml_md_fs.len; ++i)
//	{
//		Mat mat(ml_md_fs[i].md_filters_coefs[0].dims, ml_md_fs[i].md_filters_coefs[0].size, CV_64FC2, Scalar(0,0));
//		for (int j = 0; j < ml_md_fs[i].md_filters_coefs.size(); ++j)
//		{
//			Mat pow2;
//			pow(ml_md_fs[i].md_filters_coefs[j], 2, pow2);
//			mat = mat + pow2;
//		}
//
//		double d0, d1;
//		minMaxLoc(mat, &)
//	}

	return 0;
}

int Unit_Test::partion_unity_test(int argc, char **argv)
{
	Mat mat;
	Media_Format mfm;
	load_as_tensor("Test-Data/Lena512.png", mat, &mfm);

//	MLevel_MDFilter_System_Param ml_md_fs_param;
//	ml_md_fs_param.md_fs_param_for_each_level.reserve(2);
//	ml_md_fs_param.md_fs_param_for_each_level[0].reserve(2);
//	ml_md_fs_param.md_fs_param_for_each_level[1].reserve(2);
//	for (int i = 0; i < ml_md_fs_param.md_fs_param_for_each_level.len; ++i)
//	{
//		for (int j = 0; j < ml_md_fs_param.md_fs_param_for_each_level[i].len; ++j)
//		{
//			ml_md_fs_param.md_fs_param_for_each_level[i][j].ctrl_points = Smart64FArray(6, (double[]){-2, -1.145796, 0, 1.145796, 2, M_PI});
//			ml_md_fs_param.md_fs_param_for_each_level[i][j].epsilons = Smart64FArray(6, (double[]){0.35, 0.3, 0.125, 0.3, 0.35, 0.0778});
//			ml_md_fs_param.md_fs_param_for_each_level[i][j].degree = 1;
//			ml_md_fs_param.md_fs_param_for_each_level[i][j].folds = SmartIntArray(6, (int[]){4,4,4,4,4,4});
//			ml_md_fs_param.md_fs_param_for_each_level[i][j].opt = "sincos";
//		}
//	}

	MLevel_MDFilter_System_Param ml_md_fs_param;
	ml_md_fs_param.md_fs_param_for_each_level.reserve(3);
	ml_md_fs_param.md_fs_param_for_each_level[0].reserve(2);
	ml_md_fs_param.md_fs_param_for_each_level[1].reserve(2);
	ml_md_fs_param.md_fs_param_for_each_level[2].reserve(2);
	for (int i = 0; i < ml_md_fs_param.md_fs_param_for_each_level.len; ++i)
	{
		for (int j = 0; j < ml_md_fs_param.md_fs_param_for_each_level[i].len; ++j)
		{
			ml_md_fs_param.md_fs_param_for_each_level[i][j].ctrl_points = Smart64FArray(3, (double[]){-33.0/32.0, 33.0/32.0, M_PI});
			ml_md_fs_param.md_fs_param_for_each_level[i][j].epsilons = Smart64FArray(3, (double[]){69.0/128.0, 69.0/128.0, 51.0/512.0});
			ml_md_fs_param.md_fs_param_for_each_level[i][j].degree = 1;
			ml_md_fs_param.md_fs_param_for_each_level[i][j].folds = SmartIntArray(3, (int[]){2,2,2});
			ml_md_fs_param.md_fs_param_for_each_level[i][j].opt = "sincos";
		}
	}

	ML_MD_Filter_System ml_md_fs;
	ML_MChannel_Coefs_Set ml_mc_coefs_set;
	decompose_by_ml_md_filter_bank(ml_md_fs_param, mat, ml_md_fs, ml_mc_coefs_set);


	for (int i = 0; i < ml_md_fs_param.md_fs_param_for_each_level.len; ++i)
	{
		for (int j = 0; j < (int)ml_mc_coefs_set[i].size(); ++j)
		{
			stringstream ss;
			ss <<  "Test-Data/coef-" << i << "-" << j << ".jpg";
			save_as_media(ss.str(), ml_mc_coefs_set[i][j], &mfm);

			Mat mat;
			normalized_fft(ml_mc_coefs_set[i][j], mat);
			center_shift(mat, mat);
			stringstream ss2;
			ss2 << "Test-Data/fd-coef-" << i << "-" << j << ".jpg";
			save_as_media(ss2.str(), mat, &mfm);
		}
	}

	for (int i = 0; i < ml_md_fs.len; ++i)
	{
		Mat mat(ml_md_fs[i].md_filters_coefs[0].dims, ml_md_fs[i].md_filters_coefs[0].size, CV_64FC2, Scalar(0,0));
		for (int j = 0; j < (int)ml_md_fs[i].md_filters_coefs.size(); ++j)
		{
			Mat pow2;
			pw_pow(ml_md_fs[i].md_filters_coefs[j], 2, pow2);
			mat = mat + pow2;
		}

		double d0, d1, d2, d3;
		Point p0, p1, p2, p3;

		Mat channels[2];
		split(mat, channels);

		minMaxLoc(channels[0], &d0, &d1, &p0, &p1);
		minMaxLoc(channels[1], &d2, &d3, &p2, &p3);

		cout << "Lvl-" << i <<endl;
//		int r = channels[0].rows, w = channels[0].cols;
//		cout << channels[0].at<double>(0, 0) << " " << channels[0].at<double>(r / 5, w / 4) << " " << channels[0].at<double>(r / 4, w / 4) << " " <<  channels[0].at<double>(r / 4, 2 * w / 4) << " " <<  channels[0].at<double>(r / 4, 3 * w / 4) << endl;
//		cout << channels[0].at<double>(3 * r / 5, 5 * w / 6) << " " << channels[0].at<double>(2 * r / 5, w / 4) << " "<< channels[0].at<double>(2 * r / 4, w / 4) << " " <<  channels[0].at<double>(2 * r / 4, 2 * w / 4) << " " <<  channels[0].at<double>(2 * r / 4, 3 * w / 4) << endl;
//		cout << channels[0].at<double>(4 * r / 5, 5 * w / 6) << " " << channels[0].at<double>(4 * r / 5, w / 4) << " "<< channels[0].at<double>(3 * r / 4, w / 4) << " " <<  channels[0].at<double>(3 * r / 4, 2 * w / 4) << " " <<  channels[0].at<double>(r - 1, w - 1) << endl;
		cout << "Real min-" << d0 << "    Real max-" << d1 << endl;
		cout << p0.y << "," << p0.x << "  " << p1.y << "," << p1.x << endl;
		cout << "Image min-" << d2 << "   Image max-" << d3 << endl;
		cout << p2.y << "," << p2.x << "  " << p3.y << "," << p3.x << endl << endl;

		print_mat_details_g<Vec<double, 1> >(channels[0], 0, "Test-Data/print-mat.txt");

	}

	return 0;
}

int Unit_Test::reconstruct_test(int argc, char **argv)
{
	Mat mat;
	Media_Format mfm;
	load_as_tensor("Test-Data/Lena512.png", mat, &mfm);

//	MLevel_MDFilter_System_Param ml_md_fs_param;
//	ml_md_fs_param.md_fs_param_for_each_level.reserve(2);
//	ml_md_fs_param.md_fs_param_for_each_level[0].reserve(2);
//	ml_md_fs_param.md_fs_param_for_each_level[1].reserve(2);
//	for (int i = 0; i < ml_md_fs_param.md_fs_param_for_each_level.len; ++i)
//	{
//		for (int j = 0; j < ml_md_fs_param.md_fs_param_for_each_level[i].len; ++j)
//		{
//			ml_md_fs_param.md_fs_param_for_each_level[i][j].ctrl_points = Smart64FArray(6, (double[]){-2, -1.145796, 0, 1.145796, 2, M_PI});
//			ml_md_fs_param.md_fs_param_for_each_level[i][j].epsilons = Smart64FArray(6, (double[]){0.35, 0.3, 0.125, 0.3, 0.35, 0.0778});
//			ml_md_fs_param.md_fs_param_for_each_level[i][j].degree = 1;
//			ml_md_fs_param.md_fs_param_for_each_level[i][j].folds = SmartIntArray(6, (int[]){4,4,4,4,4,4});
//			ml_md_fs_param.md_fs_param_for_each_level[i][j].opt = "sincos";
//		}
//	}

	int nlevels = 1;
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

	ML_MD_Filter_System ml_md_fs;
	ML_MChannel_Coefs_Set ml_mc_coefs_set;
	decompose_by_ml_md_filter_bank(ml_md_fs_param, mat, ml_md_fs, ml_mc_coefs_set);


//	for (int i = 0; i < ml_md_fs_param.md_fs_param_for_each_level.len; ++i)
//	{
//		for (int j = 0; j < (int)ml_mc_coefs_set[i].size(); ++j)
//		{
//			stringstream ss;
//			ss <<  "Test-Data/coef-" << i << "-" << j << ".jpg";
//			save_as_media(ss.str(), ml_mc_coefs_set[i][j], &mfm);
//
//			Mat mat;
//			normalized_fft(ml_mc_coefs_set[i][j], mat);
//			center_shift(mat, mat);
//			stringstream ss2;
//			ss2 << "Test-Data/fd-coef-" << i << "-" << j << ".jpg";
//			save_as_media(ss2.str(), mat, &mfm);
//		}
//	}

	Mat rec;
	reconstruct_by_ml_md_filter_bank(ml_md_fs_param, ml_mc_coefs_set, rec);
	save_as_media("Test-Data/rec-lena512.png", rec, &mfm);

//	Mat dif = mat - rec;
//	cout << "dif at  = " << dif.at<Vec2d>(10,200)[0] << ", " << dif.at<Vec2d>(10,200)[1] << endl;
//	Mat c[2];
//	split(dif, c);
//	double d0, d1, d2, d3;
//	minMaxLoc(c[0], &d0, &d1);
//	cout << "Real min: " << d0 << ", " << "max: " << d1 << endl;
//
//	minMaxLoc(c[1], &d0, &d1);
//	cout << "Img min: " << d0 << ", " << "max: " << d1 << endl;
	double score, msr;
	psnr(mat, rec, score, msr);
	cout << "PSNR: score: "<<score << ", msr: " << msr << endl;


//	normalized_fft(rec, rec);
//	center_shift(rec, rec);
//	save_as_media("Test-Data/rec-fd-lena512.png", rec, &mfm);
//


//	for (int i = 0; i < ml_md_fs.len; ++i)
//	{
//		Mat mat(ml_md_fs[i].md_filters_coefs[0].dims, ml_md_fs[i].md_filters_coefs[0].size, CV_64FC2, Scalar(0,0));
//		for (int j = 0; j < ml_md_fs[i].md_filters_coefs.size(); ++j)
//		{
//			Mat pow2;
//			pow(ml_md_fs[i].md_filters_coefs[j], 2, pow2);
//			mat = mat + pow2;
//		}
//
//		double d0, d1;
//		minMaxLoc(mat, &)
//	}

	return 0;
}

int Unit_Test::test_any(int argc, char **argv)
{
	SmartArray<Mat> ones_for_each_dim(2);
	for (int i = 0; i < 2; ++i)
	{
		int this_dim_coef_size = 6;
		Mat &ones_seq = ones_for_each_dim[i];
//						int seq_size[2] = {1, this_dim_coef_size};
		ones_seq.create(2, (int[]){1, this_dim_coef_size}, CV_64FC2);
		ones_seq.at<complex<double> >(0, 0) = complex<double>(1,0);
		for (int j = 1; j < this_dim_coef_size; ++j)
		{
			ones_seq.at<complex<double> >(0, j) = ones_seq.at<complex<double> >(0, j - 1) + (1.0);
		}
	}

	cout << "Ones array for dim 0: " << endl;
	print_mat_details(ones_for_each_dim[0]);
	cout << endl;

	Mat phase_change;
	tensor_product(ones_for_each_dim, phase_change);
	cout << "Phase matrix: " << endl;
	print_mat_details(phase_change);

	return 0;
}
