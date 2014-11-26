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
	param.m = 1;

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
	param2.m = 1;

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
	double ctrl_param[6] = {-33.0/32.0, 69.0/128.0, 33.0/32.0, 69.0/128.0, M_PI, 51.0/512.0};
	int n = 3;
//	double ctrl_param[4] = {-M_PI / 3, M_PI / 8, M_PI / 3, M_PI / 8};
//	int n = 2;
//	double ctrl_param[4] = {-1, 0.4, 1, 0.4};
//	int n = 2;
	Mat x_pts;
	linspace(-M_PI, 0, M_PI, 0, 20, x_pts);

	Filter_Set filters;

	cout << "x_pts: " << endl;
	print_mat_details(x_pts);

	construct_1d_filter_banks(x_pts, n, ctrl_param, 1, "sqrt", filters);

	cout << "Filter Bank: " << endl;
	for (int i = 0; i < n; ++i)
	{
		cout << "Filter " << i << ":" << endl;
		print_mat_details(filters[i].filter);
	}

//	Chi_Ctrl_Param param;
//	param.cL = 1.0;
//	param.epL = 0.4;
//	param.cR = -1.0 + 2 * M_PI;
//	param.epR = 0.4;
//	param.m = 1;
//
//	Mat y_pts1, y_pts2;
//	fchi(x_pts, param, "sincos", y_pts1);
//	fchi(x_pts + Scalar(2 * M_PI, 0), param, "sincos", y_pts2);
////	print_mat_details(y_pts1);
////	print_mat_details(y_pts2);
//
//	cout << "Final: " << endl;
//	print_mat_details(y_pts1 + y_pts2);

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
	vector<SmartIntArray> support;
	downsample_in_fd_by2(filter, folds, folded_filter, support);

	cout << "Filter: " << endl;
	print_mat_details(filter);
	cout << "Folded: " << endl;
	print_mat_details(folded_filter);

	cout << "Support Set: " << endl;
	for (int i = 0; i < (int)support.size(); ++i)
	{
		cout << "[" << support[i][0] << "," << support[i][1] << "] ";
	}
	cout << endl;

	return 0;
}
