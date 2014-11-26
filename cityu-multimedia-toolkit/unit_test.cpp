#include "unit_test.h"

#include "c-style/include/wavelets_tools.h"

int Unit_Test::mat_extension(int argc, char **argv)
{
	Mat mat(2, (int[]){5,5}, CV_64FC2, Scalar(0));
	mat.at<Vec2d>(0,0)[0] = 1;
	mat.at<Vec2d>(0,0)[1] = 1.2;
	mat.at<Vec2d>(0,1)[0] = 2;
	mat.at<Vec2d>(0,1)[1] = 2.2;
	mat.at<Vec2d>(0,3)[0] = 3;
	mat.at<Vec2d>(0,3)[1] = 3.2;
	mat.at<Vec2d>(1,0)[0] = 4;
	mat.at<Vec2d>(1,0)[1] = 4.2;
	mat.at<Vec2d>(1,2)[0] = 13;
	mat.at<Vec2d>(1,2)[1] = 13.2;
	mat.at<Vec2d>(1,4)[0] = 14;
	mat.at<Vec2d>(1,4)[1] = 14.2;
	mat.at<Vec2d>(3,0)[0] = 24;
	mat.at<Vec2d>(3,0)[1] = 24.2;
	mat.at<Vec2d>(3,2)[0] = 23;
	mat.at<Vec2d>(3,2)[1] = 23.2;
	mat.at<Vec2d>(3,4)[0] = 24;
	mat.at<Vec2d>(3,4)[1] = 24.2;
	mat.at<Vec2d>(4,0)[0] = 34;
	mat.at<Vec2d>(4,0)[1] = 34.2;
	mat.at<Vec2d>(4,2)[0] = 33;
	mat.at<Vec2d>(4,2)[1] = 33.2;
	mat.at<Vec2d>(4,4)[0] = 34;
	mat.at<Vec2d>(4,4)[1] = 34.2;

	Mat ext;
	mat_border_extension(mat, 2, (int[]){5,5}, "mir1", ext);

	print_mat_details(mat);
	print_mat_details(ext);

	Mat cut;
	mat_border_cut(ext, 2, (int[]){5,5}, cut);
	print_mat_details(cut);

	return 0;
}
