#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>



#include "include/wavelets_tools.h"
#include "unit_test.h"

#include <iostream>
#include <sstream>

using namespace cv;

int main(int argc, char **argv)
{


//	int size[2] = {1, 3};
//	complex<double> c(1,-1);
//	Mat_<std::complex<double> > mm1(2, size);
//	for (int i = 0; i < mm1.rows; ++i)
//	{
//		for (int j = 0; j < mm1.cols; ++j)
//		{
//			mm1.at<std::complex<double> >(i,j) = complex<double>(1, -2);
//		}
//	}
//
//	Mat_<std::complex<double> > mm2 = mm1.clone(), mm3;
//
//	mm3 = mm1.mul(mm2);
//
//	print_mat_details(mm3);


//	print_mat_details_g<complex<double> >(mm);
//
//	Mat_<Vec2d> u0(2, (int[]){1,2}, Vec2d(1,-2));
//	Mat_<Vec2d> v0(2, (int[]){1,2}, Vec2d(-3,2));
//	Mat_<Vec3d> v3d_mat(2, (int[]){1,3}, Vec3d(-3,2,1));
//
//
//	cout << "u: " << endl;
//	print_mat_details_g<Vec2d>(u0);
//	cout << "v: " << endl;
//	print_mat_details_g<Vec2d>(v0);
//
//	Mat_<Vec2d > product = u0.mul(v0);
//	cout << "product: " << endl;
//	print_mat_details_g<Vec2d>(product);
//
//	cout << "pow: " << endl;
//	pow(v3d_mat, 2, v3d_mat);
//	print_mat_details_g<Vec3d>(v3d_mat);
//	return 0;

//	Mat u(2, (int[]){3,3}, CV_64FC2, (double[]){1,2,3,3,2,1,1,2,3,3,2,1,1,2,3,3,2,1});
//	Mat v(2,(int[]){1,3}, CV_64FC2, (double[]){2,2,2,1,1,1});
//
//
//	cout << "u: " << endl;
//	print_mat_details(u);
//	cout << "v: " << endl;
//	print_mat_details(v);
//
//	Mat product = u * v;
//	cout << "product: " << endl;
//	print_mat_details(product);
//	return 0;

	Unit_Test unit;
//	unit.mat_extension(argc, argv);

//	unit.fchi_test(argc, argv);

//	unit.fft_test(argc, argv);

//	unit.construct_filter_test(argc, argv);

//	unit.downsample_fd_by2_test(argc, argv);

//	unit.tensor_product_test(argc, argv);

//	unit.mat_select_test(argc, argv);

//	unit.decompose_test(argc, argv);

	unit.reconstruct_test(argc, argv);

//	unit.mat_subfill_test(argc, argv);

//	unit.partion_unity_test(argc, argv);

//	unit.test_any(argc, argv);
}
