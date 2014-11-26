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

	Unit_Test unit;
//	unit.mat_extension(argc, argv);

//	unit.fchi_test(argc, argv);

//	unit.fft_test(argc, argv);

//	unit.construct_filter_test(argc, argv);

	unit.downsample_fd_by2_test(argc, argv);
}
