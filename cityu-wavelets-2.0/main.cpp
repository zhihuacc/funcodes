#include "include/mat_toolbox.h"
#include "include/wavelets_toolbox.h"
#include "unit_test.h"
int main(int argc, char **argv)
{

	Unit_Test unit_test;
//	unit_test.psnr_test(argc, argv);

//	unit_test.denoising(argc, argv);

//	unit_test.mat_select_test(argc, argv);

//	unit_test.decomposition_test(argc, argv);

//	unit_test.reconstruction_test(argc, argv);

//	unit_test.construct_1d_filter_test(argc, argv);

//	unit_test.fft_center_shift_test(argc, argv);

	unit_test.test_any(argc, argv);
}
