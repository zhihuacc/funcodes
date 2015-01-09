#ifndef UNIT_TEST_H
#define UNIT_TEST_H

class Unit_Test
{
public:
	static int decomposition_test(int argc, char **argv);
	static int reconstruction_test(int argc, char **argv);
	static int construct_1d_filter_test(int argc, char **argv);
	static int fft_center_shift_test(int argc, char **argv);
	static int mat_select_test(int argc, char **argv);
	static int denoising(int argc, char **argv);
	static int psnr_test(int argc, char **argv);
	static int test_any(int argc, char **argv);
};

#endif
