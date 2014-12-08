#ifndef _UNIT_TEST_H
#define _UNIT_TEST_H

class Unit_Test
{
public:
	static int mat_extension(int argc, char **argv);
	static int fchi_test(int argc, char **argv);
	static int fft_test(int argc, char **argv);
	static int construct_filter_test(int argc, char **argv);
	static int downsample_fd_by2_test(int argc, char **argv);
	static int tensor_product_test(int argc, char **argv);
	static int mat_select_test(int argc, char ** argv);
	static int mat_subfill_test(int argc, char **argv);

	static int decompose_test(int argc, char **argv);
	static int reconstruct_test(int argc, char **argv);

	static int partion_unity_test(int argc, char **argv);
	static int test_any(int argc, char **argv);
};

#endif
