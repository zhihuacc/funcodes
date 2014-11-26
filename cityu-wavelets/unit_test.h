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
};

#endif
