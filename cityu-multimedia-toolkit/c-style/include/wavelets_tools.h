#ifndef _WAVELETS_TOOLS_H
#define _WAVELETS_TOOLS_H

#include <opencv2/core/core.hpp>
#include <string>

using namespace cv;
using namespace std;

int normalized_fft(const Mat &time_domain, Mat &freq_domain);
int normalized_ifft(const Mat &freq_domain, Mat &time_domain);
int mat_border_extension(const Mat &origin, int n, int *border, const string &opt, Mat &extended);
int mat_border_cut(const Mat &extended, int n, int *border, Mat &origin);

struct Chi_Ctrl_Param
{
	double eL;
	double eR;
	double epL;
	double epR;
};

int fchi(const Mat &x_pt, const string &opt, Mat &y_val);

void print_mat_details(const Mat &mat, const string &filename = "cout");

#endif
