#ifndef _WAVELETS_TOOLS_H
#define _WAVELETS_TOOLS_H

#include <opencv2/core/core.hpp>
#include <string>
#include <vector>

#include "../include/hammers_and_nails.h"

using namespace cv;
using namespace std;

int normalized_fft(const Mat &time_domain, Mat &freq_domain);
int normalized_ifft(const Mat &freq_domain, Mat &time_domain);
int mat_border_extension(const Mat &origin, int n, const int *border, const string &opt, Mat &extended);
int mat_border_cut(const Mat &extended, int n, const int *border, Mat &origin);
int psnr(const Mat &left, const Mat &right, double &psnr, double &msr);

struct Chi_Ctrl_Param
{
	double cL;
	double cR;
	double epL;
	double epR;
	double m;
};

struct Filter_Info
{
	Mat filter;
	bool isLowPass;
	bool needShift;
	Chi_Ctrl_Param param;
};

typedef vector<Filter_Info> Filter_Set;

/* Construct a series of filter banks in frequency domain, given a set of control points.
 *
 * cp_num: Input param. The number of control points.
 * ctrl_pts: Input param. The values of control points and corresponding epsilons.
 *           ctrl_pts[i] is ctrl point and ctrl_pts[i + 1] is corresponding epsilon.
 * degree: Positve integear values
 * opt: Input param. Valid value include "sincos" and "sqrt". It indicates which formula is used.
 * output: output param. A series of filters returned.
 */
int construct_1d_filter_banks(const Mat &x_pts, int cp_num, const double *ctrl_pts, int degree, const string &opt, Filter_Set &ouput);

/*
 * Calculate Y values given X values(sample points), according the formula in the paper.
 *
 * x_pt: Input param. The given sample points.
 * param: Input param. The control points used to define the formula.
 * opt:  Input param. Vaild value include "sincos" and "sqrt". It indicates which formula will be used.
 */
int fchi(const Mat &x_pt, const Chi_Ctrl_Param &param, const string &opt, Mat &y_val);

int downsample_in_fd_by2(const Mat &filter, SmartIntArray &folds, Mat &folded_filter, vector<SmartIntArray> &support);

int linspace(double e1_r, double e1_i, double e2_r, double e2_i, int n, Mat &sample_points);

void print_mat_details(const Mat &mat, int field = 0, const string &filename = "cout");

struct Media_Format
{
	union {
		struct
		{
			int height;
			int width;
		} imgage_prop;
		struct
		{
			int fps;
			int fourcc;
			int frame_count;
			int frame_height;
			int frame_width;
		} video_prop;
	};
};

int load_as_tensor(const string &filename, Mat &output, Media_Format *media_file_fmt);
int save_as_media(const string &filename, const Mat &mat, const Media_Format *media_file_fmt);


#endif
