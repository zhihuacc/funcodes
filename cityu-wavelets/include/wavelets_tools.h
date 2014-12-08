#ifndef _WAVELETS_TOOLS_H
#define _WAVELETS_TOOLS_H

#include <opencv2/core/core.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>


#include "../include/hammers_and_nails.h"

using namespace cv;
using namespace std;

int normalized_fft(const Mat &time_domain, Mat &freq_domain);
int normalized_ifft(const Mat &freq_domain, Mat &time_domain);

int center_shift(const Mat &input, Mat &output);
int icenter_shift(const Mat &input, Mat &output);

int pw_mul(const Mat &left, const Mat &right, Mat &product);
int pw_pow(const Mat &base, double expo, Mat &res);
int pw_sqrt(const Mat &base, Mat &res);

int mat_border_extension(const Mat &origin, int n, const int *border, const string &opt, Mat &extended);
int mat_border_cut(const Mat &extended, int n, const int *border, Mat &origin);
int psnr(const Mat &left, const Mat &right, double &psnr, double &msr);

struct Chi_Ctrl_Param
{
	double cL;
	double cR;
	double epL;
	double epR;
	double degree;
};

struct OneD_Filter_System_Param
{
	Smart64FArray 	ctrl_points;
	Smart64FArray	epsilons;
	SmartIntArray	folds;
	int degree;
	string opt;
};

typedef SmartArray<OneD_Filter_System_Param>	MD_Filter_System_Param;

struct MLevel_MDFilter_System_Param
{
	SmartArray<MD_Filter_System_Param>		md_fs_param_for_each_level;
	SmartArray<SmartIntArray>				lowpass_approx_ds_folds;
	SmartArray<bool>						ml_downsample_flags;
};

typedef vector<Mat>	OneLevel_Coefs_Set;
typedef vector<OneLevel_Coefs_Set> ML_MChannel_Coefs_Set;


/* Construct a series of filter banks in frequency domain, given a set of control points.
 *
 * cp_num: Input param. The number of control points.
 * ctrl_pts: Input param. The values of control points and corresponding epsilons.
 *           ctrl_pts[i] is ctrl point and ctrl_pts[i + 1] is corresponding epsilon.
 * degree: Positve integear values
 * opt: Input param. Valid value include "sincos" and "sqrt". It indicates which formula is used.
 * output: output param. A series of filters returned.
 */
//int construct_1d_filter_system(const Mat &x_pts, int cp_num, const double *ctrl_pts, int degree, const string &opt, Filter_Set &ouput);

//int construct_1d_filter_system(const Mat &x_pts, const OneD_Filter_System_Param &oned_filter_system_param, OneD_Filter_System &output);

struct OneD_Filter
{
	Mat coefs;
	bool isLowPass;
	int fold;
	SmartIntArray support_after_ds;
};

typedef SmartArray<OneD_Filter> OneD_Filter_System;

//struct OneD_Filter_System2
//{
//	SmartArray<OneD_Filter>	oned_filters_set;
//	SmartIntArray					folds;
//};

struct MD_Filter_System
{
	SmartArray<OneD_Filter_System> 	oned_fs_for_each_dim;
	vector<Mat>				md_filters_coefs;			//Arrange filters by order of decomposition
	vector<SmartIntArray>	ds_folds_for_each_filter;
	vector<Mat>				md_filter_coefs_after_tp;
};

typedef SmartArray<MD_Filter_System>	ML_MD_Filter_System;


int construct_1d_filter_system(const Mat &x_pts, const OneD_Filter_System_Param &oned_filter_system_param, OneD_Filter_System &output);
/*
 * Calculate Y values given X values(sample points), according the formula in the paper.
 *
 * x_pt: Input param. The given sample points.
 * param: Input param. The control points used to define the formula.
 * opt:  Input param. Vaild value include "sincos" and "sqrt". It indicates which formula will be used.
 */
int fchi(const Mat &x_pt, const Chi_Ctrl_Param &param, const string &opt, Mat &y_val);

int downsample_in_fd_by2(const Mat &filter, SmartIntArray &folds, Mat &folded_filter, SmartArray<SmartIntArray> &support);


/*	Decompose 'input' by filter bank which is constructed according to 'filter_system_param'.
 *
 * filter_system_param:  Input param. All parameters which control what the filter bank looks like.
 * input:	Input param. Data to decompose.
 * coefs_set: Output param.  Approximate and detail coefs obtained.
 */
int decompose_by_ml_md_filter_bank(const MLevel_MDFilter_System_Param &filter_system_param, const Mat &input, ML_MD_Filter_System &ml_md_filter_system, ML_MChannel_Coefs_Set &coefs_set);

int reconstruct_by_ml_md_filter_bank(const MLevel_MDFilter_System_Param &filter_system_param, const ML_MChannel_Coefs_Set &coefs_set, Mat &rec);

int tensor_product(const SmartArray<Mat> &components_for_each_dim, Mat &product);

int mat_select(const Mat &origin_mat, const SmartArray<SmartIntArray> &index_set_for_each_dim, Mat &sub_mat);

int mat_subfill(const Mat &origin_mat, const SmartArray<SmartIntArray> &index_set_for_each_dim, const Mat &submat, Mat &filled_mat);

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

int linspace(double e1_r, double e1_i, double e2_r, double e2_i, int n, Mat &sample_points);

void print_mat_details(const Mat &mat, int field = 0, const string &filename = "cout");


template<class T>
void print_mat_details_g(const Mat_<T> &mat, int field = 0, const string &filename = "cout")
{
	streambuf *stdoutbuf = cout.rdbuf();
	ofstream outfile;
	if (filename != "cout")
	{
		outfile.open(filename.c_str(), ios_base::out | ios_base::app);
		cout.rdbuf(outfile.rdbuf());
	}
	int max_term = 8;
	bool first_row = true;

	int *pos = new int[mat.dims];
	const int *range = mat.size;;
	int dims = mat.dims;

	for (int i = 0; i < dims; i++)
	{
		pos[i] = 0;
	}

	cout << setiosflags(ios::fixed) << setprecision(3);
	int i = dims - 1;
	int t = 0;
	while(true)
	{
		while (i >= 0 && pos[i] >= range[i])
		{
			pos[i] = 0;
			--i;

			if (i >= 0)
			{
				++pos[i];
				continue;
			}
		}

		if (i < 0)
		{
			break;
		}

		if (pos[dims - 1] == 0){
			if (!first_row)
			{
				cout << endl << endl;
			}
			first_row = false;
			cout << "[";
			for (int idx = 0; idx < dims; ++idx)
			{
				cout << pos[idx];
				if (idx < dims - 1)
				{
					cout << ",";
				}
				else
				{
					cout << "]";
				}
			}
			cout << endl;
			t = 0;
		}


		int cn = mat.channels();
		if (field == 0)
		{
			cout << "(";
			for (int c = 0; c < cn; ++c)
			{
				cout << mat(pos)[c];
				if (c != cn - 1)
				{
					cout << ",";
				}
			}
			cout << ")";
		}
		else if (field == cn + 1)
		{
			double sum = 0.0;
			for (int i = 0; i < cn; ++i)
			{
				sum += pow(mat(pos)[i], 2);
			}

			cout << sqrt(sum);
		}
		else if (field > 0 && field <= cn)
		{
			cout << mat(pos)[field];
		}
		else
		{
			return;
		}
//		else if(field == 1)
//		{
//			cout << mat(pos)[0];
//		}
//		else if(field == 2)
//		{
//			cout << mat(pos)[1];
//		}
//		else if (field == 3)
//		{
//			cout << sqrt(pow(mat(pos)[0],2) + pow(mat(pos)[1], 2));
//		}
		if (t < max_term - 1)
		{
			cout << " ";
		}
		else if(t == max_term - 1)
		{
			cout << endl;
		}
		++t;
		t %= max_term;

		i = mat.dims - 1;
		++pos[i];
	}
	cout << endl;

	delete [] pos;

	if (filename != "cout")
	{
		cout.rdbuf(stdoutbuf);
	}
}


#endif
