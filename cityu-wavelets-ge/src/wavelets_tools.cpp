#include "../include/wavelets_tools.h"
#include "../include/math_helpers.h"

#include <fftw3.h>
#include <iostream>
#include <iomanip>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


int normalized_fft(const Mat &time_domain, Mat &freq_domain)
{
	int N = time_domain.total();
	if (time_domain.type() != CV_64FC2 || N < 1)
	{
		return -1;
	}

	fftw_complex *before;
	fftw_complex *after;
	fftw_plan plan;

	before = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
	after = before;     // in-place transform to save space.

	plan = fftw_plan_dft(time_domain.dims, time_domain.size, before, after, FFTW_FORWARD, FFTW_ESTIMATE);

	// Initialize input.
	MatConstIterator_<Vec2d> cit = time_domain.begin<Vec2d>();
	MatConstIterator_<Vec2d> cit_end = time_domain.end<Vec2d>();
	for(int i = 0; cit != cit_end; ++cit, ++i)
	{
		before[i][0] = (*cit)[0];
		before[i][1] = (*cit)[1];
	}


	fftw_execute(plan);

	// Prepare output
	Mat transformed_mat(time_domain.dims, time_domain.size, CV_64FC2);
	MatIterator_<Vec2d> it = transformed_mat.begin<Vec2d>();
	MatIterator_<Vec2d> it_end = transformed_mat.end<Vec2d>();
	for(int j = 0; it != it_end; ++it, ++j)
	{
		(*it)[0] = after[j][0];
		(*it)[1] = after[j][1];
	}

	freq_domain = transformed_mat * (1.0 / sqrt(N));

    fftw_destroy_plan(plan);
    fftw_free(before);

	return 0;
}

int normalized_ifft(const Mat &freq_domain, Mat &time_domain)
{
	int N = freq_domain.total();
	if (N <= 0)
	{
		return -1;
	}

	fftw_complex *before;
	fftw_complex *after;
	fftw_plan plan;

	before = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
	after = before;     // in-place transform to save space.

	plan = fftw_plan_dft(freq_domain.dims, freq_domain.size, before, after, FFTW_BACKWARD, FFTW_ESTIMATE);

	// Initialize input.
	MatConstIterator_<Vec2d> cit = freq_domain.begin<Vec2d>();
	MatConstIterator_<Vec2d> cit_end = freq_domain.end<Vec2d>();
	for(int i = 0; cit != cit_end; ++cit, ++i)
	{
		before[i][0] = (*cit)[0];
		before[i][1] = (*cit)[1];
	}


	fftw_execute(plan);

	// Prepare output
	Mat transformed_mat(freq_domain.dims, freq_domain.size, CV_64FC2);
	MatIterator_<Vec2d> it = transformed_mat.begin<Vec2d>();
	MatIterator_<Vec2d> it_end = transformed_mat.end<Vec2d>();
	for(int i = 0; it != it_end; ++it, ++i)
	{
		(*it)[0] = after[i][0];
		(*it)[1] = after[i][1];
	}

	time_domain = transformed_mat * (1.0 / sqrt(N));

    fftw_destroy_plan(plan);
    fftw_free(before);

	return 0;
}

int center_shift(const Mat &input, Mat &output)
{
	Mat mat = input;

	if (mat.empty())
	{
		return -1;
	}

	int *pos = new int[mat.dims];
	int *pos2 = new int[mat.dims];
	const int *range = mat.size;
	int *offset = new int[mat.dims];
	int dims = mat.dims;

	for (int i = 0; i < dims; i++)
	{
		pos[i] = 0;
		offset[i] = range[i] / 2;
		pos2[i] = offset[i];
	}

	Mat shifted(dims, range, CV_64FC2);

	int i = dims - 1;
	while(true)
	{
		while (i >= 0 && pos[i] >= range[i])
		{
			pos[i] = 0;
			pos2[i] = offset[i];
			--i;
			if (i >= 0)
			{
				++pos[i];
				pos2[i] = (pos[i] + offset[i]) % range[i];
				continue;
			}
		}

		if (i < 0)
		{
			break;
		}

		shifted.at<Vec2d>(pos2)[0] = mat.at<Vec2d>(pos)[0];
		shifted.at<Vec2d>(pos2)[1] = mat.at<Vec2d>(pos)[1];

		i = mat.dims - 1;
		++pos[i];
		pos2[i] = (pos[i] + offset[i]) % range[i];
	}

	delete [] pos;
	delete [] pos2;
	delete [] offset;

	output = shifted;

	return 0;
}
int icenter_shift(const Mat &input, Mat &output)
{
	Mat mat = input;

	if (mat.empty())
	{
		return -1;
	}

	int *pos = new int[mat.dims];
	int *pos2 = new int[mat.dims];
	const int *range = mat.size;
	int *offset = new int[mat.dims];
	int dims = mat.dims;

	for (int i = 0; i < dims; i++)
	{
		pos[i] = 0;
		offset[i] = (range[i] & 1) ? (range[i] / 2 + 1) : (range[i] / 2);
		pos2[i] = offset[i];
	}

	Mat shifted(dims, range, CV_64FC2);

	int i = dims - 1;
	while(true)
	{
		while (i >= 0 && pos[i] >= range[i])
		{
			pos[i] = 0;
			pos2[i] = offset[i];
			--i;
			if (i >= 0)
			{
				++pos[i];
				pos2[i] = (pos[i] + offset[i]) % range[i];
				continue;
			}
		}

		if (i < 0)
		{
			break;
		}

		shifted.at<Vec2d>(pos2)[0] = mat.at<Vec2d>(pos)[0];
		shifted.at<Vec2d>(pos2)[1] = mat.at<Vec2d>(pos)[1];

		i = mat.dims - 1;
		++pos[i];
		pos2[i] = (pos[i] + offset[i]) % range[i];
	}

	delete [] pos;
	delete [] pos2;
	delete [] offset;

	output = shifted;

	return 0;
}

// Work only for Vec2d
int pw_mul(const Mat &left, const Mat &right, Mat &product)
{
    // Need more check
    if (left.dims != right.dims)
    {
    	return -1;
    }

    Mat product_mat(left.dims, left.size, CV_64FC2);
    MatIterator_<Vec2d> it0 = product_mat.begin<Vec2d>(), end0 = product_mat.end<Vec2d>();
    MatConstIterator_<Vec2d> it1 = left.begin<Vec2d>(), it2 = right.begin<Vec2d>();

    for (; it0 != end0; ++it0, ++it1, ++it2)
    {
    	(*it0)[0] = (*it1)[0] * (*it2)[0] - (*it1)[1] * (*it2)[1];
    	(*it0)[1] = (*it1)[0] * (*it2)[1] + (*it1)[1] * (*it2)[0];
    }

    product = product_mat;

    return 0;
}

// Work only for Vec2d
int pw_pow(const Mat &base, double expo, Mat &res)
{


    Mat res_mat(base.dims, base.size, CV_64FC2);
    MatIterator_<Vec2d> it0 = res_mat.begin<Vec2d>(), end0 = res_mat.end<Vec2d>();
    MatConstIterator_<Vec2d> it1 = base.begin<Vec2d>();

    for (; it0 != end0; ++it0, ++it1)
    {
    	Vec2d elem = *it1;
    	complex<double> *b = reinterpret_cast<complex<double> *>(&elem);
    	complex<double> c;
    	c = pow(*b, expo);
    	*it0 = *reinterpret_cast<Vec2d*>(&c);
    }

    res = res_mat;
	return 0;
}

int pw_sqrt(const Mat &base, Mat &res)
{
    Mat res_mat(base.dims, base.size, CV_64FC2);
    MatIterator_<Vec2d> it0 = res_mat.begin<Vec2d>(), end0 = res_mat.end<Vec2d>();
    MatConstIterator_<Vec2d> it1 = base.begin<Vec2d>();

    for (; it0 != end0; ++it0, ++it1)
    {
    	Vec2d elem = *it1;
    	complex<double> *b = reinterpret_cast<complex<double> *>(&elem);
    	complex<double> c;
    	c = sqrt(*b);
    	*it0 = *reinterpret_cast<Vec2d*>(&c);
    }

    res = res_mat;
	return 0;
}


int mat_border_extension(const Mat &origin, int n, const int *border, const string &opt, Mat &extended)
{
	if (n < 1 || border == NULL || n != origin.dims)
	{
		return -1;
	}

	//TODO: Check if border[i] is less than origin.size[i]

	int bd_mode = -1;
	if (opt == "sym")
	{
		bd_mode = 0;
	}
	else if (opt == "mir0")
	{
		bd_mode = 1;
	}
	else if (opt == "mir1")
	{
		bd_mode = 2;
	}
	else if (opt == "blk")
	{
		bd_mode = 3;
	}
	{
		return -2;
	}

	const int *origin_size = origin.size;
	int *dst_pos;
	int *shift;
	int *start_pos;
	int *cur_pos;
	int *step;
	int *ext_size;


	shift = new int[n];
	dst_pos = new int[n];
	start_pos = new int[n];
	cur_pos = new int[n];
	step = new int[n];
	ext_size = new int[n];
	for (int i = 0; i < n; ++i)
	{
		shift[i] = border[i] / 2;
		start_pos[i] = 0;
		cur_pos[i] = start_pos[i];
		dst_pos[i] = start_pos[i] - shift[i];
		step[i] = 1;
		ext_size[i] = origin_size[i] + border[i];
	}

	Mat ext_mat(n, ext_size, CV_64FC2, Scalar(0,0));

	{
		int dims;
		int *src_start_pos;
		int *src_cur_pos;
		int *src_step;
		int *src_end_pos;


		//User-Defined initialization
		dims = n;
		src_start_pos = start_pos;
		src_cur_pos = cur_pos;
		src_step = step;
		src_end_pos = ext_size;
		//--

		int cur_dim = dims - 1;
		while(true)
		{
			while (cur_dim >= 0 && src_cur_pos[cur_dim] >= src_end_pos[cur_dim])
			{
				src_cur_pos[cur_dim] = src_start_pos[cur_dim];
				--cur_dim;
				if (cur_dim >= 0)
				{
					src_cur_pos[cur_dim] += src_step[cur_dim];

					continue;
				}
			}

			if (cur_dim < 0)
			{
				break;
			}

			//User-Defined actions
			bool invld = false;
			for (cur_dim = 0; cur_dim < dims; ++cur_dim)
			{
				dst_pos[cur_dim] = src_cur_pos[cur_dim] - shift[cur_dim];

				if (bd_mode == 0)  //sym
				{
					if (dst_pos[cur_dim] < 0)
					{
						dst_pos[cur_dim] += origin_size[cur_dim];
					}
					else if (dst_pos[cur_dim] >= origin_size[cur_dim])
					{
						dst_pos[cur_dim] -= origin_size[cur_dim];
					}
				}
				else if (bd_mode == 1)  //mir1, no duplicate for first and last elements.
				{
					if (dst_pos[cur_dim] < 0)
					{
						dst_pos[cur_dim] = -dst_pos[cur_dim];
					}
					else if (dst_pos[cur_dim] >= origin_size[cur_dim])
					{
						//This requires each dim is no less than 2
						dst_pos[cur_dim] = 2 * origin_size[cur_dim] - 2 - dst_pos[cur_dim];
					}
				}
				else if (bd_mode == 2)  //mir2
				{
					if (dst_pos[cur_dim] < 0)
					{
						dst_pos[cur_dim] = -dst_pos[cur_dim] - 1;
					}
					else if (dst_pos[cur_dim] >= origin_size[cur_dim])
					{
						//This requires each dim is no less than 2
						dst_pos[cur_dim] = 2 * origin_size[cur_dim] - 1 - dst_pos[cur_dim];
					}
				}
				else if (bd_mode == 3)  //blk
				{
					if (dst_pos[cur_dim] < 0 || dst_pos[cur_dim] >= origin_size[cur_dim])
					{
						invld = true;
						break;
					}
				}

			}

			if (invld)
			{
				ext_mat.at<Vec2d>(src_cur_pos)[0] = 0;
				ext_mat.at<Vec2d>(src_cur_pos)[1] = 0;
			}
			else
			{
				ext_mat.at<Vec2d>(src_cur_pos)[0] = origin.at<Vec2d>(dst_pos)[0];
				ext_mat.at<Vec2d>(src_cur_pos)[1] = origin.at<Vec2d>(dst_pos)[1];
			}
			//--

			cur_dim = dims - 1;
			src_cur_pos[cur_dim] += src_step[cur_dim];
		}

	}

	delete [] dst_pos;
	delete [] shift;
	delete [] start_pos;
	delete [] cur_pos;
	delete [] step;
	delete [] ext_size;

	extended = ext_mat;

	return 0;
}

int mat_border_cut(const Mat &extended, int n, const int *border, Mat &origin)
{
	if (n < 1 || border == NULL || n != extended.dims)
	{
		return -1;
	}

	//TODO: check if size[i] > border[i]


	const int *ext_size = extended.size;
	int *dst_start_pos = new int[n];
	int *dst_pos = new int[n];
	int *start_pos = new int[n];
	int *cur_pos = new int[n];
	int *step = new int[n];
	int *end_pos = new int[n];
	for (int i = 0; i < n; ++i)
	{
		start_pos[i] = 0;
		cur_pos[i] = start_pos[i];
		step[i] = 1;
		end_pos[i] = ext_size[i] - border[i];
		dst_start_pos[i] = border[i] / 2;
		dst_pos[i] = dst_start_pos[i];
	}

	Mat cut_mat(n, end_pos, CV_64FC2, Scalar(0,0));

	{
		int dims;
		int *src_start_pos;
		int *src_cur_pos;
		int *src_step;
		int *src_end_pos;


		//User-Defined initialization
		dims = n;
		src_start_pos = start_pos;
		src_cur_pos = cur_pos;
		src_step = step;
		src_end_pos = end_pos;
		//--

		int cur_dim = dims - 1;
		while(true)
		{
			while (cur_dim >= 0 && src_cur_pos[cur_dim] >= src_end_pos[cur_dim])
			{
				src_cur_pos[cur_dim] = src_start_pos[cur_dim];
				--cur_dim;
				if (cur_dim >= 0)
				{
					src_cur_pos[cur_dim] += src_step[cur_dim];

					continue;
				}
			}

			if (cur_dim < 0)
			{
				break;
			}

			//User-Defined actions
			for (; cur_dim < dims; ++cur_dim)
			{
				dst_pos[cur_dim] = src_cur_pos[cur_dim] + dst_start_pos[cur_dim];
			}
			cut_mat.at<Vec2d>(src_cur_pos)[0] = extended.at<Vec2d>(dst_pos)[0];
			cut_mat.at<Vec2d>(src_cur_pos)[1] = extended.at<Vec2d>(dst_pos)[1];
			//--

			cur_dim = dims - 1;
			src_cur_pos[cur_dim] += src_step[cur_dim];
		}
	}

	delete [] dst_start_pos;
	delete [] dst_pos;
	delete [] start_pos;
	delete [] cur_pos;
	delete [] step;
	delete [] end_pos;

	origin = cut_mat;

	return 0;
}

void print_mat_details(const Mat &mat, int field, const string &filename)
{
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


		if (field == 0)
		{
			cout << "(" << mat.at<Vec2d>(pos)[0] << ","
						<< mat.at<Vec2d>(pos)[1] << ")";
		}
		else if(field == 1)
		{
			cout << mat.at<Vec2d>(pos)[0];
		}
		else if(field == 2)
		{
			cout << mat.at<Vec2d>(pos)[1];
		}
		else if (field == 3)
		{
			cout << sqrt(pow(mat.at<Vec2d>(pos)[0],2) + pow(mat.at<Vec2d>(pos)[1], 2));
		}
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
}


static double sincos_bump(double x, int m)
{
	double f;
	if (x <= 0 || x >= 1)
	{
		f = 0.0;
	}
	else
	{
		f = 0.0;
		for (int j = 0; j < m; ++j)
		{
			f += nchoosek(m - 1 + j, j) * pow(x, j);
		}

		f *= pow(1 - x, m);
		f = sin(0.5 * M_PI * f);
	}

	return f;
}

static double sqrt_bump(double x, int m)
{
	double f;
	if (x <= 0 || x >= 1)
	{
		f = 0.0;
	}
	else
	{
		f = 0.0;
		for (int k = 0; k < m; ++k)
		{
			f += (k&1 ? -1 : 1) * nchoosek(2 * m - 1, m - 1 - k) * nchoosek(m - 1 + k, k) * pow(x, k);
		}

		f *= pow(x, m);
		f = sqrt(f);
	}

	return f;
}

int fchi(const Mat &x_pt, const Chi_Ctrl_Param &ctrl_param, const string &opt, Mat &y_val)
{
	Chi_Ctrl_Param param = ctrl_param;

	Mat mat(x_pt.dims, x_pt.size, CV_64FC2, Scalar(0, 0));

	if (opt == "sincos")
	{
		MatConstIterator_<Vec2d> it1 = x_pt.begin<Vec2d>(), end1 = x_pt.end<Vec2d>();
		MatIterator_<Vec2d> it2 = mat.begin<Vec2d>();
		for (; it1 != end1; ++it1, ++it2)
		{
			double w = (*it1)[0];
			double f = 0.0;
			if (w <= (param.cL - param.epL) || w >= (param.cR + param.epR))
			{
				f = 0.0;
			}
			else if (w >= (param.cL + param.epL) && w <= (param.cR - param.epR))
			{
				f = 1.0;
			}
			else if (w > (param.cL - param.epL) && w < (param.cL + param.epL))
			{
				double r = (param.cL - w) / param.epL;
				f = sincos_bump((1 + r) / 2, param.degree);
			}
			else if (w > (param.cR - param.epR) && w < (param.cR + param.epR))
			{
				double r = (w - param.cR) / param.epR;
				f = sincos_bump((1 + r) / 2, param.degree);
			}

			(*it2)[0] = f;
		}
	}
	else if (opt == "sqrt")
	{
		MatConstIterator_<Vec2d> it1 = x_pt.begin<Vec2d>(), end1 = x_pt.end<Vec2d>();
		MatIterator_<Vec2d> it2 = mat.begin<Vec2d>();
		for (; it1 != end1; ++it1, ++it2)
		{
			double w = (*it1)[0];
			double f = 0.0;

			if (w <= (param.cL - param.epL) || w >= (param.cR + param.epR))
			{
				f = 0.0;
			}
			else if (w >= (param.cL + param.epL) && w <= (param.cR - param.epR))
			{
				f = 1.0;
			}
			else if (w > (param.cL - param.epL) && w < (param.cL + param.epL))
			{
				double r = (param.cL - w) / param.epL;
				f = sqrt_bump((1 + r)  /2, param.degree);
			}
			else if (w > (param.cR - param.epR) && w < (param.cR + param.epR))
			{
				double r = (w - param.cR) / param.epR;
				f = sqrt_bump((1 + r) / 2, param.degree);
			}

			(*it2)[0] = f;
		}
	}
	else
	{
		return -2;
	}

	y_val = mat;

	return 0;
}

int construct_1d_filter_system(const Mat &x_pts, const OneD_Filter_System_Param &oned_filter_system_param, OneD_Filter_System &output)
{
	int ctrl_pts_num = oned_filter_system_param.ctrl_points.len;
	if (ctrl_pts_num < 2)
	{
		return -1;
	}
	OneD_Filter_System filter_system(ctrl_pts_num);
	const Smart64FArray &ctrl_points = oned_filter_system_param.ctrl_points;
	const Smart64FArray &epsilons = oned_filter_system_param.epsilons;
	const string &opt = oned_filter_system_param.opt;

//	filters.reserve(ctrl_pts_num);
//	filters.resize(ctrl_pts_num);

	Mat shift_right_x = x_pts.clone();
	shift_right_x += Scalar(-2 * M_PI, 0);

	Mat shift_left_x = x_pts.clone();
	shift_left_x += Scalar(2 * M_PI, 0);

	for (int i = 0; i < ctrl_pts_num; ++i)
	{
		OneD_Filter &this_filter_block = filter_system[i];
		Chi_Ctrl_Param this_filter_param;
		if (i != ctrl_pts_num - 1)
		{
			this_filter_param.cL = ctrl_points[i];
			this_filter_param.epL = epsilons[i];
			this_filter_param.cR = ctrl_points[i + 1];
			this_filter_param.epR = epsilons[i + 1];
		}
		else
		{
			this_filter_param.cL = ctrl_points[i];
			this_filter_param.epL = epsilons[i];
			this_filter_param.cR = ctrl_points[0];
			this_filter_param.epR = epsilons[0];
		}
		this_filter_param.degree = oned_filter_system_param.degree;

		if (this_filter_param.cR < this_filter_param.cL)
		{
			this_filter_param.cR += 2 * M_PI;
		}

		this_filter_block.isLowPass = false;
		if (this_filter_param.cL - this_filter_param.epL < 0 && this_filter_param.cR + this_filter_param.epR >= 0)
		{
			this_filter_block.isLowPass = true;
		}

		Mat shift_right_filter;
		Mat shift_left_filter;

		fchi(shift_right_x, this_filter_param, opt, shift_right_filter);
		fchi(x_pts, this_filter_param, opt, this_filter_block.coefs);
		fchi(shift_left_x, this_filter_param, opt, shift_left_filter);

		this_filter_block.coefs = this_filter_block.coefs + shift_right_filter + shift_left_filter;

		SmartIntArray ds_folds(2);
		ds_folds[0] = 1;
		ds_folds[1] = oned_filter_system_param.folds[i];

		Mat dummy;
		SmartArray<SmartIntArray> supp;
		downsample_in_fd_by2(this_filter_block.coefs, ds_folds, dummy, supp);

		SmartIntArray reduced_supp(supp.len);
		for (int i = 0; i < supp.len; ++i)
		{
			reduced_supp[i] = supp[i][1];
		}
		this_filter_block.support_after_ds = reduced_supp;
	}

	output = filter_system;

	return 0;
}

int linspace(double e1_r, double e1_i, double e2_r, double e2_i, int n, Mat &sample_points)
{
	if (n < 1)
	{
		return -1;
	}


	double intvl1 = (e2_r - e1_r) / n, intvl2 = (e2_i - e1_i) / n;
	Mat samples(2, (int[]){1,n}, CV_64FC2, Scalar(0,0));
	for (int i = 0; i < n; ++i)
	{
		samples.at<Vec2d>(0,i)[0] = e1_r + i * intvl1;
		samples.at<Vec2d>(0,i)[1] = e1_i + i * intvl2;
	}

	sample_points = samples;

	return 0;
}


int downsample_in_fd_by2(const Mat &filter, SmartIntArray &folds, Mat &folded_filter, SmartArray<SmartIntArray> &support)
{
	if (filter.dims != folds.len || filter.type() != CV_64FC2)
	{
		return -1;
	}

	int dims = filter.dims;
	SmartIntArray start_pos1(dims);
	SmartIntArray cur_pos1(dims);
	SmartIntArray step1(dims);
	SmartIntArray origin_range(dims, filter.size);
	SmartIntArray start_pos2(dims);
	SmartIntArray cur_pos2(dims);
	SmartIntArray folded_range(dims);

	int folded_total = 1;
	for (int i = 0; i < dims; ++i)
	{
		step1[i] = 1;
		folded_range[i] = (origin_range[i] < 2) ? 1 : (origin_range[i] / folds[i]);
		folded_total *= folded_range[i];
	}

	SmartArray<SmartIntArray> supp_set(folded_total);
	Mat folded_mat(dims, (const int*)folded_range, CV_64FC2, Scalar(0,0));
	{
		int src_dims;
		SmartIntArray src_start_pos;
		SmartIntArray src_cur_pos;
		SmartIntArray src_step;
		SmartIntArray src_end_pos;


		//User-Defined initialization
		src_dims = dims;
		src_start_pos = start_pos1;
		src_cur_pos = cur_pos1;
		src_step = step1;
		src_end_pos = folded_range;
		int supp_idx = 0;
		//--

		int cur_dim = src_dims - 1;
		while(true)
		{
			while (cur_dim >= 0 && src_cur_pos[cur_dim] >= src_end_pos[cur_dim])
			{
				src_cur_pos[cur_dim] = src_start_pos[cur_dim];
				--cur_dim;
				if (cur_dim >= 0)
				{
					src_cur_pos[cur_dim] += src_step[cur_dim];

					continue;
				}
			}

			if (cur_dim < 0)
			{
				break;
			}

			//User-Defined actions
			start_pos2 = src_cur_pos;
			cur_pos2 = src_cur_pos.clone();
			complex<double> sum;
			supp_set[supp_idx] = (src_cur_pos.clone());
			{
				int src_dims;
				SmartIntArray src_start_pos;
				SmartIntArray src_cur_pos;
				SmartIntArray src_step;
				SmartIntArray src_end_pos;


				//User-Defined initialization
				src_dims = dims;
				src_start_pos = start_pos2;
				src_cur_pos = cur_pos2;
				src_step = folded_range;
				src_end_pos = origin_range;
				//--

				int cur_dim = src_dims - 1;
				while(true)
				{
					while (cur_dim >= 0 && src_cur_pos[cur_dim] >= src_end_pos[cur_dim])
					{
						src_cur_pos[cur_dim] = src_start_pos[cur_dim];
						--cur_dim;
						if (cur_dim >= 0)
						{
							src_cur_pos[cur_dim] += src_step[cur_dim];

							continue;
						}
					}

					if (cur_dim < 0)
					{
						break;
					}

					complex<double> ele = filter.at<complex<double> >(src_cur_pos);
					if (ele.real() > 0.0 || ele.imag() > 0.0)
					{
						supp_set[supp_idx] = src_cur_pos.clone();
					}
					sum += ele;

					cur_dim = dims - 1;
					src_cur_pos[cur_dim] += src_step[cur_dim];
				}

			}
			folded_mat.at<complex<double> >(src_cur_pos) = sum;
			++supp_idx;
			//--

			cur_dim = src_dims - 1;
			src_cur_pos[cur_dim] += src_step[cur_dim];
		}

	}

	folded_filter = folded_mat;
	support = supp_set;

	return 0;
}

int decompose_by_ml_md_filter_bank(const MLevel_MDFilter_System_Param &filter_system_param, const Mat &input, ML_MD_Filter_System &ml_md_filter_system, ML_MChannel_Coefs_Set &coefs_set)
{

	if (filter_system_param.md_fs_param_for_each_level.len < 1
		|| input.dims != filter_system_param.md_fs_param_for_each_level[0].len)
	{
		return -1;
	}

	int levels = filter_system_param.md_fs_param_for_each_level.len;
	int input_dims = input.dims;
//	SmartIntArray origin_input_size(input_dims, input.size);
//	SmartIntArray cur_lvl_input_size = origin_input_size.clone();
	SmartIntArray step1(input_dims);
	for (int i = 0; i < input_dims; ++i)
	{
		step1[i] = 1;
	}

	ml_md_filter_system.reserve(levels);
	coefs_set.reserve(levels);
	coefs_set.resize(levels);
	// Every Level
	for (int cur_lvl = 0; cur_lvl < levels; ++cur_lvl)
	{
		Mat last_approx;
		if (cur_lvl == 0)
		{
			Mat fd_input;
			normalized_fft(input, fd_input);
//			save_as_media("Test-Data/freq-origin.jpg", fd_input, NULL);
			center_shift(fd_input, fd_input);
//			save_as_media("Test-Data/freq-shifted.jpg", fd_input, NULL);
			last_approx = fd_input;
		}
		else
		{
			last_approx = coefs_set[cur_lvl - 1][coefs_set[cur_lvl - 1].size() - 1];
			coefs_set[cur_lvl - 1].pop_back();
		}
		SmartIntArray this_level_filter_size_for_each_dim(input_dims, last_approx.size);

		SmartIntArray oned_filter_num_for_each_dim(input_dims);
		MD_Filter_System &this_level_md_fs = ml_md_filter_system[cur_lvl];
		this_level_md_fs.oned_fs_for_each_dim.reserve(input_dims);
		for (int cur_dim = 0; cur_dim < input_dims; ++cur_dim)		// Every dim in this level
		{
			const OneD_Filter_System_Param &oned_filter_system_param = filter_system_param.md_fs_param_for_each_level[cur_lvl][cur_dim];

			int this_dim_filter_size = this_level_filter_size_for_each_dim[cur_dim];
			Mat x_pts;
			linspace(-M_PI, 0, M_PI, 0, this_dim_filter_size, x_pts);

			OneD_Filter_System &oned_filter_system = this_level_md_fs.oned_fs_for_each_dim[cur_dim];
			//TODO: Make sure low-pass be arranged at the end of oned_filter_system
			construct_1d_filter_system(x_pts, oned_filter_system_param, oned_filter_system);

			oned_filter_num_for_each_dim[cur_dim] = oned_filter_system.len;
		}

		// Find all combinations to do tensor product
		SmartArray<Mat> chosen_filter_for_each_dim(input_dims);
		SmartArray<SmartIntArray> supp_after_ds_for_each_dim(input_dims);
		for (int cur_dim = 0; cur_dim < input_dims; ++cur_dim)
		{
			chosen_filter_for_each_dim[cur_dim] = this_level_md_fs.oned_fs_for_each_dim[cur_dim][0].coefs;
			supp_after_ds_for_each_dim[cur_dim] = this_level_md_fs.oned_fs_for_each_dim[cur_dim][0].support_after_ds;
		}

		Mat lowpass_filter(input_dims, this_level_filter_size_for_each_dim, CV_64FC2, Scalar(0,0));
		SmartIntArray start_pos1(input_dims);
		{
			int src_dims;
			SmartIntArray src_start_pos;
			SmartIntArray src_cur_pos;
			SmartIntArray src_step;
			SmartIntArray src_end_pos;


			//User-Defined initialization
			src_dims = input_dims;
			src_start_pos = start_pos1;
			src_cur_pos = start_pos1.clone();
			src_step = step1;
			src_end_pos = oned_filter_num_for_each_dim;
			//--

			int cur_dim = src_dims - 1;
			while(true)
			{
				while (cur_dim >= 0 && src_cur_pos[cur_dim] >= src_end_pos[cur_dim])
				{
					src_cur_pos[cur_dim] = src_start_pos[cur_dim];
					--cur_dim;
					if (cur_dim >= 0)
					{
						src_cur_pos[cur_dim] += src_step[cur_dim];

						continue;
					}
				}

				if (cur_dim < 0)
				{
					break;
				}

				//User-Defined actions
				//A combination is found.
				bool is_lowpass = true;
				//Start at 0, since we need to check lowpass for all filters
				for (int i = 0; i < src_dims; ++i)
				{
					const OneD_Filter &fb = this_level_md_fs.oned_fs_for_each_dim[i][src_cur_pos[i]];
					chosen_filter_for_each_dim[i] = fb.coefs;
					supp_after_ds_for_each_dim[i]= fb.support_after_ds;
					is_lowpass = is_lowpass && fb.isLowPass;
				}

				Mat md_filter;
				tensor_product(chosen_filter_for_each_dim, md_filter);

				if (is_lowpass)
				{
//					pow(md_filter, 2, md_filter);
					pw_pow(md_filter, 2, md_filter);
					lowpass_filter += md_filter;
				}
				else
				{
//					Mat filtered = last_approx.mul(md_filter);
//					Mat filtered;
//					pw_mul(last_approx, md_filter, filtered);
//					Mat mat_after_ds;
//					mat_select(filtered, supp_after_ds_for_each_dim, mat_after_ds);
//					this_level_md_fs.md_filters_coefs.push_back(md_filter);
//
//					icenter_shift(mat_after_ds, mat_after_ds);
//
//					Mat td;
//					normalized_ifft(mat_after_ds, td);
//					coefs_set[cur_lvl].push_back(td);

					//--  Optimization --
					Mat last_approx_subarea, md_filter_subarea;
					mat_select(last_approx, supp_after_ds_for_each_dim, last_approx_subarea);
					mat_select(md_filter, supp_after_ds_for_each_dim, md_filter_subarea);

					Mat filtered;
					pw_mul(last_approx_subarea, md_filter_subarea, filtered);
					icenter_shift(filtered, filtered);
					normalized_ifft(filtered, filtered);
					coefs_set[cur_lvl].push_back(filtered);
					//--
				}
				//--

				cur_dim = src_dims - 1;
				src_cur_pos[cur_dim] += src_step[cur_dim];
			}

		}

//		sqrt(lowpass_filter, lowpass_filter);
		pw_sqrt(lowpass_filter, lowpass_filter);
		this_level_md_fs.md_filters_coefs.push_back(lowpass_filter);
		if (true)
		{
//			Mat filtered;
//			pw_mul(last_approx, lowpass_filter, filtered);
//			SmartArray<SmartIntArray> range_for_each_dim(lowpass_filter.dims);
//			for (int i = 0; i < lowpass_filter.dims; ++i)
//			{
//				range_for_each_dim[i].reserve(3);
//				range_for_each_dim[i][0] = lowpass_filter.size[i] / 4;
//				range_for_each_dim[i][1] = -1;
//				range_for_each_dim[i][2] = lowpass_filter.size[i] / 4 * 3 - 1;
//			}
//
//			mat_select(filtered, range_for_each_dim, filtered);
//
//			if (cur_lvl == levels - 1)
//			{
//				icenter_shift(filtered, filtered);
//
//				Mat td;
//				normalized_ifft(filtered, td);
//				coefs_set[cur_lvl].push_back(td);
//			}
//			else
//			{
//				coefs_set[cur_lvl].push_back(filtered);
//			}

			SmartArray<SmartIntArray> range_for_each_dim(lowpass_filter.dims);
			for (int i = 0; i < lowpass_filter.dims; ++i)
			{
				range_for_each_dim[i].reserve(3);
				range_for_each_dim[i][0] = lowpass_filter.size[i] / 4;
				range_for_each_dim[i][1] = -1;
				range_for_each_dim[i][2] = lowpass_filter.size[i] / 4 * 3 - 1;
			}

			Mat last_approx_subarea, filter_subarea, filtered;
			mat_select(last_approx, range_for_each_dim, last_approx_subarea);
			mat_select(lowpass_filter, range_for_each_dim, filter_subarea);
			pw_mul(last_approx_subarea, filter_subarea, filtered);

			if (cur_lvl == levels - 1)
			{
				icenter_shift(filtered, filtered);
				normalized_ifft(filtered, filtered);
				coefs_set[cur_lvl].push_back(filtered);
			}
			else
			{
				coefs_set[cur_lvl].push_back(filtered);
			}
		}

	}

	return 0;
}

int reconstruct_by_ml_md_filter_bank(const MLevel_MDFilter_System_Param &filter_system_param, const ML_MChannel_Coefs_Set &ml_mc_coefs_set, Mat &rec)
{

	if (filter_system_param.md_fs_param_for_each_level.len < 1)
	{
		return -1;
	}
	int levels = filter_system_param.md_fs_param_for_each_level.len;
	int sig_dims = filter_system_param.md_fs_param_for_each_level[0].len;

	SmartIntArray step1(sig_dims);
	for (int i = 0; i < sig_dims; ++i)
	{
		step1[i] = 1;
	}

//	ML_MChannel_Coefs_Set ml_mc_coefs_set_cpy = ml_mc_coefs_set;

	// Every Level
	Mat upper_level_lowpass_approx;
	Mat this_level_lowpass_approx;
	for (int cur_lvl = levels - 1; cur_lvl >= 0; --cur_lvl)
	{
		const vector<Mat> &this_level_coefs_set = ml_mc_coefs_set[cur_lvl];
		if (cur_lvl == levels - 1)
		{
			this_level_lowpass_approx = this_level_coefs_set[this_level_coefs_set.size() - 1].clone();
			normalized_fft(this_level_lowpass_approx, this_level_lowpass_approx);
			center_shift(this_level_lowpass_approx, this_level_lowpass_approx);
		}
		else
		{
			this_level_lowpass_approx = upper_level_lowpass_approx;
		}

		SmartIntArray this_level_filter_size_for_each_dim(sig_dims);
		for (int i = 0; i < sig_dims; ++i)
		{
			this_level_filter_size_for_each_dim[i] = this_level_lowpass_approx.size[i] * filter_system_param.lowpass_approx_ds_folds[cur_lvl][i];
		}

		SmartIntArray this_level_filter_num_for_each_dim(sig_dims);
		MD_Filter_System this_level_md_fs;
		this_level_md_fs.oned_fs_for_each_dim.reserve(sig_dims);
		for (int cur_dim = 0; cur_dim < sig_dims; ++cur_dim)		// Every dim in this level
		{
			const OneD_Filter_System_Param &oned_filter_system_param = filter_system_param.md_fs_param_for_each_level[cur_lvl][cur_dim];

			int this_dim_filter_size = this_level_filter_size_for_each_dim[cur_dim];
			Mat x_pts;
			linspace(-M_PI, 0, M_PI, 0, this_dim_filter_size, x_pts);

			OneD_Filter_System &oned_filter_system = this_level_md_fs.oned_fs_for_each_dim[cur_dim];
			//TODO: Make sure low-pass be arranged at the end of oned_filter_system
			construct_1d_filter_system(x_pts, oned_filter_system_param, oned_filter_system);

			this_level_filter_num_for_each_dim[cur_dim] = oned_filter_system.len;
		}

		// Find all combinations to do tensor product
		SmartArray<Mat> chosen_filter_for_each_dim(sig_dims);
		SmartArray<SmartIntArray> supp_after_ds_for_each_dim(sig_dims);
		for (int cur_dim = 0; cur_dim < sig_dims; ++cur_dim)
		{
			chosen_filter_for_each_dim[cur_dim] = this_level_md_fs.oned_fs_for_each_dim[cur_dim][0].coefs;
			supp_after_ds_for_each_dim[cur_dim] = this_level_md_fs.oned_fs_for_each_dim[cur_dim][0].support_after_ds;
		}

		Mat this_level_highpass_sum = Mat(sig_dims, this_level_filter_size_for_each_dim, CV_64FC2, Scalar(0,0));
		int coef_index = 0;
		Mat lowpass_filter(sig_dims, this_level_filter_size_for_each_dim, CV_64FC2, Scalar(0,0));
		SmartIntArray start_pos1(sig_dims);
		{
			int src_dims;
			SmartIntArray src_start_pos;
			SmartIntArray src_cur_pos;
			SmartIntArray src_step;
			SmartIntArray src_end_pos;


			//User-Defined initialization
			src_dims = sig_dims;
			src_start_pos = start_pos1;
			src_cur_pos = start_pos1.clone();
			src_step = step1;
			src_end_pos = this_level_filter_num_for_each_dim;
			//--

			int cur_dim = src_dims - 1;
			while(true)
			{
				while (cur_dim >= 0 && src_cur_pos[cur_dim] >= src_end_pos[cur_dim])
				{
					src_cur_pos[cur_dim] = src_start_pos[cur_dim];
					--cur_dim;
					if (cur_dim >= 0)
					{
						src_cur_pos[cur_dim] += src_step[cur_dim];

						continue;
					}
				}

				if (cur_dim < 0)
				{
					break;
				}

				//User-Defined actions
				//A combination is found.
				bool is_lowpass = true;
				//Start at 0, since we need to check lowpass for all filters
				for (int i = 0; i < src_dims; ++i)
				{
					const OneD_Filter &fb = this_level_md_fs.oned_fs_for_each_dim[i][src_cur_pos[i]];
					chosen_filter_for_each_dim[i] = fb.coefs;
					supp_after_ds_for_each_dim[i]= fb.support_after_ds;
					is_lowpass = is_lowpass && fb.isLowPass;
				}

				Mat md_filter;
				tensor_product(chosen_filter_for_each_dim, md_filter);

				if (is_lowpass)
				{
					pw_pow(md_filter, 2, md_filter);
					lowpass_filter += md_filter;
				}
				else
				{
					Mat this_channel_coef = this_level_coefs_set[coef_index].clone();

					// To change phase of the coef
//					SmartArray<Mat> ones_for_each_dim(sig_dims);
//					for (int i = 0; i < sig_dims; ++i)
//					{
//						int this_dim_coef_size = this_channel_coef.size[i];
//						Mat &ones_seq = ones_for_each_dim[i];
//						ones_seq.create(2, (int[]){1, this_dim_coef_size}, CV_64FC2);
//						ones_seq.at<complex<double> >(0, 0) = complex<double>(1,0);
//						for (int j = 1; j < this_dim_coef_size; ++j)
//						{
//							ones_seq.at<complex<double> >(0, j) = ones_seq.at<complex<double> >(0, j - 1) * (-1.0);
//						}
//					}
//					Mat phase_change;
//					tensor_product(ones_for_each_dim, phase_change);
//
//					pw_mul(this_channel_coef, phase_change, this_channel_coef);

					normalized_fft(this_channel_coef, this_channel_coef);
					center_shift(this_channel_coef, this_channel_coef);

//					Mat expanded_coef(sig_dims, this_level_filter_size_for_each_dim, CV_64FC2, Scalar(0,0));
//					mat_subfill(expanded_coef, supp_after_ds_for_each_dim, this_channel_coef, expanded_coef);
//					pw_mul(expanded_coef, md_filter, expanded_coef);

					Mat filter_subarea, filtered, expanded_coef;
					expanded_coef = Mat(sig_dims, this_level_filter_size_for_each_dim, CV_64FC2, Scalar(0,0));
					mat_select(md_filter, supp_after_ds_for_each_dim, filter_subarea);
					pw_mul(this_channel_coef, filter_subarea, filtered);
					mat_subfill(expanded_coef, supp_after_ds_for_each_dim, filtered, expanded_coef);

					this_level_highpass_sum += expanded_coef;
					++coef_index;
				}
				//--

				cur_dim = src_dims - 1;
				src_cur_pos[cur_dim] += src_step[cur_dim];
			}

		}


		pw_sqrt(lowpass_filter, lowpass_filter);
		if (true)
		{

			SmartArray<SmartIntArray> range_for_each_dim(lowpass_filter.dims);
			for (int i = 0; i < lowpass_filter.dims; ++i)
			{
				range_for_each_dim[i].reserve(3);
				range_for_each_dim[i][0] = lowpass_filter.size[i] / 4;
				range_for_each_dim[i][1] = -1;
				range_for_each_dim[i][2] = lowpass_filter.size[i] / 4 * 3 - 1;
			}

//			Mat expanded_coef(sig_dims, this_level_filter_size_for_each_dim, CV_64FC2, Scalar(0,0));
//
//			mat_subfill(expanded_coef, range_for_each_dim, this_level_lowpass_approx, expanded_coef);
//
//			pw_mul(expanded_coef, lowpass_filter, expanded_coef);
//
//			upper_level_lowpass_approx = this_level_highpass_sum + expanded_coef;

			Mat filter_subarea, filtered, expanded_coef;
			mat_select(lowpass_filter, range_for_each_dim, filter_subarea);
			pw_mul(this_level_lowpass_approx, filter_subarea, filtered);

			expanded_coef = Mat(sig_dims, this_level_filter_size_for_each_dim, CV_64FC2, Scalar(0,0));
			mat_subfill(expanded_coef, range_for_each_dim, filtered, expanded_coef);
			upper_level_lowpass_approx = this_level_highpass_sum + expanded_coef;
		}

	}

	icenter_shift(upper_level_lowpass_approx, upper_level_lowpass_approx);
	normalized_ifft(upper_level_lowpass_approx, upper_level_lowpass_approx);

	rec = upper_level_lowpass_approx;

	return 0;
}



/*
 * Do tensor product given a series of vectors.
 *   TP(v0,v1,...,vn)(i0,i1,...,in) = v0[i0]*v1[i1]*...*vn[in].
 *
 * components_for_each_dim: Input param. The given n vectors
 * product: output. The resulting n-d matrix.
 */
int tensor_product(const SmartArray<Mat> &components_for_each_dim, Mat &product)
{
	//TODO: make sure all input components are continuous mat, and row vectors.
	int dims = components_for_each_dim.len;
	SmartIntArray dim_size(dims);
	Mat sub_mat = components_for_each_dim[dims - 1];    //row vector
	dim_size[dims - 1] = sub_mat.total();
	for (int cur_dim = dims - 2; cur_dim >= 0; --cur_dim)
	{
		const Mat &cur_dim_mat = components_for_each_dim[cur_dim];   // column vector;
		dim_size[cur_dim] = cur_dim_mat.total();
		sub_mat = cur_dim_mat.t() * sub_mat;	//This is a 2D matrix. Transpose is O(1) operation.
		sub_mat = sub_mat.reshape(0, 1);    //Convert to row vector. O(1) operation
//		print_mat_details_g<Vec2d>(sub_mat, 0, "Test-Data/tensor-product.txt");
	}

//	//NOTE: Wrong Doing!!! Memory Corruption would happen, since sub_mat would dealloc memory when going out of scope.
//	Mat md_mat(dims, (const int *)dim_size, sub_mat.type(), sub_mat.data);
//	sub_mat.create(dims, dim_size, sub_mat.type());
	Mat reshaped(dims, (const int *)dim_size, sub_mat.type());
	MatConstIterator_<Vec2d> it0 = sub_mat.begin<Vec2d>(), end0 = sub_mat.end<Vec2d>();
	MatIterator_<Vec2d> it1 = reshaped.begin<Vec2d>();
	for (; it0 != end0; ++it0, ++it1)
	{
		*it1 = *it0;
	}
	product = reshaped;
//	print_mat_details_g<Vec2d>(reshaped, 0, "Test-Data/tensor-product.txt");

	return 0;
}

int mat_select(const Mat &origin_mat, const SmartArray<SmartIntArray> &index_set_for_each_dim, Mat &sub_mat)
{
	int dims = index_set_for_each_dim.len;
	SmartIntArray start_pos1(dims);
	SmartIntArray cur_pos1(dims);
	SmartIntArray step1(dims);
	SmartIntArray end_pos1(dims);
	SmartIntArray sel_idx(dims);
	SmartArray<SmartIntArray> index_set_for_each_dim_cpy = index_set_for_each_dim.clone();
	for (int i = 0; i < dims; ++i)
	{
		step1[i] = 1;
		if (index_set_for_each_dim[i].len == 3 && index_set_for_each_dim[i][1] < 0)
		{
			const SmartIntArray &this_idx_set = index_set_for_each_dim[i];
			int start = this_idx_set[0];
			int step = -this_idx_set[1];
			int stop = this_idx_set[2];
			SmartIntArray expanded((stop - start) / step + 1);
			for (int j = start; j <= stop; j += step)
			{
				expanded[j - start] = j;
			}
			index_set_for_each_dim_cpy[i] = expanded;
		}
		end_pos1[i] = index_set_for_each_dim_cpy[i].len;
		sel_idx[i] = index_set_for_each_dim_cpy[i][0];
	}

	Mat selected(dims, (const int*)end_pos1, origin_mat.type());
	{
		int src_dims;
		SmartIntArray src_start_pos;
		SmartIntArray src_cur_pos;
		SmartIntArray src_step;
		SmartIntArray src_end_pos;


		//User-Defined initialization
		src_dims = dims;
		src_start_pos = start_pos1;
		src_cur_pos = cur_pos1;
		src_step = step1;
		src_end_pos = end_pos1;
		//--

		int cur_dim = src_dims - 1;
		while(true)
		{
			while (cur_dim >= 0 && src_cur_pos[cur_dim] >= src_end_pos[cur_dim])
			{
				src_cur_pos[cur_dim] = src_start_pos[cur_dim];
				--cur_dim;
				if (cur_dim >= 0)
				{
					src_cur_pos[cur_dim] += src_step[cur_dim];

					continue;
				}
			}

			if (cur_dim < 0)
			{
				break;
			}

			//User-Defined actions
			for (; cur_dim < src_dims; ++cur_dim)
			{
				sel_idx[cur_dim] = index_set_for_each_dim_cpy[cur_dim][src_cur_pos[cur_dim]];
			}
			selected.at<complex<double> >((const int*)src_cur_pos) = origin_mat.at<complex<double> >((const int*)sel_idx);
			//--

			cur_dim = src_dims - 1;
			src_cur_pos[cur_dim] += src_step[cur_dim];
		}

	}

	sub_mat = selected;
	return 0;
}

int mat_subfill(const Mat &origin_mat, const SmartArray<SmartIntArray> &index_set_for_each_dim, const Mat &submat, Mat &filled_mat)
{
	int dims = index_set_for_each_dim.len;
	SmartIntArray start_pos1(dims);
	SmartIntArray cur_pos1(dims);
	SmartIntArray step1(dims);
	SmartIntArray end_pos1(dims);
	SmartIntArray sel_idx(dims);
	SmartArray<SmartIntArray> index_set_for_each_dim_cpy = index_set_for_each_dim.clone();
	for (int i = 0; i < dims; ++i)
	{
		step1[i] = 1;
		if (index_set_for_each_dim[i].len == 3 && index_set_for_each_dim[i][1] < 0)
		{
			const SmartIntArray &this_idx_set = index_set_for_each_dim[i];
			int start = this_idx_set[0];
			int step = -this_idx_set[1];
			int stop = this_idx_set[2];
			SmartIntArray expanded((stop - start) / step + 1);
			for (int j = start; j <= stop; j += step)
			{
				expanded[j - start] = j;
			}
			index_set_for_each_dim_cpy[i] = expanded;
		}
		end_pos1[i] = index_set_for_each_dim_cpy[i].len;
		sel_idx[i] = index_set_for_each_dim_cpy[i][0];
	}

	Mat origin_cpy = origin_mat.clone();
	{
		int src_dims;
		SmartIntArray src_start_pos;
		SmartIntArray src_cur_pos;
		SmartIntArray src_step;
		SmartIntArray src_end_pos;


		//User-Defined initialization
		src_dims = dims;
		src_start_pos = start_pos1;
		src_cur_pos = cur_pos1;
		src_step = step1;
		src_end_pos = end_pos1;
		//--

		int cur_dim = src_dims - 1;
		while(true)
		{
			while (cur_dim >= 0 && src_cur_pos[cur_dim] >= src_end_pos[cur_dim])
			{
				src_cur_pos[cur_dim] = src_start_pos[cur_dim];
				--cur_dim;
				if (cur_dim >= 0)
				{
					src_cur_pos[cur_dim] += src_step[cur_dim];

					continue;
				}
			}

			if (cur_dim < 0)
			{
				break;
			}

			//User-Defined actions
			for (; cur_dim < src_dims; ++cur_dim)
			{
				sel_idx[cur_dim] = index_set_for_each_dim_cpy[cur_dim][src_cur_pos[cur_dim]];
			}
			origin_cpy.at<complex<double> >((const int *)sel_idx) = submat.at<complex<double> >((const int *)src_cur_pos);
			//--

			cur_dim = src_dims - 1;
			src_cur_pos[cur_dim] += src_step[cur_dim];
		}

	}

	filled_mat = origin_cpy;
	return 0;
}


int psnr(const Mat &left, const Mat &right, double &psnr, double &msr)
{
	if (left.dims != right.dims)
	{
		return -1;
	}

	Mat dif = left - right;
	double msr_stat = 0.0;
	MatConstIterator_<Vec2d> it0 = dif.begin<Vec2d>(), end0 = dif.end<Vec2d>(),
			                 it1 = right.begin<Vec2d>();

	for (; it0 != end0; ++it0, ++it1)
	{
//		double dif = sqrt((*it0)[0] * (*it0)[0] + (*it0)[1] * (*it0)[1])
//				    - sqrt((*it1)[0] * (*it1)[0] + (*it1)[1] * (*it1)[1]);

		double sqr = (*it0)[0] * (*it0)[0] + (*it0)[1] * (*it0)[1];
		msr_stat += sqr;
	}

	msr_stat = sqrt(msr_stat / left.total());

//	if (msr_stat < 0.000001)
//	{
//		return -1;
//	}

	double psnr_stat = 0.0;

	psnr_stat = log(255.0 / msr_stat);
	psnr_stat = 20 * psnr_stat / log(10);

	psnr = psnr_stat;
	msr = msr_stat;

	return 0;
}

int load_as_tensor(const string &filename, Mat &output, Media_Format *media_file_fmt)
{
	size_t pos = filename.find_last_of('.');
	if (pos == std::string::npos)
	{
		return -1;
	}

	string suffix = filename.substr(pos);
	Mat mat;
	// images
	if (suffix == string(".jpg") || suffix == string(".bmp") || suffix == string(".jpeg") || suffix == string(".png"))
	{
		Mat img = imread(filename);
		if (img.empty())
		{
			return -2;
		}

		switch (img.type())
		{
		case CV_8UC3:
			cvtColor(img, img, CV_BGR2GRAY);
			break;
		case CV_8UC1:
			break;
		default:
			return -3;
		}

		mat.create(img.size(), CV_64FC2);
		for (int r = 0; r < img.rows; ++r)
		{
			for (int c = 0; c < img.cols; ++c)
			{
				mat.at<Vec2d>(r, c)[0] = img.at<uchar>(r, c);
				mat.at<Vec2d>(r, c)[1] = 0;
			}
		}

		output = mat;

		return 0;
	}

	// videos
	if (suffix == string(".avi") || suffix == string(".wav"))
	{
		VideoCapture cap(filename);
		if (!cap.isOpened())
		{
			return -2;
		}


		int sz[3];
		sz[0] = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_COUNT));
		sz[1] = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_HEIGHT));
		sz[2] = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_WIDTH));


		if (media_file_fmt != NULL)
		{
			media_file_fmt->video_prop.frame_count = sz[0];
			media_file_fmt->video_prop.frame_height = sz[1];
			media_file_fmt->video_prop.frame_width = sz[2];
			media_file_fmt->video_prop.fps = static_cast<int>(cap.get(CV_CAP_PROP_FPS));
			media_file_fmt->video_prop.fourcc = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));
		}


		mat.create(3, sz, CV_64FC2);
		for (int f = 0; f < sz[0]; f++)
		{
			Mat frame;
			bool bret = cap.read(frame);
			if (!bret)
			{
				break;
			}

			// Handle only 8UC1.
			switch(frame.type())
			{
			case CV_8UC3:
				cvtColor(frame, frame, CV_BGR2GRAY);
				break;
			case CV_8UC1:
				break;
			default:

				return -3;
			}

//			imshow("Current Frame", frame);
//			waitKey();


			Range ranges[3];
			ranges[0] = Range(f, f + 1);
			ranges[1] = Range::all();
			ranges[2] = Range::all();

			Mat plane = mat.row(f);
			for (int r = 0; r < sz[1]; ++r)
			{
				for (int c = 0; c < sz[2]; ++c)
				{
					plane.at<Vec2d>(0, r, c)[0] = frame.at<uchar>(r, c);
					plane.at<Vec2d>(0, r, c)[1] = 0;
				}
			}
		}

		output = mat;

		return 0;
	}

	return -4;
}

int save_as_media(const string &filename, const Mat &mat, const Media_Format *media_file_fmt)
{
    if (mat.dims == 2 && mat.size[0] != 1)
    {
         Mat img_r(mat.size(), CV_64FC1);
         Mat img_i(mat.size(), CV_64FC1);
         Mat img_abs(mat.size(), CV_64FC1);


         for (int i = 0; i < mat.rows; i++)
         {
         	for (int j = 0; j < mat.cols; j++)
         	{
         		double d = mat.at<Vec2d>(i, j)[0] * mat.at<Vec2d>(i, j)[0]
         		         + mat.at<Vec2d>(i, j)[1] * mat.at<Vec2d>(i, j)[1];

         		img_r.at<double>(i, j) = mat.at<Vec2d>(i,j)[0];
         		img_i.at<double>(i, j) = mat.at<Vec2d>(i,j)[1];
         		img_abs.at<double>(i, j) = sqrt(d);
         	}

         }

         double d0, d1;
         minMaxLoc(img_r, &d0, &d1);
         img_r.convertTo(img_r, 255.0 / (d0 - d1), -255.0 / (d0 - d1));
         img_r.convertTo(img_r, CV_8UC1, 1, 0);
         imwrite(filename+ "-r.jpg", img_r);

         img_i.convertTo(img_i, 255.0 / (d0 - d1), -255.0 / (d0 - d1));
         img_i.convertTo(img_i, CV_8UC1, 1, 0);
         imwrite(filename + "-i.jpg", img_i);


         img_abs.convertTo(img_abs, 255.0 / (d0 - d1), -255.0 / (d0 - d1));
         img_abs.convertTo(img_abs, CV_8UC1, 1, 0);
         imwrite(filename + "-abs.jpg", img_abs);

         return 0;
    }

    if (mat.dims == 3)
    {
    	if (media_file_fmt == NULL)
    	{
    		return -3;
    	}

    	VideoWriter writer(filename, CV_FOURCC('P','I','M','1')
    			         , media_file_fmt->video_prop.fps
    			         , Size(mat.size[2], mat.size[1]), false);

    	if (!writer.isOpened())
    	{
    		return -2;
    	}

    	for (int f = 0; f < mat.size[0]; f++)
    	{
    		Mat plane = mat.row(f);
            Mat frame(Size(plane.size[2], plane.size[1]), CV_64FC1);

            for (int i = 0; i < mat.size[1]; i++)
            {
            	for (int j = 0; j < mat.size[2]; j++)
            	{
            		double d = plane.at<Vec2d>(0, i, j)[0] * plane.at<Vec2d>(0, i, j)[0]
            		         + plane.at<Vec2d>(0, i, j)[1] * plane.at<Vec2d>(0, i, j)[1];

            		frame.at<double>(i, j) = sqrt(d);
            	}
            }

            frame.convertTo(frame, CV_8UC1, 1, 0);
            writer << frame;
    	}
    }
	return 0;
}
