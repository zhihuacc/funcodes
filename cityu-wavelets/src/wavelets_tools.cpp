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
				f = sincos_bump((1 + r) / 2, param.m);
			}
			else if (w > (param.cR - param.epR) && w < (param.cR + param.epR))
			{
				double r = (w - param.cR) / param.epR;
				f = sincos_bump((1 + r) / 2, param.m);
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
				f = sqrt_bump((1 + r)  /2, param.m);
			}
			else if (w > (param.cR - param.epR) && w < (param.cR + param.epR))
			{
				double r = (w - param.cR) / param.epR;
				f = sqrt_bump((1 + r) / 2, param.m);
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

int construct_1d_filter_banks(const Mat &x_pts, int cp_num, const double *ctrl_pts, int degree, const string &opt, Filter_Set &output)
{
	Filter_Set filters;
	if (cp_num < 2 || ctrl_pts == NULL)
	{
		return -1;
	}

	filters.reserve(cp_num);
	filters.resize(cp_num);

	Mat shift_right_x = x_pts.clone() + Scalar(-2 * M_PI, 0);
	Mat shift_left_x = x_pts.clone() + Scalar(2 * M_PI, 0);

	for (int i = 0; i < cp_num; ++i)
	{
		Filter_Info &this_filter = filters[i];
		Chi_Ctrl_Param &param = filters[i].param;
		int idx = 2 * i;
		if (i != cp_num - 1)
		{
			param.cL = ctrl_pts[idx];
			param.epL = ctrl_pts[idx + 1];
			param.cR = ctrl_pts[idx + 2];
			param.epR = ctrl_pts[idx + 3];
		}
		else
		{
			param.cL = ctrl_pts[idx];
			param.epL = ctrl_pts[idx + 1];
			param.cR = ctrl_pts[0];
			param.epR = ctrl_pts[1];
		}
		param.m = degree;

		this_filter.needShift = false;
		if (param.cR < param.cL)
		{
			this_filter.needShift = true;
			param.cR += 2 * M_PI;
		}

		this_filter.isLowPass = false;
		if (param.cL < 0 && param.cR > 0)
		{
			this_filter.isLowPass = true;
		}

		Mat shift_right_filter;
		Mat shift_left_filter;

		fchi(shift_right_x, param, opt, shift_right_filter);
		fchi(x_pts, param, opt, this_filter.filter);
		fchi(shift_left_x, param, opt, shift_left_filter);

		this_filter.filter = this_filter.filter + shift_right_filter + shift_left_filter;


	}

	output = filters;

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


int downsample_in_fd_by2(const Mat &filter, SmartIntArray &folds, Mat &folded_filter, vector<SmartIntArray> &support)
{

	int dims = filter.dims;
	SmartIntArray start_pos1(dims);
	SmartIntArray cur_pos1(dims);
	SmartIntArray step1(dims);
	SmartIntArray origin_range(dims, filter.size);
	SmartIntArray start_pos2(dims);
	SmartIntArray cur_pos2(dims);
	SmartIntArray folded_range(dims);

	for (int i = 0; i < dims; ++i)
	{
		step1[i] = 1;
		folded_range[i] = (origin_range[i] < 2) ? 1 : (origin_range[i] / folds[i]);
	}
	vector<SmartIntArray> supp_set;
	Mat folded_mat(folded_range.dims, (const int*)folded_range, CV_64FC2, Scalar(0,0));
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
			double sum_r = 0.0, sum_i = 0.0;
			supp_set.push_back(src_cur_pos.clone());
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

					double d_r = 0.0, d_i = 0.0;
					d_r = filter.at<Vec2d>(src_cur_pos)[0];
					d_i = filter.at<Vec2d>(src_cur_pos)[1];
					if (d_r > 0.0 || d_i > 0.0)
					{
						supp_set[supp_set.size() - 1] = src_cur_pos.clone();
					}
					sum_r += d_r;
					sum_i += d_i;

					cur_dim = dims - 1;
					src_cur_pos[cur_dim] += src_step[cur_dim];
				}

			}
			folded_mat.at<Vec2d>(src_cur_pos)[0] = sum_r;
			folded_mat.at<Vec2d>(src_cur_pos)[1] = sum_i;
			//--

			cur_dim = src_dims - 1;
			src_cur_pos[cur_dim] += src_step[cur_dim];
		}

	}

	folded_filter = folded_mat;
	support = supp_set;

	return 0;
}


int psnr(const Mat &left, const Mat &right, double &psnr, double &msr)
{
	if (left.dims != right.dims)
	{
		return -1;
	}

	double msr_stat = 0.0;
	MatConstIterator_<Vec2d> it0 = left.begin<Vec2d>(), end0 = left.end<Vec2d>(),
			                 it1 = right.begin<Vec2d>();

	for (; it0 != end0; ++it0, ++it1)
	{
		double dif = sqrt((*it0)[0] * (*it0)[0] + (*it0)[1] * (*it0)[1])
				    - sqrt((*it1)[0] * (*it1)[0] + (*it1)[1] * (*it1)[1]);


		msr_stat += dif * dif;
	}

	msr_stat = sqrt(msr_stat / left.total());

	if (msr_stat < 0.000001)
	{
		return -1;
	}

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
	if (suffix == string(".jpg") || suffix == string(".bmp") || suffix == string(".jpeg"))
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
         Mat img(mat.size(), CV_64FC1);

         for (int i = 0; i < mat.rows; i++)
         {
         	for (int j = 0; j < mat.cols; j++)
         	{
         		double d = mat.at<Vec2d>(i, j)[0] * mat.at<Vec2d>(i, j)[0]
         		         + mat.at<Vec2d>(i, j)[1] * mat.at<Vec2d>(i, j)[1];

         		img.at<double>(i, j) = sqrt(d);
         	}
         }

         img.convertTo(img, CV_8UC1, 1, 0);

         imwrite(filename, img);

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
