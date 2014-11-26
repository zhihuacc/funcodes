#include "../include/wavelets_tools.h"

#include <fftw3.h>
#include <iostream>
#include <iomanip>



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

int mat_border_extension(const Mat &origin, int n, int *border, const string &opt, Mat &extended)
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

int mat_border_cut(const Mat &extended, int n, int *border, Mat &origin)
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

int fchi(const Mat &x_pt, const string &opt, Mat &y_val)
{

	return 0;
}

void print_mat_details(const Mat &mat, const string &filename)
{
	int max_term = 8;

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
			cout << endl << endl << "[";
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


		cout << "(" << mat.at<Vec2d>(pos)[0] << ","
				    << mat.at<Vec2d>(pos)[1] << ")";
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
