#ifndef MATRIX_TOOLBOX_H
#define MATRIX_TOOLBOX_H

//#include <complex.h>
#include <fftw3.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <fstream>
#include <iomanip>

#include "hammers_and_nails.h"

using namespace cv;

template<typename _Tp>
inline bool isGoodMat(const Mat_<Vec<_Tp, 2> > &domain)
{
	if (domain.empty() || !domain.isContinuous()
		|| (domain.depth() != CV_64F && domain.depth() != CV_32F) || domain.channels() != 2)
	{
		cout << "Failed in isGoodMat " << endl;
//		CV_ERROR(CV_StsInternal, "Test of isGoodMat failed.");
		return false;
	}

	return true;
}

template<typename _Tp>
inline bool sameSize(const Mat_<Vec<_Tp, 2> > &left, const Mat_<Vec<_Tp, 2> > &right)
{
	SmartIntArray sl(left.dims, left.size), sr(right.dims, right.size);
	bool ret = sl == sr;
	if (ret == false)
	{
		cout << "Failed in sameSize " << endl;
	}

	return ret;
}


/*
 * Do DFT transform. Make sure domain isContinuous().
 *
 * time_domain: input time-domain signal.
 * feq_domain: output frequency-domain spectrum. time_domain and freq_domain can be the same Mat.
 */
template<typename _Tp>
int normalized_fft(const Mat_<Vec<_Tp, 2> > &time_domain, Mat_<Vec<_Tp, 2> > &freq_domain)
{
	//TODO: need to check if the two matrix match in size.
	if (!isGoodMat(time_domain))
	{
		return -1;
	}

	fftw_complex *before;
	fftw_complex *after;
	fftw_plan plan;

	Mat_<Vec<_Tp, 2> > tmp(time_domain.dims, time_domain.size);
	before = reinterpret_cast<fftw_complex *>(time_domain.data);
	after = reinterpret_cast<fftw_complex *>(tmp.data);
	// Here we can only use 'FFTW_ESTIMATE', because 'FFTW_MEASURE' would touch 'before'.
	plan = fftw_plan_dft(time_domain.dims, time_domain.size, before, after, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(plan);
	fftw_destroy_plan(plan);

	freq_domain = tmp  * (1.0 / sqrt(tmp.total()));

    return 0;
}
/*
 * Do DFT transform
 *
 * feq_domain: input frequency-domain spectrum.
 * time_domain: output time-domain signal.time_domain and freq_domain can be the same Mat.
 */
template<typename _Tp>
int normalized_ifft(const Mat_<Vec<_Tp, 2> > &freq_domain, Mat_<Vec<_Tp, 2> > &time_domain)
{
	//TODO: need to check if the two matrix match in size.
	if (!isGoodMat(freq_domain))
	{
		return -1;
	}

	fftw_complex *before;
	fftw_complex *after;
	fftw_plan plan;

	Mat_<Vec<_Tp, 2> > tmp(freq_domain.dims, freq_domain.size);

	before = reinterpret_cast<fftw_complex*>(freq_domain.data);
	after = reinterpret_cast<fftw_complex*>(tmp.data);

	// Here we can only use 'FFTW_ESTIMATE', because 'FFTW_MEASURE' would touch 'before'.
	plan = fftw_plan_dft(freq_domain.dims, freq_domain.size, before, after, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(plan);
    fftw_destroy_plan(plan);

    time_domain = tmp  * (1.0 / sqrt(tmp.total()));

    return 0;
}

/*
 * Half shift the matrix. This is NOT done in-place. 'shifted' can NOT be equal to 'data'.
 */
template<typename _Tp>
int center_shift(const Mat_<Vec<_Tp, 2> > &data, Mat_<Vec<_Tp, 2> > &shifted)
{
	if (!isGoodMat(data))
	{
		return -1;
	}

	int ndims = data.dims;
	SmartIntArray start_pos1(ndims);
	SmartIntArray cur_pos1(ndims);
	SmartIntArray step1(ndims, 1);
	SmartIntArray range1(ndims, data.size);
	SmartIntArray offset(ndims);
	SmartIntArray pos2(ndims);

	for (int i = 0; i < ndims; ++i)
	{
//		step1[i] = 1;
		offset[i] = range1[i] >> 1;
		pos2[i] = offset[i];
	}

	Mat_<Vec<_Tp, 2> > shifted_mat(data.dims, data.size);
	{
		int src_dims;
		SmartIntArray src_start_pos;
		SmartIntArray src_cur_pos;
		SmartIntArray src_step;
		SmartIntArray src_end_pos;


		//User-Defined initialization
		src_dims = ndims;
		src_start_pos = start_pos1;
		src_cur_pos = cur_pos1;
		src_step = step1;
		src_end_pos = range1;
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
			for (; cur_dim < src_dims; cur_dim++)
			{
				pos2[cur_dim] = cur_pos1[cur_dim] + offset[cur_dim];
				if (pos2[cur_dim] >= range1[cur_dim])
				{
					pos2[cur_dim] -= range1[cur_dim];
				}
				else if (pos2[cur_dim] < 0)
				{
					pos2[cur_dim] += range1[cur_dim];
				}
			}
			shifted_mat(pos2) = data(cur_pos1);
			//--

			cur_dim = src_dims - 1;
			src_cur_pos[cur_dim] += src_step[cur_dim];
		}

	}

	shifted = shifted_mat;
	return 0;
}

/*
 * Half shift back the matrix. This is NOT done in-place. 'shifted' can NOT be equal to 'data'.
 */
template<typename _Tp>
int icenter_shift(const Mat_<Vec<_Tp, 2> > &data, Mat_<Vec<_Tp, 2> > &shifted)
{
	if (!isGoodMat(data))
	{
		return -1;
	}

	int ndims = data.dims;
	SmartIntArray start_pos1(ndims);
	SmartIntArray cur_pos1(ndims);
	SmartIntArray step1(ndims, 1);
	SmartIntArray range1(ndims, data.size);
	SmartIntArray offset(ndims);
	SmartIntArray pos2(ndims);

	for (int i = 0; i < ndims; ++i)
	{
		offset[i] = (range1[i] & 1) ? (range1[i] - (range1[i] >> 1)) : range1[i] >> 1;
		pos2[i] = offset[i]; // pos2 = start_pos1 + offset.
	}


	Mat_<Vec<_Tp, 2> > shifted_mat(data.dims, data.size);
	{
		int src_dims;
		SmartIntArray src_start_pos;
		SmartIntArray src_cur_pos;
		SmartIntArray src_step;
		SmartIntArray src_end_pos;


		//User-Defined initialization
		src_dims = ndims;
		src_start_pos = start_pos1;
		src_cur_pos = cur_pos1;
		src_step = step1;
		src_end_pos = range1;
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
			for (; cur_dim < src_dims; cur_dim++)
			{
//				pos2[cur_dim] = (src_cur_pos[cur_dim] + offset[cur_dim]) % range1[cur_dim];
				pos2[cur_dim] = cur_pos1[cur_dim] + offset[cur_dim];
				if (pos2[cur_dim] >= range1[cur_dim])
				{
					pos2[cur_dim] -= range1[cur_dim];
				}
				else if (pos2[cur_dim] < 0)
				{
					pos2[cur_dim] += range1[cur_dim];
				}
			}
			shifted_mat(pos2) = data(cur_pos1);
			//--

			cur_dim = src_dims - 1;
			src_cur_pos[cur_dim] += src_step[cur_dim];
		}
	}

	shifted = shifted_mat;
	return 0;
}

/*
 * Do complex-value multiplication point-wisely. Make sure that product has been allocated memory properly
 * before calling. NO memory allocation happens inside.
 */
template<typename _Tp>
int pw_mul(const Mat_<Vec<_Tp, 2> > &left, const Mat_<Vec<_Tp, 2> > &right, Mat_<Vec<_Tp, 2> > &product)
{
    // Need more check if that left and right match
    if (!isGoodMat(left) || !isGoodMat(right) || !sameSize(left, right))
    {
    	return -1;
    }

    Mat_<Vec<_Tp, 2> > product_mat(left.dims, left.size);
    int N = left.total();
    complex<_Tp> *pleft = reinterpret_cast<complex<_Tp> *>(left.data), *pright = reinterpret_cast<complex<_Tp> *>(right.data),
    		     *pprod  = reinterpret_cast<complex<_Tp> *>(product_mat.data);
    for (int i = 0; i < N; ++i, ++pleft, ++pright, ++pprod)
    {
    	*pprod = (*pleft) * (*pright);
    }

    product = product_mat;
    return 0;
}

template<typename _Tp>
int pw_mul(const Mat_<Vec<_Tp, 2> > &left, const complex<_Tp> alpha, Mat_<Vec<_Tp, 2> > &product)
{
    // Need more check if that left and right match
    if (!isGoodMat(left))
    {
    	return -1;
    }

    Mat_<Vec<_Tp, 2> > product_mat(left.dims, left.size);
    int N = left.total();
    complex<_Tp> *pleft = reinterpret_cast<complex<_Tp> *>(left.data),
    		     *pprod  = reinterpret_cast<complex<_Tp> *>(product_mat.data);
    for (int i = 0; i < N; ++i, ++pleft, ++pprod)
    {
    	*pprod = (*pleft) * alpha;
    }

    product = product_mat;

    return 0;
}

template<typename _Tp>
int pw_div(const Mat_<Vec<_Tp, 2> > &left, const Mat_<Vec<_Tp, 2> > &right, Mat_<Vec<_Tp, 2> > &res)
{
    // Need more check if that left and right match
    if (!isGoodMat(left) || !isGoodMat(right) || !sameSize(left, right))
    {
    	return -1;
    }

    Mat_<Vec<_Tp, 2> > res_mat(left.dims, left.size);

    int N = left.total();
    complex<_Tp> *pleft = reinterpret_cast<complex<_Tp> *>(left.data), *pright = reinterpret_cast<complex<_Tp> *>(right.data),
    		     *pres  = reinterpret_cast<complex<_Tp> *>(res_mat.data);
    for (int i = 0; i < N; ++i, ++pleft, ++pright, ++pres)
    {
    	*pres = (*pleft) / (*pright);
    }
    res = res_mat;

    return 0;
}

template<typename _Tp>
int pw_div(const Mat_<Vec<_Tp, 2> > &left, const complex<_Tp> alpha, Mat_<Vec<_Tp, 2> > &res)
{
    // Need more check if that left and right match
    if (!isGoodMat(left))
    {
    	return -1;
    }

    Mat_<Vec<_Tp, 2> > res_mat(left.dims, left.size);
    int N = left.total();
    complex<_Tp> *pleft = reinterpret_cast<complex<_Tp> *>(left.data),
    		     *pres  = reinterpret_cast<complex<_Tp> *>(res_mat.data);
    for (int i = 0; i < N; ++i, ++pleft, ++pres)
    {
    	*pres = (*pleft) / alpha;
    }

    res = res_mat;

    return 0;
}

template<typename _Tp>
int pw_div(const complex<_Tp> alpha, const Mat_<Vec<_Tp, 2> > &right, Mat_<Vec<_Tp, 2> > &res)
{
    // Need more check if that left and right match
    if (!isGoodMat(right))
    {
    	return -1;
    }

    Mat_<Vec<_Tp, 2> > res_mat(right.dims, right.size);
    int N = right.total();
    complex<_Tp> *pright = reinterpret_cast<complex<_Tp> *>(right.data),
    		     *pres  = reinterpret_cast<complex<_Tp> *>(res_mat.data);
    for (int i = 0; i < N; ++i, ++pright, ++pres)
    {
    	*pres = alpha / (*pright);
    }
    res = res_mat;

    return 0;
}

/*
 * Do complex-value pow point-wisely. NO memory allocation happens inside.
 */
template<typename _Tp>
int pw_pow(const Mat_<Vec<_Tp, 2> > &base, _Tp expo, Mat_<Vec<_Tp, 2> > &res)
{

	if (!isGoodMat(base))
	{
		return -1;
	}

	// No need to zero.
	Mat_<Vec<_Tp, 2> > res_mat(base.dims, base.size);
    int N = base.total();
    complex<_Tp> *pbase = reinterpret_cast<complex<_Tp> *>(base.data),
    		     *pres  = reinterpret_cast<complex<_Tp> *>(res_mat.data);
    //TODO 'fabs' is not good practice for template programming
    if (fabs(expo - 2) <= numeric_limits<_Tp>::epsilon())
    {
    	for (int i = 0; i < N; ++i, ++pbase, ++pres)
		{
			*pres = (*pbase) * (*pbase); //TODO I guess this is faster than pow(*pbase, 2).
		}
    }
    else
    {
		for (int i = 0; i < N; ++i, ++pbase, ++pres)
		{
			*pres = pow<_Tp>(*pbase, expo);
		}
    }
    res = res_mat;

	return 0;
}

template<typename _Tp>
int pw_abs(const Mat_<Vec<_Tp, 2> > &base, Mat_<Vec<_Tp, 2> > &res)
{
	Mat_<Vec<_Tp, 2> > res_mat(base.dims, base.size, Vec<_Tp, 2>(0,0));
    int N = base.total();
    complex<_Tp> *pbase = reinterpret_cast<complex<_Tp> *>(base.data),
    		     *pres  = reinterpret_cast<complex<_Tp> *>(res_mat.data);
    for (int i = 0; i < N; ++i, ++pbase, ++pres)
    {
    	*pres = abs<_Tp>(*pbase);
    }
    res = res_mat;

    return 0;
}

/*
 * Do complex-value sqrt point-wisely. NO memory allocation happens inside.
 */
template<typename _Tp>
int pw_sqrt(const Mat_<Vec<_Tp, 2> > &base, Mat_<Vec<_Tp, 2> > &res)
{
	if (!isGoodMat(base))
	{
		return -1;
	}

	Mat_<Vec<_Tp, 2> > res_mat(base.dims, base.size);
    int N = base.total();
    complex<_Tp> *pbase = reinterpret_cast<complex<_Tp> *>(base.data),
    		     *pres  = reinterpret_cast<complex<_Tp> *>(res_mat.data);
    for (int i = 0; i < N; ++i, ++pbase, ++pres)
    {
    	*pres = sqrt<_Tp>(*pbase);
    }
    res = res_mat;

	return 0;
}

template<typename _Tp>
int pw_less(const Mat_<Vec<_Tp, 2> > &left, const Mat_<Vec<_Tp, 2> > &right, Mat_<Vec<_Tp, 2> > &res)
{
    if (!isGoodMat(left) || !isGoodMat(right) || !sameSize(left, right))
    {
    	return -1;
    }

	Mat_<Vec<_Tp, 2> > res_mat(left.dims, left.size, Vec<_Tp, 2>(0,0));
    int N = left.total();
    complex<_Tp> *pleft = reinterpret_cast<complex<_Tp> *>(left.data), *pright = reinterpret_cast<complex<_Tp> *>(right.data),
    		     *pres  = reinterpret_cast<complex<_Tp> *>(res_mat.data);
    for (int i = 0; i < N; ++i, ++pleft, ++pright, ++pres)
    {
    	(*pres).real( (*pleft).real() < (*pright).real() );
    }
    res = res_mat;

	return 0;
}

template<typename _Tp>
int pw_less(const Mat_<Vec<_Tp, 2> > &left, const complex<_Tp> &alpha, Mat_<Vec<_Tp, 2> > &res)
{
	Mat_<Vec<_Tp, 2> > res_mat(left.dims, left.size, Vec<_Tp, 2>(0,0));
    int N = left.total();
    complex<_Tp> *pleft = reinterpret_cast<complex<_Tp> *>(left.data),
    		     *pres  = reinterpret_cast<complex<_Tp> *>(res_mat.data);
    for (int i = 0; i < N; ++i, ++pleft, ++pres)
    {
    	(*pres).real( (*pleft).real() < alpha.real() );
    }
    res = res_mat;
	return 0;
}

template<typename _Tp>
int pw_less(const complex<_Tp> &alpha, const Mat_<Vec<_Tp, 2> > &right, Mat_<Vec<_Tp, 2> > &res)
{
	Mat_<Vec<_Tp, 2> > res_mat(right.dims, right.size, Vec<_Tp, 2>(0,0));
    int N = right.total();
    complex<_Tp> *pright = reinterpret_cast<complex<_Tp> *>(right.data),
    		     *pres  = reinterpret_cast<complex<_Tp> *>(res_mat.data);
    for (int i = 0; i < N; ++i, ++pright, ++pres)
    {
    	(*pres).real( alpha.real() < (*pright).real() );
    }
    res = res_mat;
	return 0;
}

template<typename _Tp>
int pw_lesseq(const Mat_<Vec<_Tp, 2> > &left, const Mat_<Vec<_Tp, 2> > &right, Mat_<Vec<_Tp, 2> > &res)
{
    if (!isGoodMat(left) || !isGoodMat(right) || !sameSize(left, right))
    {
    	return -1;
    }

	Mat_<Vec<_Tp, 2> > res_mat(left.dims, left.size, Vec<_Tp, 2>(0,0));
    int N = left.total();
    complex<_Tp> *pleft = reinterpret_cast<complex<_Tp> *>(left.data), *pright = reinterpret_cast<complex<_Tp> *>(right.data),
    		     *pres  = reinterpret_cast<complex<_Tp> *>(res_mat.data);
    for (int i = 0; i < N; ++i, ++pleft, ++pright, ++pres)
    {
    	(*pres).real( (*pleft).real() <= (*pright).real() );
    }
    res = res_mat;
	return 0;
}

template<typename _Tp>
int pw_lesseq(const Mat_<Vec<_Tp, 2> > &left, const complex<_Tp> &alpha, Mat_<Vec<_Tp, 2> > &res)
{
	Mat_<Vec<_Tp, 2> > res_mat(left.dims, left.size, Vec<_Tp, 2>(0,0));
    int N = left.total();
    complex<_Tp> *pleft = reinterpret_cast<complex<_Tp> *>(left.data),
    		     *pres  = reinterpret_cast<complex<_Tp> *>(res_mat.data);
    for (int i = 0; i < N; ++i, ++pleft, ++pres)
    {
    	(*pres).real( (*pleft).real() <= alpha.real() );
    }
    res = res_mat;
	return 0;
}

template<typename _Tp>
int pw_lesseq(const complex<_Tp> &alpha, const Mat_<Vec<_Tp, 2> > &right, Mat_<Vec<_Tp, 2> > &res)
{
	Mat_<Vec<_Tp, 2> > res_mat(right.dims, right.size, Vec<_Tp, 2>(0,0));
    int N = right.total();
    complex<_Tp> *pright = reinterpret_cast<complex<_Tp> *>(right.data),
    		     *pres  = reinterpret_cast<complex<_Tp> *>(res_mat.data);
    for (int i = 0; i < N; ++i, ++pright, ++pres)
    {
    	(*pres).real( alpha.real() <= (*pright).real() );
    }
    res = res_mat;
	return 0;
}

template<typename _Tp>
int pw_max(const Mat_<Vec<_Tp, 2> > &left, const Mat_<Vec<_Tp, 2> > &right, Mat_<Vec<_Tp, 2> > &res)
{
    if (!isGoodMat(left) || !isGoodMat(right) || !sameSize(left, right))
    {
    	return -1;
    }
//    CV_ASSERT(isGoodMat(left) && isGoodMat(right) && sameSize(left, right));

	Mat_<Vec<_Tp, 2> > res_mat(left.dims, left.size, Vec<_Tp, 2>(0,0));
    int N = right.total();
    complex<_Tp> *pleft = reinterpret_cast<complex<_Tp> *>(left.data),
    			 *pright = reinterpret_cast<complex<_Tp> *>(right.data),
    		     *pres  = reinterpret_cast<complex<_Tp> *>(res_mat.data);
    for (int i = 0; i < N; ++i, ++pleft, ++pright, ++pres)
    {
    	(*pres).real( max<_Tp>((*pleft).real(), (*pright).real()) );
    }
    res = res_mat;
	return 0;
}

template<typename _Tp>
int pw_max(const Mat_<Vec<_Tp, 2> > &left, const complex<_Tp> &alpha, Mat_<Vec<_Tp, 2> > &res)
{
	Mat_<Vec<_Tp, 2> > res_mat(left.dims, left.size, Vec<_Tp, 2>(0,0));
    int N = left.total();
    complex<_Tp> *pleft = reinterpret_cast<complex<_Tp> *>(left.data),
    		     *pres  = reinterpret_cast<complex<_Tp> *>(res_mat.data);
    for (int i = 0; i < N; ++i, ++pleft, ++pres)
    {
    	(*pres).real( max<_Tp>((*pleft).real(), alpha.real()) );
    }
    res = res_mat;
	return 0;
}

template<typename _Tp>
int pw_min(const Mat_<Vec<_Tp, 2> > &left, const Mat_<Vec<_Tp, 2> > &right, Mat_<Vec<_Tp, 2> > &res)
{
    if (!isGoodMat(left) || !isGoodMat(right) || !sameSize(left, right))
    {
    	return -1;
    }
	Mat_<Vec<_Tp, 2> > res_mat(left.dims, left.size, Vec<_Tp, 2>(0,0));
    int N = right.total();
    complex<_Tp> *pleft = reinterpret_cast<complex<_Tp> *>(left.data),
    			 *pright = reinterpret_cast<complex<_Tp> *>(right.data),
    		     *pres  = reinterpret_cast<complex<_Tp> *>(res_mat.data);
    for (int i = 0; i < N; ++i, ++pleft, ++pright, ++pres)
    {
    	(*pres).real( min<_Tp>((*pleft).real(), (*pright).real()) );
    }
    res = res_mat;
	return 0;
}


template<typename _Tp>
int mat_select(const Mat_<Vec<_Tp, 2> > &origin_mat, const SmartArray<SmartIntArray> &index_set_for_each_dim, Mat_<Vec<_Tp, 2> > &sub_mat)
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

	Mat_<Vec<_Tp, 2> > selected(dims, end_pos1);
//	sub_mat.create(dims, end_pos1);
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
			selected.template at<Vec<_Tp, 2> >(src_cur_pos) = origin_mat.template at<Vec<_Tp, 2> >(sel_idx);
			//--

			cur_dim = src_dims - 1;
			src_cur_pos[cur_dim] += src_step[cur_dim];
		}

	}
	sub_mat = selected;

	return 0;
}

template<typename _Tp>
int mat_subfill(Mat_<Vec<_Tp, 2> > &origin_mat, const SmartArray<SmartIntArray> &index_set_for_each_dim, const Mat_<Vec<_Tp, 2> > &sub_mat)
{
	int dims = index_set_for_each_dim.len;
	SmartIntArray start_pos1(dims);
	SmartIntArray cur_pos1(dims);
	SmartIntArray step1(dims, 1);
	SmartIntArray end_pos1(dims);
	SmartIntArray sel_idx(dims);
	SmartArray<SmartIntArray> index_set_for_each_dim_cpy = index_set_for_each_dim.clone();
	for (int i = 0; i < dims; ++i)
	{
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


//	Mat_<Vec<_Tp, 2> > origin_cpy = origin_mat.clone();
//	filled_mat.create(origin_mat.dims, origin_mat.size);
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
			origin_mat.template at<Vec<_Tp, 2> >(sel_idx) = sub_mat.template at<Vec<_Tp, 2> >(src_cur_pos);
			//--

			cur_dim = src_dims - 1;
			src_cur_pos[cur_dim] += src_step[cur_dim];
		}
	}
//	filled_mat = origin_cpy;

	return 0;
}

template<typename _Tp>
int mat_subadd(Mat_<Vec<_Tp, 2> > &origin_mat, const SmartArray<SmartIntArray> &index_set_for_each_dim, const Mat_<Vec<_Tp, 2> > &sub_mat)
{
	int dims = index_set_for_each_dim.len;
	SmartIntArray start_pos1(dims);
	SmartIntArray cur_pos1(dims);
	SmartIntArray step1(dims, 1);
	SmartIntArray end_pos1(dims);
	SmartIntArray sel_idx(dims);
	SmartArray<SmartIntArray> index_set_for_each_dim_cpy = index_set_for_each_dim.clone();
	for (int i = 0; i < dims; ++i)
	{
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
			origin_mat.template at<complex<_Tp> >(sel_idx) += sub_mat.template at<complex<_Tp> >(src_cur_pos);
			//--

			cur_dim = src_dims - 1;
			src_cur_pos[cur_dim] += src_step[cur_dim];
		}
	}

	return 0;
}

template<typename _Tp>
double lpnorm(const Mat_<Vec<_Tp, 2> > &mat, _Tp p)
{
	Mat_<Vec<_Tp, 2> > tmp;
	pw_abs<_Tp>(mat, tmp);
	pw_pow<_Tp>(tmp, p, tmp);

	MatConstIterator_<Vec<_Tp, 2> > it0 = tmp.begin(), end0 = tmp.end();
	_Tp norm = 0.0;
	for (; it0 != end0; ++it0)
	{
		norm += (*it0)[0];
	}

	//TODO 'pow' should be replaced later to conform to template programming
	return pow(norm, 1.0/p);
}

template<typename _Tp>
int pw_min(const Mat_<Vec<_Tp, 2> > &left, const complex<_Tp> &alpha, Mat_<Vec<_Tp, 2> > &res)
{
	Mat_<Vec<_Tp, 2> > res_mat(left.dims, left.size);
//    res.create(left.dims, left.size, left.type(), Scalar(0,0));
    int N = left.total();
    complex<_Tp> *pleft = reinterpret_cast<complex<_Tp> *>(left.data),
    		     *pres  = reinterpret_cast<complex<_Tp> *>(res_mat.data);
    for (int i = 0; i < N; ++i, ++pleft, ++pres)
    {
    	(*pres).real() = min((*pleft).real(), alpha.real());
    }
    res = res_mat;
	return 0;
}

template<typename _Tp>
int psnr(const Mat_<Vec<_Tp, 2> > &left, const Mat_<Vec<_Tp, 2> > &right, double &psnr, double &msr)
{
    if (!isGoodMat(left) || !isGoodMat(right) || !sameSize(left, right))
    {
    	return -1;
    }

	Mat_<Vec<_Tp, 2> > dif = left - right;
	double msr_stat = 0.0;
	MatConstIterator_<Vec<_Tp, 2> > it0 = dif.begin(), end0 = dif.end();

	for (; it0 != end0; ++it0)
	{
		double sqr = static_cast<double>((*it0)[0] * (*it0)[0] + (*it0)[1] * (*it0)[1]);
		msr_stat += sqr;
	}

	msr_stat = sqrt(msr_stat / left.total());

	double psnr_stat = 0.0;

	psnr_stat = log(255.0 / msr_stat);
	psnr_stat = 20 * psnr_stat / log(10);

	psnr = psnr_stat;
	msr = msr_stat;

	return 0;
}


/*
 * Current implementation allocates extra heap memory repeatedly. An in-place solution is found.
 */
template<typename _Tp>
int tensor_product(const SmartArray<Mat_<Vec<_Tp, 2> > > &components_for_each_dim, Mat_<Vec<_Tp, 2> >  &product)
{
	//TODO: make sure all input components are continuous mat, and row vectors.
	int ndims = components_for_each_dim.len;
	SmartIntArray dim_size(ndims);
	Mat sub_mat = components_for_each_dim[ndims - 1];    //row vector
	dim_size[ndims - 1] = components_for_each_dim[ndims - 1].total();
	for (int cur_dim = ndims - 2; cur_dim >= 0; --cur_dim)
	{
		const Mat &cur_dim_comp = components_for_each_dim[cur_dim];   // column vector;
		dim_size[cur_dim] = cur_dim_comp.total();
		sub_mat = cur_dim_comp.t() * sub_mat;	//This is a 2D matrix. Transpose is O(1) operation.
		sub_mat = sub_mat.reshape(0, 1);    //Convert to row vector. O(1) operation
	}


	// No need to zero.
	Mat product_mat(ndims, dim_size, DataType<Vec<_Tp, 2> >::type);
	std::copy(sub_mat.begin<Vec<_Tp, 2> >(), sub_mat.end<Vec<_Tp, 2> >(), product_mat.begin<Vec<_Tp, 2> >());
//	memcpy((void*)product_mat.data, (void*)sub_mat.data, sub_mat.total() * sub_mat.elemSize());
	product = product_mat;

	return 0;
}

//template<typename _Tp>
//int tensor_product(const SmartArray<Mat_<Vec<_Tp, 2> > > &components_at_dim, Mat_<Vec<_Tp, 2> >  &product)
//{
//	//TODO: make sure all input components are continuous mat, and row vectors.
//	int ndims = components_at_dim.len;
//	SmartIntArray tensor_size(ndims);
//	int N = 1;
//	for (int i = 0; i < ndims; ++i)
//	{
//		tensor_size[i] = components_at_dim[i].total();
//		N *= tensor_size[i];
//	}
//
//	int sub_mat_size = tensor_size[ndims - 1];
//	Mat_<Vec<_Tp, 2> > tensor(ndims, tensor_size);
//	memcpy((void*)tensor.data, (void*)components_at_dim[ndims - 1].data, sub_mat_size * sizeof(complex<_Tp>));
//	complex<_Tp> *p0 = reinterpret_cast<complex<_Tp> *>(tensor.data),
//				 *p1 = p0 + sub_mat_size;
//	for (int i = ndims - 2; i >= 0; --i)
//	{
//		complex<_Tp> *p2 = reinterpret_cast<complex<_Tp> *>(components_at_dim[i].data);
//
//		int this_dim_size = components_at_dim[i].total();
//		for (int j = 1; j < this_dim_size; ++j)
//		{
//			complex<_Tp> a = *(p2 + j);
//			for (int k = 0; k < sub_mat_size; ++k)
//			{
//				*p1 = a * (*(p0 + k));
//				++p1;
//			}
//		}
//
//		for (int k = 0; k < sub_mat_size; ++k)
//		{
//			*(p0 + k) = (*p2) * (*(p0 + k));
//		}
//		sub_mat_size *= this_dim_size;
//	}
//
//	product = tensor;
//	return 0;
//}

template<typename _Tp, int cn>
void print_mat_details_g(const Mat_<Vec<_Tp, cn> > &mat, int field = cn, const string &filename = "cout")
{
	streambuf *stdoutbuf = cout.rdbuf();
	ofstream outfile;
	if (filename != "cout")
	{
		outfile.open(filename.c_str(), ios_base::out | ios_base::app);
		cout.rdbuf(outfile.rdbuf());
	}
	int max_term = 24;
	bool first_row = true;

	int dims = mat.dims;
	SmartIntArray pos(dims);
	SmartIntArray range(dims, mat.size);


	cout << setiosflags(ios::fixed) << setprecision(3);
	int cur_dim = dims - 1;
	int t = 0;
	while(true)
	{
		while (cur_dim >= 0 && pos[cur_dim] >= range[cur_dim])
		{
			pos[cur_dim] = 0;
			--cur_dim;

			if (cur_dim >= 0)
			{
				++pos[cur_dim];
				continue;
			}
		}

		if (cur_dim < 0)
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

		if (field == cn)
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
			Vec<_Tp, cn> ele = mat(pos);
			for (int i = 0; i < cn; ++i)
			{
				sum += ele[i] * ele[i];
			}

			cout << sqrt(sum);
		}
		else if (field >= 0 && field < cn)
		{
			cout << mat(pos)[field];
		}
		else
		{
			return;
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

		cur_dim = mat.dims - 1;
		++pos[cur_dim];
	}
	cout << endl;

	if (filename != "cout")
	{
		cout.rdbuf(stdoutbuf);
	}
}

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

template<typename _Tp>
int load_as_tensor(const string &filename, Mat_<Vec<_Tp, 2> > &output, Media_Format *media_file_fmt)
{
	size_t pos = filename.find_last_of('.');
	if (pos == std::string::npos)
	{
		return -1;
	}

	string suffix = filename.substr(pos);
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

		output.create(img.size());
		for (int r = 0; r < img.rows; ++r)
		{
			for (int c = 0; c < img.cols; ++c)
			{
				output.template at<Vec<_Tp, 2> >(r, c)[0] = (_Tp)(img.at<uchar>(r, c));
				output.template at<Vec<_Tp, 2> >(r, c)[1] = 0;
			}
		}

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


		output.create(3, sz);
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

			Mat_<Vec<_Tp, 2> > plane = output.row(f);
			for (int r = 0; r < sz[1]; ++r)
			{
				for (int c = 0; c < sz[2]; ++c)
				{
					plane.template at<Vec<_Tp, 2> >(0, r, c)[0] = (_Tp)(frame.at<uchar>(r, c));
					plane.template at<Vec<_Tp, 2> >(0, r, c)[1] = 0;
				}
			}
		}

		return 0;
	}

	return -4;
}

template<typename _Tp>
int save_as_media(const string &filename, const Mat_<Vec<_Tp, 2> > &mat, const Media_Format *media_file_fmt)
{
    if (mat.dims == 2 && mat.size[0] != 1)
    {
//    	Mat_<Vec<_Tp, 2> > img_r(mat.size(), CV_64FC1);
//    	Mat_<Vec<_Tp, 2> > img_i(mat.size(), CV_64FC1);
//	    Mat_<Vec<_Tp, 1> > img_abs(mat.size());
    	Mat img_abs(mat.size(), CV_64FC1);

		 for (int i = 0; i < mat.rows; i++)
		 {
			for (int j = 0; j < mat.cols; j++)
			{
				double d = (double)(mat.template at<Vec<_Tp, 2> >(i, j)[0] * mat.template at<Vec<_Tp, 2> >(i, j)[0]
						 + mat.template at<Vec<_Tp, 2> >(i, j)[1] * mat.template at<Vec<_Tp, 2> >(i, j)[1]);

				//TODO sqrt is not good practice for template programming.
				//TODO BIG Problem. The two statement below produce different pictures.
				img_abs.at<double>(i, j) = (double)sqrt(d);
	//         		img_abs.at<double>(i, j) = mat(i,j)[0];
			}

		 }

         Mat scaled;
         double d0, d1;
         minMaxLoc(img_abs, &d0, &d1);
//         img_abs.convertTo(scaled, CV_8UC1, 255.0 / (d1 - d0), -255.0 / (d1 - d0));
//         img_abs.convertTo(scaled, CV_8UC1, 1, 0);
//         imwrite(filename, scaled);

//         minMaxLoc(img_i, &d0, &d1);
//         img_i.convertTo(img_i, CV_8UC1, 255.0 / (d1 - d0), -255.0 / (d1 - d0));
//         imwrite(filename + "-i.jpg", img_i);

         img_abs.convertTo(img_abs, CV_8UC1, 1, 0);
         imwrite(filename, img_abs);

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
    		Mat_<Vec<_Tp, 2> > plane = mat.row(f);
            Mat frame(Size(plane.size[2], plane.size[1]), CV_64FC1);

            for (int i = 0; i < mat.size[1]; i++)
            {
            	for (int j = 0; j < mat.size[2]; j++)
            	{
            		double d = (_Tp)(plane.template at<Vec<_Tp, 2> >(0, i, j)[0] * plane.template at<Vec<_Tp, 2> >(0, i, j)[0]
            		         + plane.template at<Vec<_Tp, 2> >(0, i, j)[1] * plane.template at<Vec<_Tp, 2> >(0, i, j)[1]);

            		frame.at<double>(i, j) = sqrt(d);
            	}
            }

            frame.convertTo(frame, CV_8UC1, 1, 0);
            writer << frame;
    	}
    }
	return 0;
}


template <typename _Tp>
int mat_border_cut(const Mat_<Vec<_Tp, 2> > &extended, const SmartIntArray &border, Mat_<Vec<_Tp, 2> > &origin)
{
	int ndims = extended.dims;
	if (ndims <= 0 || ndims != border.len)
	{
		return -1;
	}

	//TODO: check if size[i] > border[i]
	SmartIntArray ext_size(ndims, extended.size);
	SmartIntArray dst_start_pos(ndims);
	SmartIntArray dst_pos(ndims);
	SmartIntArray start_pos(ndims);
	SmartIntArray cur_pos(ndims);
	SmartIntArray step(ndims, 1);
	SmartIntArray end_pos(ndims);
	for (int i = 0; i < ndims; ++i)
	{
		end_pos[i] = ext_size[i] - border[i];
		dst_start_pos[i] = border[i] / 2;
		dst_pos[i] = dst_start_pos[i];
	}

	Mat_<Vec<_Tp, 2> > cut_mat(ndims, end_pos);
	{
		int src_dims;
		SmartIntArray src_start_pos;
		SmartIntArray src_cur_pos;
		SmartIntArray src_step;
		SmartIntArray src_end_pos;

		//User-Defined initialization
		src_dims = ndims;
		src_start_pos = start_pos;
		src_cur_pos = cur_pos;
		src_step = step;
		src_end_pos = end_pos;
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
				dst_pos[cur_dim] = src_cur_pos[cur_dim] + dst_start_pos[cur_dim];
			}
			cut_mat.template at<Vec<_Tp, 2> >(src_cur_pos) = extended.template at<Vec<_Tp, 2> >(dst_pos);
			//--

			cur_dim = src_dims - 1;
			src_cur_pos[cur_dim] += src_step[cur_dim];
		}
	}

//	delete [] dst_start_pos;
//	delete [] dst_pos;
//	delete [] start_pos;
//	delete [] cur_pos;
//	delete [] step;
//	delete [] end_pos;

	origin = cut_mat;

	return 0;
}

template<typename _Tp>
int mat_border_extension(const Mat_<Vec<_Tp, 2> > &origin, const SmartIntArray &border, const string &opt, Mat_<Vec<_Tp, 2> > &extended)
{
	if (origin.dims <= 0 || origin.dims != border.len)
	{
		return -1;
	}

	//TODO: Check if border[i] is less than origin.size[i]

	int bd_mode = -1;
	if (opt == "sym")
	{
		bd_mode = 0;
	}
	else if (opt == "mir101")
	{
		bd_mode = 1;
	}
	else if (opt == "mir1001")
	{
		bd_mode = 2;
	}
	else if (opt == "blk")
	{
		bd_mode = 3;
	}
	else if (opt == "cut")
	{
		Mat_<Vec<_Tp, 2> > cut;
		mat_border_cut<_Tp>(origin, border, cut);
		extended = cut;

		return 0;
	}
	else
	{
		return -2;
	}

	int ndims = origin.dims;
	SmartIntArray origin_size(ndims ,origin.size);
	SmartIntArray dst_pos(ndims);
	SmartIntArray shift(ndims);
	SmartIntArray start_pos(ndims);
	SmartIntArray cur_pos(ndims);
	SmartIntArray step(ndims, 1);
	SmartIntArray ext_size(ndims);


//	shift = new int[n];
//	dst_pos = new int[n];
//	start_pos = new int[n];
//	cur_pos = new int[n];
//	step = new int[n];
//	ext_size = new int[n];
	for (int i = 0; i < ndims; ++i)
	{
		shift[i] = border[i] / 2;
		dst_pos[i] = start_pos[i] - shift[i];
		ext_size[i] = origin_size[i] + border[i];
	}

	Mat_<Vec<_Tp, 2> > ext_mat(ndims, ext_size);
	{
		int src_dims;
		SmartIntArray src_start_pos;
		SmartIntArray src_cur_pos;
		SmartIntArray src_step;
		SmartIntArray src_end_pos;


		//User-Defined initialization
		src_dims = ndims;
		src_start_pos = start_pos;
		src_cur_pos = cur_pos;
		src_step = step;
		src_end_pos = ext_size;
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
			bool invld = false;
			for (cur_dim = 0; cur_dim < src_dims; ++cur_dim)
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
				else if (bd_mode == 1)  //mir1, no duplicate for first and last elements. 101
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
				else if (bd_mode == 2)  //mir2 1001
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
				ext_mat.template at<Vec<_Tp, 2> >(src_cur_pos) = Vec<_Tp, 2>(0,0);
			}
			else
			{
				ext_mat.template at<Vec<_Tp, 2> >(src_cur_pos) = origin.template at<Vec<_Tp, 2> >(dst_pos);
			}
			//--

			cur_dim = src_dims - 1;
			src_cur_pos[cur_dim] += src_step[cur_dim];
		}

	}
	extended = ext_mat;
	return 0;
}

template<typename _Tp>
int interpolate(const Mat_<Vec<_Tp, 2> > &input, const SmartIntArray &times, Mat_<Vec<_Tp, 2> > &output)
{
	int ndims = input.dims;
	if (ndims != times.len)
	{
		return -1;
	}
	SmartIntArray new_range(ndims, input.size);
	SmartIntArray input_range(ndims, input.size);
	for (int i = 0; i < ndims; ++i)
	{
		new_range[i] *= times[i];
	}

	SmartIntArray cur_pos1(ndims);
	Mat_<Vec<_Tp, 2> > expanded(ndims, new_range, Vec<_Tp, 2>(0,0));
	{
		int src_dims;
		SmartIntArray src_start_pos;
		SmartIntArray src_cur_pos;
		SmartIntArray src_step;
		SmartIntArray src_end_pos;

		//User-Defined initialization
		src_dims = ndims;
		src_start_pos = SmartIntArray(ndims, 0);
		src_cur_pos = cur_pos1;
		src_step = SmartIntArray(ndims, 1);
		src_end_pos = input_range;
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
			SmartIntArray cur_pos2 = cur_pos1.clone();
			SmartIntArray step2(ndims);
			for (int i = 0; i < ndims; ++i)
			{
				cur_pos2[i] *= times[i];
				step2[i] = cur_pos2[i] + times[i];
			}
			{
				int src_dims;
				SmartIntArray src_start_pos;
				SmartIntArray src_cur_pos;
				SmartIntArray src_step;
				SmartIntArray src_end_pos;

				//User-Defined initialization
				src_dims = ndims;
				src_start_pos = cur_pos2.clone();
				src_cur_pos = cur_pos2;
				src_step = SmartIntArray(ndims, 1);
				src_end_pos = step2;
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

					expanded.template at<Vec<_Tp, 2> >(src_cur_pos) = input.template at<Vec<_Tp, 2> >(cur_pos1);

					cur_dim = src_dims - 1;
					src_cur_pos[cur_dim] += src_step[cur_dim];
				}
			}
			//--

			cur_dim = src_dims - 1;
			src_cur_pos[cur_dim] += src_step[cur_dim];
		}
	}

	output = expanded;

	return 0;
}

template<typename _Tp>
int md_filtering(const Mat_<Vec<_Tp, 2> > &input, const Mat_<Vec<_Tp, 2> > &filter, const SmartIntArray &center, Mat_<Vec<_Tp, 2> > &filtered)
{
	int input_dims = input.dims;
	SmartIntArray start_pos1(input_dims);
	SmartIntArray cur_pos1(input_dims);
	SmartIntArray step1(input_dims, 1);
	SmartIntArray range1(input_dims, filter.size);


	Mat_<Vec<_Tp, 2> > full_filter(input.dims, input.size, Vec<_Tp, 2>(0,0));
	{
		int src_dims;
		SmartIntArray src_start_pos;
		SmartIntArray src_cur_pos;
		SmartIntArray src_step;
		SmartIntArray src_end_pos;

		//User-Defined initialization
		src_dims = input_dims;
		src_start_pos = start_pos1;
		src_cur_pos = cur_pos1;
		src_step = step1;
		src_end_pos = range1;
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
			full_filter.template at<Vec<_Tp, 2> >(src_cur_pos) = filter.template at<Vec<_Tp, 2> >(src_cur_pos);
			//--

			cur_dim = src_dims - 1;
			src_cur_pos[cur_dim] += src_step[cur_dim];
		}
	}


	Mat_<Vec<_Tp, 2> > fd_filter, fd_input;
	normalized_fft<_Tp>(full_filter, fd_filter);
	normalized_fft<_Tp>(input, fd_input);
	pw_mul<_Tp>(fd_input, fd_filter, fd_input);
	cur_pos1 = SmartIntArray::konst(input_dims, 0);
	range1 = SmartIntArray(input_dims, fd_input.size);
	{
		int src_dims;
		SmartIntArray src_start_pos;
		SmartIntArray src_cur_pos;
		SmartIntArray src_step;
		SmartIntArray src_end_pos;

		//User-Defined initialization
		src_dims = input_dims;
		src_start_pos = start_pos1;
		src_cur_pos = cur_pos1;
		src_step = step1;
		src_end_pos = range1;
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
			double dot = 0.0;
			for (int i = 0; i < src_dims; ++i)
			{
				dot += (double)center[i] * src_cur_pos[i] / range1[i];
			}
			fd_input.template at<complex<_Tp> >(src_cur_pos) *= complex<_Tp>(cos(2 * M_PI * dot), sin(2 * M_PI * dot));
			//--

			cur_dim = src_dims - 1;
			src_cur_pos[cur_dim] += src_step[cur_dim];
		}
	}
	normalized_ifft<_Tp>(fd_input, filtered);

	filtered = filtered * sqrt(filtered.total());

	return 0;
}

template <typename _Tp>
int rotate180shift1(const Mat_<Vec<_Tp, 2> > &mat, Mat_<Vec<_Tp, 2> > &rot_mat)
{
	int ndims = mat.dims;
	SmartIntArray start_pos(ndims);
	SmartIntArray cur_pos(ndims);
	SmartIntArray step(ndims, 1);
	SmartIntArray range(ndims, mat.size);
	SmartIntArray sym_cur_pos(ndims);
//	for (int i = 0; i < ndims; ++i)
//	{
//		sym_cur_pos[i] = range[i] - 1;
//	}
	Mat_<Vec<_Tp, 2> > tmp(ndims, range, Vec<_Tp, 2>(0,0));
	{
		int src_dims;
		SmartIntArray src_start_pos;
		SmartIntArray src_cur_pos;
		SmartIntArray src_step;
		SmartIntArray src_end_pos;

		//User-Defined initialization
		src_dims = ndims;
		src_start_pos = start_pos;
		src_cur_pos = cur_pos;
		src_step = step;
		src_end_pos = range;
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
				sym_cur_pos[cur_dim] = range[cur_dim] - src_cur_pos[cur_dim];
				if (sym_cur_pos[cur_dim] >= range[cur_dim])
				{
					sym_cur_pos[cur_dim] = 0;
				}
			}
			tmp.template at<Vec<_Tp, 2> >(sym_cur_pos) = mat.template at<Vec<_Tp, 2> >(src_cur_pos);
			//--

			cur_dim = src_dims - 1;
			src_cur_pos[cur_dim] += src_step[cur_dim];
		}
	}

	rot_mat = tmp;
	return 0;
}

template <typename _Tp>
int rotate180(const Mat_<Vec<_Tp, 2> > &mat, Mat_<Vec<_Tp, 2> > &rot_mat)
{
	int ndims = mat.dims;
	SmartIntArray start_pos(ndims);
	SmartIntArray cur_pos(ndims);
	SmartIntArray step(ndims, 1);
	SmartIntArray range(ndims, mat.size);
	SmartIntArray sym_cur_pos(ndims);
	for (int i = 0; i < ndims; ++i)
	{
		sym_cur_pos[i] = range[i] - 1;
	}
	Mat_<Vec<_Tp, 2> > tmp(ndims, range, Vec<_Tp, 2>(0,0));
	{
		int src_dims;
		SmartIntArray src_start_pos;
		SmartIntArray src_cur_pos;
		SmartIntArray src_step;
		SmartIntArray src_end_pos;

		//User-Defined initialization
		src_dims = ndims;
		src_start_pos = start_pos;
		src_cur_pos = cur_pos;
		src_step = step;
		src_end_pos = range;
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
				sym_cur_pos[cur_dim] = range[cur_dim] - 1 - src_cur_pos[cur_dim];
			}
			tmp.template at<Vec<_Tp, 2> >(sym_cur_pos) = mat.template at<Vec<_Tp, 2> >(src_cur_pos);
			//--

			cur_dim = src_dims - 1;
			src_cur_pos[cur_dim] += src_step[cur_dim];
		}
	}

	rot_mat = tmp;
	return 0;
}

template <typename _Tp>
int conj(Mat_<Vec<_Tp, 2> > &mat)
{
	Mat_<Vec<_Tp, 2> > tmp(mat.dims, mat.size);
	Vec<_Tp, 2> *pa = reinterpret_cast<Vec<_Tp, 2> *>(mat.data);
	int N = mat.total();
	for (int i = 0; i < N; ++i, ++pa)
	{
		(*pa)[1] = -(*pa)[1];
	}

	return 0;
}


#endif
