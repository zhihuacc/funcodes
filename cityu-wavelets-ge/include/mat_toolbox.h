#ifndef WAVELETS_TOOLBOX_H
#define WAVELETS_TOOLBOX_H

#include <fftw3.h>
#include <opencv2/core/core.hpp>
#include "hammers_and_nails.h"

using namespace cv;

template<typename _Tp, int cn>
bool isGoodMat(const Mat_<Vec<_Tp, cn> > &domain)
{
	if (domain.empty() || domain.isContinous()
		|| (domain.depth() != CV_64F && domain.depth() != CV_32F) || domain.channels() != 2)
	{
		return false;
	}

	return true;
}
/*
 * Do DFT transform. Make sure domain isContinuous().
 *
 * time_domain: input time-domain signal.
 * feq_domain: output frequency-domain spectrum. time_domain and freq_domain can be the same Mat.
 */
template<typename _Tp, int cn>
int normalized_fft(const Mat_<Vec<_Tp, cn> > &time_domain, Mat_<Vec<_Tp, cn> > &freq_domain)
{
	//TODO: need to check if the two matrix match in size.
	if (!isGoodMat(time_domain))
	{
		return -1;
	}

	fftw_complex *before;
	fftw_complex *after;
	fftw_plan plan;

	before = time_domain.data;
	after = freq_domain.data;

	plan = fftw_plan_dft(time_domain.dims, time_domain.size, before, after, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(plan);
    fftw_destroy_plan(plan);

    freq_domain = freq_domain  * (1.0 / sqrt(freq_domain.total()));

    return 0;
}
/*
 * Do DFT transform
 *
 * feq_domain: input frequency-domain spectrum.
 * time_domain: output time-domain signal.time_domain and freq_domain can be the same Mat.
 */
template<typename _Tp, int cn>
int normalized_ifft(const Mat_<Vec<_Tp, cn> > &freq_domain, Mat_<Vec<_Tp, cn> > &time_domain)
{
	//TODO: need to check if the two matrix match in size.
	if (!isGoodMat(freq_domain))
	{
		return -1;
	}

	fftw_complex *before;
	fftw_complex *after;
	fftw_plan plan;

	before = freq_domain.data;
	after = before;

	plan = fftw_plan_dft(freq_domain.dims, freq_domain.size, before, after, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(plan);
    fftw_destroy_plan(plan);

    time_domain = time_domain  * (1.0 / sqrt(time_domain.total()));

    return 0;
}

/*
 * Half shift the matrix. This is NOT done in-place. 'shifted' can NOT be equal to 'data'.
 */
template<typename _Tp, int cn>
int center_shift(const Mat_<Vec<_Tp, cn> > &data, Mat_<Vec<_Tp, cn> > &shifted)
{
	if (!isGoodMat(data) || !isGoodMat(shifted))
	{
		return -1;
	}

	int ndims = data.dims;
	SmartIntArray start_pos1(ndims);
	SmartIntArray cur_pos1(ndims);
	SmartIntArray step1(ndims);
	SmartIntArray range1(ndims, data.size);
	SmartIntArray offset(ndims);
	SmartIntArray pos2(ndims);

	for (int i = 0; i < ndims; ++i)
	{
		step1[i] = 1;
		offset[i] = range1[i] >> 1;
		pos2[i] = offset[i];
	}


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
				pos2[cur_dim] = (src_cur_pos[cur_dim] + offset[cur_dim]) % range1[cur_dim];
			}
			shifted(pos2) = data(src_cur_pos);
			//--

			cur_dim = src_dims - 1;
			src_cur_pos[cur_dim] += src_step[cur_dim];
		}

	}

	return 0;
}

/*
 * Half shift back the matrix. This is NOT done in-place. 'shifted' can NOT be equal to 'data'.
 */
template<typename _Tp, int cn>
int icenter_shift(const Mat_<Vec<_Tp, cn> > &data, Mat_<Vec<_Tp, cn> > &shifted)
{
	if (!isGoodMat(data) || !isGoodMat(shifted))
	{
		return -1;
	}

	int ndims = data.dims;
	SmartIntArray start_pos1(ndims);
	SmartIntArray cur_pos1(ndims);
	SmartIntArray step1(ndims);
	SmartIntArray range1(ndims, data.size);
	SmartIntArray offset(ndims);
	SmartIntArray pos2(ndims);

	for (int i = 0; i < ndims; ++i)
	{
		step1[i] = 1;
		offset[i] = (range1[i] & 1) ? (range1[i] - (range1[i] >> 1)) : range1[i] >> 1;
		pos2[i] = offset[i];
	}


//	Mat_<T> shifted_mat(range1.size(), (const int*)range1, data.type());
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
				pos2[cur_dim] = (src_cur_pos[cur_dim] + offset[cur_dim]) % range1[cur_dim];
			}
			shifted(pos2) = data(src_cur_pos);
			//--

			cur_dim = src_dims - 1;
			src_cur_pos[cur_dim] += src_step[cur_dim];
		}

	}

	return 0;
}

/*
 * Do complex-value multiplication point-wisely. Make sure that product has been allocated memory properly
 * before calling. NO memory allocation happens inside.
 */
template<typename _Tp, int cn>
int pw_mul(const Mat_<Vec<_Tp, cn> > &left, const Mat_<Vec<_Tp, cn> > &right, Mat_<Vec<_Tp, cn> > &product)
{
    // Need more check if that left and right match
    if (!isGoodMat(left) || !isGoodMat(right))
    {
    	return -1;
    }

    MatIterator_<Vec<_Tp, cn> > it0 = product.template begin<Vec<_Tp, cn> >(), end0 = product.template end<Vec<_Tp, cn> >();
    MatConstIterator_<Vec<_Tp, cn> > it1 = left.template begin<Vec<_Tp, cn> >(), it2 = right.template begin<Vec<_Tp, cn> >();

    for (; it0 != end0; ++it0, ++it1, ++it2)
    {
    	(*it0)[0] = (*it1)[0] * (*it2)[0] - (*it1)[1] * (*it2)[1];
    	(*it0)[1] = (*it1)[0] * (*it2)[1] + (*it1)[1] * (*it2)[0];
    }

    return 0;
}

/*
 * Do complex-value pow point-wisely. NO memory allocation happens inside.
 */
template<typename _Tp, int cn>
int pw_pow(const Mat_<Vec<_Tp, cn> > &base, double expo, Mat_<Vec<_Tp, cn> > &res)
{

	if (!isGoodMat(base) || !isGoodMat(res))
	{
		return -1;
	}

    MatIterator_<Vec<_Tp, cn> > it0 = res.template begin<Vec<_Tp, cn> >(), end0 = res.template end<Vec<_Tp, cn> >();
    MatConstIterator_<Vec<_Tp, cn> > it1 = base.begin();

	complex<_Tp> *b, c;
    for (; it0 != end0; ++it0, ++it1)
    {
    	Vec<_Tp, cn> elem = *it1;
    	b = reinterpret_cast<complex<_Tp> *>(&elem);
    	c = pow(*b, expo);
    	*it0 = *reinterpret_cast<Vec<_Tp, cn> *>(&c);
    }

	return 0;
}

/*
 * Do complex-value sqrt point-wisely. NO memory allocation happens inside.
 */
template<typename _Tp, int cn>
int pw_sqrt(const Mat_<Vec<_Tp, cn> > &base, Mat_<Vec<_Tp, cn> > &res)
{
	if (!isGoodMat(base) || !isGoodMat(res))
	{
		return -1;
	}

    MatIterator_<Vec<_Tp, cn> > it0 = res.begin(), end0 = res.end();
    MatConstIterator_<Vec<_Tp, cn> > it1 = base.begin();
    complex<_Tp> *b, c;
    for (; it0 != end0; ++it0, ++it1)
    {
    	Vec<_Tp, cn>  elem = *it1;
    	b = reinterpret_cast<complex<_Tp> *>(&elem);
    	c = sqrt(*b);
    	*it0 = *reinterpret_cast<Vec<_Tp, cn> *>(&c);
    }

	return 0;
}

/*
 * Current implementation allocates extra heap memory repeatedly. An in-place solution is found.
 */
template<typename _Tp, int cn>
int tensor_product(const SmartArray<Mat_<Vec<_Tp, cn> > > &components_for_each_dim, Mat_<Vec<_Tp, cn> >  &product)
{
	//TODO: make sure all input components are continuous mat, and row vectors.
	int dims = components_for_each_dim.len;
	SmartIntArray dim_size(dims);
	Mat_<Vec<_Tp, cn> > sub_mat = components_for_each_dim[dims - 1];    //row vector
	dim_size[dims - 1] = sub_mat.total();
	for (int cur_dim = dims - 2; cur_dim >= 0; --cur_dim)
	{
		const Mat_<Vec<_Tp, cn> > &cur_dim_mat = components_for_each_dim[cur_dim];   // column vector;
		dim_size[cur_dim] = cur_dim_mat.total();
		sub_mat = cur_dim_mat.t() * sub_mat;	//This is a 2D matrix. Transpose is O(1) operation.
		sub_mat = sub_mat.reshape(0, 1);    //Convert to row vector. O(1) operation
	}

	Mat reshaped(dims, (const int *)dim_size, sub_mat.type());
	MatConstIterator_<Vec<_Tp, cn> > it0 = sub_mat.template begin<Vec<_Tp, cn> >(), end0 = sub_mat.template end<Vec<_Tp, cn> >();
	MatIterator_<Vec<_Tp, cn> > it1 = reshaped.template begin<Vec<_Tp, cn> >();
	for (; it0 != end0; ++it0, ++it1)
	{
		*it1 = *it0;
	}
	product = reshaped;
//	print_mat_details_g<Vec2d>(reshaped, 0, "Test-Data/tensor-product.txt");

	return 0;
}

#endif
