#include "../include/tensor.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fftw3.h>

#include <memory>


Tensor::Tensor():_order(0){}

// TODO: Need more attention on dimension. need handle exception
Tensor::Tensor(int d, const int *size)
{
	_order = d;
	if (d == 2 && size[0] == 1)
	{
		_order = 1;
	}
	_data_mat = Mat(d, size, CV_64FC2, Scalar(0, 0));
}

Tensor::Tensor(int order, const Mat &m):_order(order), _data_mat(m)
{
}


Tensor::~Tensor()
{

}


int Tensor::dft(Tensor &output) const
{
	fftw_complex *before;
	fftw_complex *after;
	fftw_plan plan;
	int N;

	N = _data_mat.total();
	before = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
	after = before;     // in-place transform to save space.

	plan = fftw_plan_dft(_data_mat.dims, _data_mat.size, before, after, FFTW_FORWARD, FFTW_ESTIMATE);

	// Initialize input.
	MatConstIterator_<Vec2d> cit = _data_mat.begin<Vec2d>();
	MatConstIterator_<Vec2d> cit_end = _data_mat.end<Vec2d>();
	for(int i = 0; cit != cit_end; ++cit, ++i)
	{
		before[i][0] = (*cit)[0];
		before[i][1] = (*cit)[1];
	}


	fftw_execute(plan);

	// Prepare output
	Mat transformed_mat(_data_mat.dims, _data_mat.size, CV_64FC2);
	MatIterator_<Vec2d> it = transformed_mat.begin<Vec2d>();
	MatIterator_<Vec2d> it_end = transformed_mat.end<Vec2d>();
	for(int j = 0; it != it_end; ++it, ++j)
	{
		(*it)[0] = after[j][0];
		(*it)[1] = after[j][1];
	}

	output = Tensor(_order, transformed_mat);

    fftw_destroy_plan(plan);
    fftw_free(before);
    before = NULL;

	return 0;
}

int Tensor::idft(Tensor &output) const
{
	fftw_complex *before;
	fftw_complex *after;
	fftw_plan plan;
	int N;

	N = _data_mat.total();
	if (N <= 0)
	{
		return -1;
	}

	before = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
	after = before;     // in-place transform to save space.

	plan = fftw_plan_dft(_data_mat.dims, _data_mat.size, before, after, FFTW_BACKWARD, FFTW_ESTIMATE);

	// Initialize input.
	MatConstIterator_<Vec2d> cit = _data_mat.begin<Vec2d>();
	MatConstIterator_<Vec2d> cit_end = _data_mat.end<Vec2d>();
	for(int i = 0; cit != cit_end; ++cit, ++i)
	{
		before[i][0] = (*cit)[0];
		before[i][1] = (*cit)[1];
	}


	fftw_execute(plan);

	// Prepare output
	Mat transformed_mat(_data_mat.dims, _data_mat.size, CV_64FC2);
	MatIterator_<Vec2d> it = transformed_mat.begin<Vec2d>();
	MatIterator_<Vec2d> it_end = transformed_mat.end<Vec2d>();
	for(int i = 0; it != it_end; ++it, ++i)
	{
		(*it)[0] = after[i][0];
		(*it)[1] = after[i][1];
	}

	transformed_mat = transformed_mat * (1.0 / N);
	output = Tensor(_order, transformed_mat);

    fftw_destroy_plan(plan);
    fftw_free(before);
    before = NULL;

	return 0;
}

int Tensor::idft2(Tensor &output) const
{
	fftw_complex *before;
	fftw_complex *after;
	fftw_plan plan;
	int N;

	N = _data_mat.total();
	if (N <= 0)
	{
		return -1;
	}

	before = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
	after = before;     // in-place transform to save space.

	plan = fftw_plan_dft(_data_mat.dims, _data_mat.size, before, after, FFTW_BACKWARD, FFTW_ESTIMATE);

	// Initialize input.
	MatConstIterator_<Vec2d> cit = _data_mat.begin<Vec2d>();
	MatConstIterator_<Vec2d> cit_end = _data_mat.end<Vec2d>();
	for(int i = 0; cit != cit_end; ++cit, ++i)
	{
		before[i][0] = (*cit)[0];
		before[i][1] = (*cit)[1];
	}


	fftw_execute(plan);

	// Prepare output
	Mat transformed_mat(_data_mat.dims, _data_mat.size, CV_64FC2);
	MatIterator_<Vec2d> it = transformed_mat.begin<Vec2d>();
	MatIterator_<Vec2d> it_end = transformed_mat.end<Vec2d>();
	for(int i = 0; it != it_end; ++it, ++i)
	{
		(*it)[0] = after[i][0];
		(*it)[1] = after[i][1];
	}

	transformed_mat = transformed_mat * (1.0 / N);
	output = Tensor(_order, transformed_mat);

    fftw_destroy_plan(plan);
    fftw_free(before);
    before = NULL;

	return 0;
}

int Tensor::center_shift(Tensor &output) const
{
	Mat mat = _data_mat;

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

	output = Tensor(_order, shifted);

	return 0;
}

int Tensor::icenter_shift(Tensor &output) const
{
	Mat mat = _data_mat;

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

	output = Tensor(_order, shifted);

	return 0;
}

int Tensor::pw_mul(const Tensor &other, Tensor &product) const
{
    Mat left = _data_mat, right = other._data_mat;

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

    product = Tensor(_order, product_mat);

    return 0;
}

int Tensor::pw_add(const Tensor &other, Tensor &sum) const
{
    Mat left = _data_mat, right = other._data_mat;

    // Need more check
    if (left.dims != right.dims)
    {
    	return -1;
    }

    Mat sum_mat(left.dims, left.size, CV_64FC2);
    MatIterator_<Vec2d> it0 = sum_mat.begin<Vec2d>(), end0 = sum_mat.end<Vec2d>();
    MatConstIterator_<Vec2d> it1 = left.begin<Vec2d>(), it2 = right.begin<Vec2d>();

    for (; it0 != end0; ++it0, ++it1, ++it2)
    {
    	(*it0)[0] = (*it1)[0] + (*it2)[0];
    	(*it0)[1] = (*it1)[1] + (*it2)[1];
    }

    sum = Tensor(_order, sum_mat);

    return 0;
}

int Tensor::conv(const Tensor &filter, Tensor &output) const
{
	Tensor fd0, fd1, fd2;

	// Check dimension

	this->dft(fd1);
	filter.dft(fd2);

	fd1.pw_mul(fd2, fd0);
	fd0.idft(output);

	return 0;
}

int Tensor::set_value(int *idx, double *val)
{
	// Need more check on idx
    _data_mat.at<Vec2d>(idx)[0] = val[0];
    _data_mat.at<Vec2d>(idx)[1] = val[1];

    return 0;
}

int Tensor::downsample_by2(Tensor &output) const
{
	Mat mat = _data_mat;
	int *pos = new int[mat.dims];
	int *pos2 = new int[mat.dims];
	const int *range = mat.size;
	int *decimated_size = new int[mat.dims];
	int dims = mat.dims;

	for (int i = 0; i < dims; i++)
	{
		pos[i] = 0;
		pos2[i] = 0;
		decimated_size[i] = range[i] < 2 ? 1 : (range[i] / 2);    // size must be power of 2
	}

	Mat decimated(mat.dims, decimated_size, CV_64FC2);
	int i = dims - 1;
	while(true)
	{
		while (i >= 0 && pos[i] >= range[i])
		{
			pos[i] = 0;
			pos2[i] = 0;
			--i;
			if (i >= 0)
			{
				pos[i] += 2;
				pos2[i] += 1;
				continue;
			}
		}

		if (i < 0)
		{
			break;
		}


		decimated.at<Vec2d>(pos2)[0] = mat.at<Vec2d>(pos)[0];
		decimated.at<Vec2d>(pos2)[1] = mat.at<Vec2d>(pos)[1];

		i = mat.dims - 1;
		pos[i] += 2;
		pos2[i] += 1;
	}

	delete [] pos;
	delete [] pos2;
	delete [] decimated_size;

	output = Tensor(_order, decimated);

	return 0;
}

int Tensor::upsample_by2(const SmartArray &restored_size, Tensor &output) const
{

	//TODO: restored_size.dims == _data_mat.dims

	int dims = _data_mat.dims;
	int *pos = new int[dims];
	int *pos2 = new int[dims];
	const int *range = _data_mat.size;
	const int *expanded_size = (const int*)restored_size;

	for (int i = 0; i < dims; i++)
	{
		pos[i] = 0;
		pos2[i] = 0;
	}

	Mat expanded(dims, expanded_size, CV_64FC2, Scalar(0, 0));
	int i = dims - 1;
	while(true)
	{
		while (i >= 0 && pos[i] >= range[i])
		{
			pos[i] = 0;
			pos2[i] = 0;
			--i;
			if (i >= 0)
			{
				++pos[i];
				pos2[i] = pos[i] * 2;
				continue;
			}
		}

		if (i < 0)
		{
			break;
		}


		expanded.at<Vec2d>(pos2)[0] = _data_mat.at<Vec2d>(pos)[0];
		expanded.at<Vec2d>(pos2)[1] = _data_mat.at<Vec2d>(pos)[1];

		i = dims - 1;
		++pos[i];
		pos2[i] = pos[i] * 2;
	}

	delete [] pos;
	delete [] pos2;

	output = Tensor(_order, expanded);

	return 0;
}

SmartArray Tensor::size()
{
	SmartArray size(_data_mat.dims);

	for (int i = 0; i < _data_mat.dims; ++i)
	{
		size[i] = _data_mat.size[i];
	}

	return size;
}

int Tensor::folding(Tensor &output) const
{
	int dims = _data_mat.dims;
	int *pos = new int[dims];
	int *pos2 = new int[dims];
	const int *range = _data_mat.size;
	int *folded_range = new int[dims];

	for (int i = 0; i < dims; ++i)
	{
		pos[i] = pos2[i] = 0;
		folded_range[i] = (range[i] < 2) ? 1 : (range[i] >> 1);
	}

	Mat folded(dims, folded_range, CV_64FC2, Scalar(0,0));

	int d = dims - 1;
	while(true)
	{
		while (d >= 0 && pos[d] >= folded_range[d])
		{
			pos[d] = 0;
			--d;
			if (d >= 0)
			{
				++pos[d];
				continue;
			}
		}

		if (d < 0)
		{
			break;
		}

		memcpy((void*)pos2, (void*)pos, dims * sizeof(int));

		double complex_sum[2];
		complex_sum[0] = complex_sum[1] = 0.0;
		int d2 = dims - 1;
		while(true)
		{
			while (d2 >= 0 && pos2[d2] >= range[d2])
			{
				pos2[d2] = 0;  //bug here
				--d2;
				if (d2 >= 0)
				{
					pos2[d2] += folded_range[d2];
					continue;
				}
			}

			if (d2 < 0)
			{
				break;
			}

			complex_sum[0] += _data_mat.at<Vec2d>(pos2)[0];
			complex_sum[1] += _data_mat.at<Vec2d>(pos2)[1];

			d2 = dims - 1;
			pos2[d2] += folded_range[d2];
		}

		folded.at<Vec2d>(pos)[0] = complex_sum[0];
		folded.at<Vec2d>(pos)[1] = complex_sum[1];

		d = dims - 1;
		++pos[d];
	}

	delete [] pos;
	delete [] pos2;
	delete [] folded_range;

	output = Tensor(_order, folded);
	return 0;
}

int Tensor::conjugate_reflection(Tensor &output) const
{
	int dims = _data_mat.dims;
	int *pos = new int[dims];
	int *pos2 = new int[dims];
	const int *range = _data_mat.size;

	for (int i = 0; i < dims; i++)
	{
		pos[i] = 0;
		pos2[i] = 0;
	}

	Mat reflected(dims, range, CV_64FC2, Scalar(0,0));
	int d = dims - 1;
	while(true)
	{
		while (d >= 0 && pos[d] >= range[d])
		{
			pos[d] = 0;
			pos2[d] = (range[d] - pos[d]) % range[d];
			--d;
			if (d >= 0)
			{
				++pos[d];
				pos2[d] = (range[d] - pos[d]) % range[d];
				continue;
			}
		}

		if (d < 0)
		{
			break;
		}


		reflected.at<Vec2d>(pos)[0] = _data_mat.at<Vec2d>(pos2)[0];
		reflected.at<Vec2d>(pos)[1] = -_data_mat.at<Vec2d>(pos2)[1];

		d = dims - 1;
		++pos[d];
		pos2[d] = (range[d] - pos[d]) % range[d];
	}

	delete [] pos;
	delete [] pos2;

	output = Tensor(_order, reflected);
	return 0;
}

#include <iostream>
double Tensor::psnr(const Tensor &other)
{
	double msr = 0.0;
	MatConstIterator_<Vec2d> it0 = _data_mat.begin<Vec2d>(), end0 = _data_mat.end<Vec2d>(),
			                 it1 = other._data_mat.begin<Vec2d>();

	for (; it0 != end0; ++it0, ++it1)
	{
		double dif = sqrt((*it0)[0] * (*it0)[0] + (*it0)[1] * (*it0)[1])
				    - sqrt((*it1)[0] * (*it1)[0] + (*it1)[1] * (*it1)[1]);


		msr += dif * dif;
	}

	msr = sqrt(msr / _data_mat.total());

	if (msr < 0.000001)
	{
		return -1;
	}

	std::cout << " -- msr: " << msr << std::endl;

	msr = log(255.0 / msr);
	msr = 20 * msr / log(10);

	std::cout << "PSNR: " << msr << std::endl;

	return msr;
}

int Tensor::tensor_product(const Tensor &other, Tensor &output) const
{

	return 0;
}

double Tensor::l2norm() const
{
    MatConstIterator_<Vec2d> it = _data_mat.begin<Vec2d>(), end = _data_mat.end<Vec2d>();

    double nor = 0.0;
    for (; it != end; ++it)
    {
    	nor += (*it)[0] * (*it)[0];
    	nor += (*it)[1] * (*it)[1];
    }

    return sqrt(nor);
}

int Tensor::scale(complex<double> alpha, complex<double> beta, Tensor &output) const
{
	Mat mat(_data_mat.dims, _data_mat.size, CV_64FC2, Scalar(0, 0));
	MatConstIterator_<Vec2d> it0 = _data_mat.begin<Vec2d>(), end0 = _data_mat.end<Vec2d>();
	MatIterator_<Vec2d> it1 = mat.begin<Vec2d>();

	for (; it0 != end0; ++it0, ++it1)
	{
		complex<double> c((*it0)[0], (*it0)[1]);
		c *= alpha;
		c += beta;

		(*it1)[0] = c.real();
		(*it1)[1] = c.imag();
	}

	output = Tensor(_order, mat);
	return 0;
}

#include <iostream>
using namespace std;

//int Tensor::print(int *idx)
//{
//	cout << "[" << idx[0] << ", "  << idx[1] << "] = " << _data_mat.at<Vec2d>(idx)[0] << ", " << _data_mat.at<Vec2d>(idx)[1] << endl;
//	return 0;
//}

/**** MDSize- *****/
inline SmartArray::SmartArray(int d) : dims(d)
{
	if (dims > 0)
	{
		p = tr1::shared_ptr<int>(new int[dims], array_deleter<int>());
	}
}
inline SmartArray::~SmartArray()
{
}

inline const int& SmartArray::operator[](int i) const { return p.get()[i]; }
inline int& SmartArray::operator[](int i) { return p.get()[i]; }
inline SmartArray::operator const int*() const { return p.get(); }

inline bool SmartArray::operator == (const SmartArray& sz) const
{
	int *pp0 = p.get(), *pp1 = sz.p.get();
    int d = dims, dsz = sz.dims;
    if( d != dsz )
        return false;
    if( d == 2 )
        return pp0[0] == pp1[0] && pp0[1] == pp1[1];

    for( int i = 0; i < d; i++ )
        if( pp0[i] != pp1[i] )
            return false;
    return true;
}

inline bool SmartArray::operator != (const SmartArray& sz) const
{
    return !(*this == sz);
}

/**** -MDSize *****/
