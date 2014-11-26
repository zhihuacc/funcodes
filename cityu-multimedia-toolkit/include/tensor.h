#ifndef _TENSOR_H
#define _TENSOR_H

#include <opencv2/core/core.hpp>
#include <tr1/memory>

using namespace std;
using namespace cv;

//struct video_params
//{
//	int frame_rate;
//	int fourcc;
//	int frame_count;
//	int frame_height;
//	int frame_width;
//};

struct Media_Format;

template<typename T>
struct array_deleter
{
	void operator() (T const *p)
	{
		delete [] p;
	}
};

struct SmartArray
{

	// TODO: Need to take care of copy (smart ptr)
	SmartArray(int dims);
	~SmartArray();
    Size operator()() const;
    const int& operator[](int i) const;
    int& operator[](int i);
    operator const int*() const;
    bool operator == (const SmartArray& sz) const;
    bool operator != (const SmartArray& sz) const;

    tr1::shared_ptr<int> p;
    int dims;
};


class Tensor
{
private:
	int _order; // 1-audio, 2 - image, 3 - video, 4 - ?
	Mat _data_mat;


public:
	Tensor();
	Tensor(int d, const int *size);
	Tensor(int order, const Mat &m);
	~Tensor();

    int dft(Tensor &output) const;
    int idft(Tensor &output) const;
    int idft2(Tensor &output) const;

    int center_shift(Tensor &output) const;
    int icenter_shift(Tensor &output) const;

    int pw_mul(const Tensor &other, Tensor &product) const;
    int pw_add(const Tensor &other, Tensor &sum) const;
    int conv(const Tensor &filter, Tensor &output) const;

    int downsample_by2(Tensor &output) const;
    int upsample_by2(const SmartArray &restored_size, Tensor &output) const;
    int folding(Tensor &output) const;
    int conjugate_reflection(Tensor &output) const;
    int tensor_product(const Tensor &other, Tensor &output) const;
    int scale(complex<double> alpha, complex<double> beta, Tensor &output) const;
    double l2norm() const;

    SmartArray size();

    int set_value(int *idx, double *val);
    double psnr(const Tensor &other);


    friend int load_as_tensor(const string &filename, Tensor &tensor, Media_Format *media_file_fmt);
    friend int save_as_media(const string &filename, const Tensor &tensor, const Media_Format *media_file_fmt);
    friend void print(const Tensor &tensor, int n, int *idx);
    friend void print(const Tensor &tensor, const string &filename);

    friend class Filter_Bank;
};

#endif
