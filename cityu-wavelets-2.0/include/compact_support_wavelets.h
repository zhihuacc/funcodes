#ifndef COMPACT_SUPPORT_WAVELETS_H
#define COMPACT_SUPPORT_WAVELETS_H

#include "math_helpers.h"
#include "mat_toolbox.h"

#define HIGHPASS_FILTER ((unsigned int)0)
#define LOWPASS_FILTER  ((unsigned int)1)
#define LOWPASS_FILTER2 ((unsigned int)2)

template <typename _Tp>
struct OneD_TD_Filter
{
	Mat_<Vec<_Tp, 2> > 	coefs;
	int 				anchor;

	OneD_TD_Filter():anchor(0) {}
};

template <typename _Tp>
struct OneD_TD_FSystem
{

	SmartArray<OneD_TD_Filter<_Tp> >	filters;
	SmartIntArray	ds_folds;
	SmartArray<unsigned int> flags;

	OneD_TD_FSystem(){}
	OneD_TD_FSystem(int n)
	{
		filters.reserve(n);
		ds_folds = SmartIntArray(n, 1);
		flags = SmartIntArray(n, 0);
	}
};

template <typename _Tp>
struct MD_TD_FSystem
{
	SmartArray<OneD_TD_FSystem<_Tp> >	oned_fs_at_dim;
};

template <typename _Tp>
struct ML_MD_TD_FSystem
{
	int nlevels;
	int ndims;
	SmartArray<MD_TD_FSystem<_Tp> > md_fs_at_level;


	ML_MD_TD_FSystem(): nlevels(0), ndims(0) {}
	ML_MD_TD_FSystem(int lvl, int d): nlevels(lvl), ndims(d)
	{
		md_fs_at_level.reserve(nlevels);
		for (int i = 0; i < nlevels; ++i)
		{
			md_fs_at_level[i].oned_fs_at_dim.reserve(ndims);
		}
	}
};

template <typename _Tp>
struct Coefs_Item
{
	Mat_<Vec<_Tp, 2> > coefs;
	bool is_lowpass;
	Coefs_Item():is_lowpass(false) {};
	Coefs_Item(const Mat_<Vec<_Tp, 2> > &c, bool b):coefs(c), is_lowpass(b) {};
};

template<typename _Tp>
struct MC_TD_Coefs_Set
{
public:
	typedef vector<Coefs_Item<_Tp> > type;
};

template<typename _Tp>
struct ML_MC_TD_Coefs_Set
{
public:
	typedef vector<vector<Coefs_Item<_Tp> > > type;
};

//template<typename _Tp>
//struct MC_Coefs_Set
//{
//	typedef vector<Mat_<Vec<_Tp, 2> > > type;
//};

template<typename _Tp>
struct ML_MC_TD_Filter_Norms_Set
{
	typedef vector<vector<_Tp> > type;
};

//template<typename _Tp>
//int conv_by_separable_kernels(const Mat_<Vec<_Tp, 2> > &mat, const SmartArray<OneD_TD_Filter<_Tp> > &skerns, bool is_real, Mat_<Vec<_Tp, 2> > &output)
//{
//	int ndims = mat.dims;
//	if (ndims != skerns.size())
//	{
//		return -1;
//	}
//
//	if (is_real == false)
//	{
//
//		Mat_<Vec<_Tp, 2> > src, dst;
//
//		src = mat.clone();
//		dst = Mat_<Vec<_Tp, 2> >(src.dims, src.size, Vec<_Tp, 2>(0,0));
//		for (int d = 0; d < ndims; ++d)
//		{
//			SmartIntArray start_pos(ndims);
//			SmartIntArray cur_pos(ndims);
//			SmartIntArray this_dim_pos(ndims);
//			SmartIntArray step(ndims, 1);
//			SmartIntArray range(ndims, mat.size);
//			step[d] = range[d];
//			{
//				int src_dims;
//				SmartIntArray src_start_pos;
//				SmartIntArray src_cur_pos;
//				SmartIntArray src_step;
//				SmartIntArray src_end_pos;
//
//				//User-Defined initialization
//				src_dims = ndims;
//				src_start_pos = start_pos;
//				src_cur_pos = cur_pos;
//				src_step = step;
//				src_end_pos = range;
//				//--
//
//				int cur_dim = src_dims - 1;
//				while(true)
//				{
//					while (cur_dim >= 0 && src_cur_pos[cur_dim] >= src_end_pos[cur_dim])
//					{
//						src_cur_pos[cur_dim] = src_start_pos[cur_dim];
//						--cur_dim;
//						if (cur_dim >= 0)
//						{
//							src_cur_pos[cur_dim] += src_step[cur_dim];
//							continue;
//						}
//					}
//					if (cur_dim < 0)
//					{
//						break;
//					}
//
//					//User-Defined actions
//					int N = src.size[d];
//					const Mat_<Vec<_Tp, 2> > &kern = skerns[d].coefs;
//					int M = kern.total();
////					SmartIntArray this_dim_pos = src_cur_pos.clone();
//					cur_pos.copy(this_dim_pos);
//					int anchor = skerns[d].anchor;
//					for (int i = 0; i < N; ++i)
//					{
//						complex<_Tp> sum = 0;
//						for (int j = 0; j < M; ++j)
//						{
//							int idx = i + anchor -j;
//							complex<_Tp> val = 0.0;
//							{
//								if (idx < 0)
//								{
//									idx += range[d];
//								}
//								else if (idx >= range[d])
//								{
//									idx -= range[d];
//								}
//								this_dim_pos[d] = idx;
//								val = src.template at<_Tp>(this_dim_pos);
//							}
//							// Zero-padding
////							{
////								if (idx < 0 || idx >= range[d])
////								{
////									val = 0;
////								}
////								else
////								{
////									this_dim_pos[d] = idx;
////									val = src.template at<complex<_Tp> >(this_dim_pos);
////								}
////							}
//							sum += (kern.template at<complex<_Tp> >(j) * val);
//						}
//						this_dim_pos[d] = i;
//						dst.template at<complex<_Tp> >(this_dim_pos) = sum;
//					}
//					//--
//
//					cur_dim = src_dims - 1;
//					src_cur_pos[cur_dim] += src_step[cur_dim];
//				}
//			}
//
//			Mat_<Vec<_Tp, 2> > tmp = src;
//			src = dst;
//			dst = tmp;
//		}
//		output = src;
//	}
//	else if (is_real == true)
//	{
//		Mat_<Vec<_Tp, 2> > src, dst;
//
//		src = mat.clone();
//		dst = Mat_<Vec<_Tp, 2> >(src.dims, src.size, Vec<_Tp, 2>(0,0));
//		for (int d = 0; d < ndims; ++d)
//		{
//			SmartIntArray start_pos(ndims);
//			SmartIntArray cur_pos(ndims);
//			SmartIntArray this_dim_pos(ndims);
//			SmartIntArray step(ndims, 1);
//			SmartIntArray range(ndims, mat.size);
//			step[d] = range[d];
//			{
//				int src_dims;
//				SmartIntArray src_start_pos;
//				SmartIntArray src_cur_pos;
//				SmartIntArray src_step;
//				SmartIntArray src_end_pos;
//
//				//User-Defined initialization
//				src_dims = ndims;
//				src_start_pos = start_pos;
//				src_cur_pos = cur_pos;
//				src_step = step;
//				src_end_pos = range;
//				//--
//
//				int cur_dim = src_dims - 1;
//				while(true)
//				{
//					while (cur_dim >= 0 && src_cur_pos[cur_dim] >= src_end_pos[cur_dim])
//					{
//						src_cur_pos[cur_dim] = src_start_pos[cur_dim];
//						--cur_dim;
//						if (cur_dim >= 0)
//						{
//							src_cur_pos[cur_dim] += src_step[cur_dim];
//							continue;
//						}
//					}
//					if (cur_dim < 0)
//					{
//						break;
//					}
//
//					//User-Defined actions
//					int N = src.size[d];
//					const Mat_<Vec<_Tp, 2> > &kern = skerns[d].coefs;
//					int M = kern.total();
////					SmartIntArray this_dim_pos = src_cur_pos.clone();
//					cur_pos.copy(this_dim_pos);
//					int anchor = skerns[d].anchor;
//					for (int i = 0; i < N; ++i)
//					{
//						_Tp sum = 0;
//						for (int j = 0; j < M; ++j)
//						{
//							int idx = i + anchor -j;
//							_Tp val = 0.0;
//							{
//								if (idx < 0)
//								{
//									idx += range[d];
//								}
//								else if (idx >= range[d])
//								{
//									idx -= range[d];
//								}
//								this_dim_pos[d] = idx;
//								val = src.template at<_Tp >(this_dim_pos);
//							}
//							// Zero-padding
////							{
////								if (idx < 0 || idx >= range[d])
////								{
////									val = 0;
////								}
////								else
////								{
////									this_dim_pos[d] = idx;
////									val = src.template at<Vec<_Tp, 2> >(this_dim_pos)[0];
////								}
////							}
//							sum += (kern.template at<_Tp >(j << 1) * val);
//						}
//						this_dim_pos[d] = i;
//						dst.template at<Vec<_Tp, 2> >(this_dim_pos) = Vec<_Tp, 2>(sum,0);
//
//					}
//					//--
//
//					cur_dim = src_dims - 1;
//					src_cur_pos[cur_dim] += src_step[cur_dim];
//				}
//			}
//
//			Mat_<Vec<_Tp, 2> >	tmp = src;
//			src = dst;
//			dst = tmp;
//		}
//		output = src;
//	}
//
//	return 0;
//}

template<typename _Tp>
int mat_subadd_equaspaced(Mat_<Vec<_Tp, 2> > &mat, const SmartIntArray &start, const SmartIntArray &steps, const Mat_<Vec<_Tp, 2> > &submat)
{
	int ndims = mat.dims;
//	SmartIntArray start_pos1 = start.clone();
	SmartIntArray cur_pos = start.clone();
	SmartIntArray step = steps.clone();
	SmartIntArray range(ndims, mat.size);
	SmartIntArray submat_pos(ndims);

	step[ndims - 1] = range[ndims - 1];

	{
		int src_dims;
		SmartIntArray src_start_pos;
		SmartIntArray src_cur_pos;
		SmartIntArray src_step;
		SmartIntArray src_end_pos;


		//User-Defined initialization
		src_dims = ndims;
		src_start_pos = start;
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
			for (int i = 0; i < ndims; ++i)
			{
				submat_pos[i] = (cur_pos[i] - start[i]) / steps[i];
			}
			complex<_Tp> *p1 = (complex<_Tp>*)(&(mat(cur_pos)));
			complex<_Tp> *p2 = (complex<_Tp>*)(&(submat(submat_pos)));
			int N = submat.size[ndims - 1];
			int stride = steps[ndims - 1];
			for (int i = 0; i < N; ++i)
			{
				*(p1 + i * stride) += *(p2 + i);
			}
			//--

			cur_dim = src_dims - 1;
			src_cur_pos[cur_dim] += src_step[cur_dim];
		}

	}
	return 0;
}

//template<typename _Tp>
//int conv_by_separable_kernels(const Mat_<Vec<_Tp, 2> > &mat, const SmartArray<OneD_TD_Filter<_Tp> > &skerns, bool is_real, Mat_<Vec<_Tp, 2> > &output)
//{
//	int ndims = mat.dims;
//	if (ndims != skerns.size())
//	{
//		return -1;
//	}
//
//	if (is_real == false)
//	{
//
//		Mat_<Vec<_Tp, 2> > src, dst;
//
//		src = mat.clone();
//		dst = Mat_<Vec<_Tp, 2> >(src.dims, src.size, Vec<_Tp, 2>(0,0));
//		for (int d = 0; d < ndims; ++d)
//		{
//			SmartIntArray start_pos(ndims);
//			SmartIntArray cur_pos(ndims);
//			SmartIntArray this_dim_pos(ndims);
//			SmartIntArray step(ndims, 1);
//			SmartIntArray range(ndims, mat.size);
//			step[d] = range[d];
//			{
//				int src_dims;
//				SmartIntArray src_start_pos;
//				SmartIntArray src_cur_pos;
//				SmartIntArray src_step;
//				SmartIntArray src_end_pos;
//
//				//User-Defined initialization
//				src_dims = ndims;
//				src_start_pos = start_pos;
//				src_cur_pos = cur_pos;
//				src_step = step;
//				src_end_pos = range;
//				//--
//
//				int cur_dim = src_dims - 1;
//				while(true)
//				{
//					while (cur_dim >= 0 && src_cur_pos[cur_dim] >= src_end_pos[cur_dim])
//					{
//						src_cur_pos[cur_dim] = src_start_pos[cur_dim];
//						--cur_dim;
//						if (cur_dim >= 0)
//						{
//							src_cur_pos[cur_dim] += src_step[cur_dim];
//							continue;
//						}
//					}
//					if (cur_dim < 0)
//					{
//						break;
//					}
//
//					//User-Defined actions
//					int N = src.size[d];
//					const Mat_<Vec<_Tp, 2> > &kern = skerns[d].coefs;
//					int M = kern.total();
//					cur_pos.copy(this_dim_pos);
//					int anchor = skerns[d].anchor;
//					complex<_Tp> *pkern = (complex<_Tp>*)(kern.data);
//					complex<_Tp> *psrow = (complex<_Tp>*)(&(src(cur_pos)));
//					complex<_Tp> *pdrow = (complex<_Tp>*)(&(dst(cur_pos)));
//					int stride = src.step[d] / sizeof(complex<_Tp>);
//					int this_dim_end = range[d];
//					for (int i = 0; i < N; ++i)
//					{
//						complex<_Tp> sum = 0;
//						for (int j = 0; j < M; ++j)
//						{
//							int idx = i + anchor -j;
//							{
//								if (idx < 0)
//								{
//									idx += this_dim_end;
//								}
//								else if (idx >= this_dim_end)
//								{
//									idx -= this_dim_end;
//								}
//							}
//
//							sum += pkern[j] * (*(psrow + idx*stride));
//						}
//						*(pdrow + i*stride) = sum;
//					}
//					//--
//
//					cur_dim = src_dims - 1;
//					src_cur_pos[cur_dim] += src_step[cur_dim];
//				}
//			}
//
//			Mat_<Vec<_Tp, 2> > tmp = src;
//			src = dst;
//			dst = tmp;
//		}
//		output = src;
//	}
//	else if (is_real == true)
//	{
//		Mat_<Vec<_Tp, 2> > src, dst;
//
//		src = mat.clone();
//		dst = Mat_<Vec<_Tp, 2> >(src.dims, src.size, Vec<_Tp, 2>(0,0));
//		for (int d = 0; d < ndims; ++d)
//		{
//			SmartIntArray start_pos(ndims);
//			SmartIntArray cur_pos(ndims);
//			SmartIntArray this_dim_pos(ndims);
//			SmartIntArray step(ndims, 1);
//			SmartIntArray range(ndims, mat.size);
//			step[d] = range[d];
//			{
//				int src_dims;
//				SmartIntArray src_start_pos;
//				SmartIntArray src_cur_pos;
//				SmartIntArray src_step;
//				SmartIntArray src_end_pos;
//
//				//User-Defined initialization
//				src_dims = ndims;
//				src_start_pos = start_pos;
//				src_cur_pos = cur_pos;
//				src_step = step;
//				src_end_pos = range;
//				//--
//
//				int cur_dim = src_dims - 1;
//				while(true)
//				{
//					while (cur_dim >= 0 && src_cur_pos[cur_dim] >= src_end_pos[cur_dim])
//					{
//						src_cur_pos[cur_dim] = src_start_pos[cur_dim];
//						--cur_dim;
//						if (cur_dim >= 0)
//						{
//							src_cur_pos[cur_dim] += src_step[cur_dim];
//							continue;
//						}
//					}
//					if (cur_dim < 0)
//					{
//						break;
//					}
//
//					//User-Defined actions
//					int N = src.size[d];
//					const Mat_<Vec<_Tp, 2> > &kern = skerns[d].coefs;
//					int M = kern.total();
//					cur_pos.copy(this_dim_pos);
//					int anchor = skerns[d].anchor;
//					_Tp *pkern = (_Tp*)(kern.data);
//					_Tp *psrow = (_Tp*)(&(src(cur_pos)));
//					_Tp *pdrow = (_Tp*)(&(dst(cur_pos)));
//					int stride = src.step[d] / sizeof(_Tp);
//					int this_dim_end = range[d];
//					for (int i = 0; i < N; ++i)
//					{
//						_Tp sum = 0;
//						for (int j = 0; j < M; ++j)
//						{
//							int idx = i + anchor -j;
//							{
//								if (idx < 0)
//								{
//									idx += this_dim_end;
//								}
//								else if (idx >= this_dim_end)
//								{
//									idx -= this_dim_end;
//								}
//							}
//							sum += (pkern[j<<1] * psrow[idx*stride]);
//						}
//						pdrow[i*stride] = sum;
//
//					}
//					//--
//
//					cur_dim = src_dims - 1;
//					src_cur_pos[cur_dim] += src_step[cur_dim];
//				}
//			}
//
//			Mat_<Vec<_Tp, 2> >	tmp = src;
//			src = dst;
//			dst = tmp;
//		}
//		output = src;
//	}
//
//	return 0;
//}

template<typename _Tp>
int conv_and_ds_in_one_direction(const Mat_<Vec<_Tp, 2> > &mat, int dim, const OneD_TD_Filter<_Tp> &kern, int ds_step, bool is_real, Mat_<Vec<_Tp, 2> > &output)
{
	int ndims = mat.dims;

	if (is_real == false)
	{
		output = Mat_<Vec<_Tp, 2> >(ndims, mat.size, Vec<_Tp, 2>(0,0));
		SmartIntArray start_pos(ndims);
		SmartIntArray cur_pos(ndims);
		SmartIntArray this_dim_pos(ndims);
		SmartIntArray step(ndims, 1);
		SmartIntArray range(ndims, mat.size);
		step[dim] = range[dim];
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
				int N = mat.size[dim];
				const Mat_<Vec<_Tp, 2> > &kern_coef = kern.coefs;
				int M = kern_coef.total();
				cur_pos.copy(this_dim_pos);
				int anchor = kern.anchor;
				complex<_Tp> *pkern = (complex<_Tp>*)(kern_coef.data);
				complex<_Tp> *psrow = (complex<_Tp>*)(&(mat(cur_pos)));
				complex<_Tp> *pdrow = (complex<_Tp> *)(&(output(cur_pos)));
				int stride = mat.step[dim] / sizeof(complex<_Tp>);
				int this_dim_end = range[dim];
				for (int i = 0; i < M; i += ds_step)
				{
					complex<_Tp> sum = 0;
					for (int j = 0; j < M; ++j)
					{
						int idx = i + anchor - j;
						if (idx < 0)
						{
							idx += this_dim_end;
						}
						sum += (pkern[j] * psrow[idx*stride]);
					}
					pdrow[i*stride] = sum;

				}
				int N0 = N - M;
				for (int i = M; i < N0; i += ds_step)
				{
					complex<_Tp> sum = 0;
					for (int j = 0; j < M; ++j)
					{
						int idx = i + anchor - j;
						sum += (pkern[j] * psrow[idx*stride]);
					}
					pdrow[i*stride] = sum;

				}
				for (int i = N0; i < N; i += ds_step)
				{
					complex<_Tp> sum = 0;
					for (int j = 0; j < M; ++j)
					{
						int idx = i + anchor - j;
						if (idx >= this_dim_end)
						{
							idx -= this_dim_end;
						}
						sum += (pkern[j] * psrow[idx*stride]);
					}
					pdrow[i*stride] = sum;

				}
				//--

				cur_dim = src_dims - 1;
				src_cur_pos[cur_dim] += src_step[cur_dim];
			}
		}
	}
	else if (is_real == true)
	{
		output = Mat_<Vec<_Tp, 2> >(ndims, mat.size, Vec<_Tp, 2>(0,0));
		SmartIntArray start_pos(ndims);
		SmartIntArray cur_pos(ndims);
		SmartIntArray this_dim_pos(ndims);
		SmartIntArray step(ndims, 1);
		SmartIntArray range(ndims, mat.size);
		step[dim] = range[dim];
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
				int N = mat.size[dim];
				const Mat_<Vec<_Tp, 2> > &kern_coef = kern.coefs;
				int M = kern_coef.total();
				cur_pos.copy(this_dim_pos);
				int anchor = kern.anchor;
				_Tp *pkern = (_Tp*)(kern_coef.data);
				_Tp *psrow = (_Tp*)(&(mat(cur_pos)));
				_Tp *pdrow = (_Tp*)(&(output(cur_pos)));
				int stride = mat.step[dim] / sizeof(_Tp);
				int this_dim_end = range[dim];
				for (int i = 0; i < M; i += ds_step)
				{
					_Tp sum = 0;
					for (int j = 0; j < M; ++j)
					{
						int idx = i + anchor - j;
						if (idx < 0)
						{
							idx += this_dim_end;
						}
						sum += (pkern[j<<1] * psrow[idx*stride]);
					}
					pdrow[i*stride] = sum;

				}
				int N0 = N - M;
				for (int i = M; i < N0; i += ds_step)
				{
					_Tp sum = 0;
					for (int j = 0; j < M; ++j)
					{
						int idx = i + anchor - j;
						sum += (pkern[j<<1] * psrow[idx*stride]);
					}
					pdrow[i*stride] = sum;

				}
				for (int i = N0; i < N; i += ds_step)
				{
					_Tp sum = 0;
					for (int j = 0; j < M; ++j)
					{
						int idx = i + anchor - j;
						if (idx >= this_dim_end)
						{
							idx -= this_dim_end;
						}
						sum += (pkern[j<<1] * psrow[idx*stride]);
					}
					pdrow[i*stride] = sum;

				}
				//--

				cur_dim = src_dims - 1;
				src_cur_pos[cur_dim] += src_step[cur_dim];
			}
		}
	}

	return 0;
}

template<typename _Tp>
int conv_by_separable_kernels2(const Mat_<Vec<_Tp, 2> > &mat, const SmartArray<OneD_TD_Filter<_Tp> > &skerns, bool is_real, Mat_<Vec<_Tp, 2> > &output)
{
	int ndims = mat.dims;
	if (ndims != skerns.size())
	{
		return -1;
	}

	if (is_real == false)
	{

		Mat_<Vec<_Tp, 2> > src, dst;

		src = mat.clone();
		dst = Mat_<Vec<_Tp, 2> >(src.dims, src.size, Vec<_Tp, 2>(0,0));
		for (int d = 0; d < ndims; ++d)
		{
			SmartIntArray start_pos(ndims);
			SmartIntArray cur_pos(ndims);
			SmartIntArray this_dim_pos(ndims);
			SmartIntArray step(ndims, 1);
			SmartIntArray range(ndims, mat.size);
			step[d] = range[d];
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
					int N = src.size[d];
					const Mat_<Vec<_Tp, 2> > &kern = skerns[d].coefs;
					int M = kern.total();
					cur_pos.copy(this_dim_pos);
					int anchor = skerns[d].anchor;
					complex<_Tp> *pkern = (complex<_Tp>*)(kern.data);
					complex<_Tp> *psrow = (complex<_Tp>*)(&(src(cur_pos)));
					complex<_Tp> *pdrow = (complex<_Tp>*)(&(dst(cur_pos)));
					int stride = src.step[d] / sizeof(complex<_Tp>);
					int this_dim_end = range[d];
					for (int i = 0; i < M; ++i)
					{
						complex<_Tp> sum = 0;
						for (int j = 0; j < M; ++j)
						{
							int idx = i + anchor - j;
							if (idx < 0)
							{
								idx += this_dim_end;
							}
							sum += (pkern[j] * psrow[idx*stride]);
						}
						pdrow[i*stride] = sum;

					}
					int N0 = N - M;
					for (int i = M; i < N0; ++i)
					{
						complex<_Tp> sum = 0;
						for (int j = 0; j < M; ++j)
						{
							int idx = i + anchor - j;
							sum += (pkern[j] * psrow[idx*stride]);
						}
						pdrow[i*stride] = sum;

					}
					for (int i = N0; i < N; ++i)
					{
						complex<_Tp> sum = 0;
						for (int j = 0; j < M; ++j)
						{
							int idx = i + anchor - j;
							if (idx >= this_dim_end)
							{
								idx -= this_dim_end;
							}
							sum += (pkern[j] * psrow[idx*stride]);
						}
						pdrow[i*stride] = sum;

					}
					//--

					cur_dim = src_dims - 1;
					src_cur_pos[cur_dim] += src_step[cur_dim];
				}
			}

			Mat_<Vec<_Tp, 2> > tmp = src;
			src = dst;
			dst = tmp;
		}
		output = src;
	}
	else if (is_real == true)
	{
		Mat_<Vec<_Tp, 2> > src, dst;

		src = mat.clone();
		dst = Mat_<Vec<_Tp, 2> >(src.dims, src.size, Vec<_Tp, 2>(0,0));
		for (int d = 0; d < ndims; ++d)
		{
			SmartIntArray start_pos(ndims);
			SmartIntArray cur_pos(ndims);
			SmartIntArray this_dim_pos(ndims);
			SmartIntArray step(ndims, 1);
			SmartIntArray range(ndims, mat.size);
			step[d] = range[d];
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
					int N = src.size[d];
					const Mat_<Vec<_Tp, 2> > &kern = skerns[d].coefs;
					int M = kern.total();
					cur_pos.copy(this_dim_pos);
					int anchor = skerns[d].anchor;
					_Tp *pkern = (_Tp*)(kern.data);
					_Tp *psrow = (_Tp*)(&(src(cur_pos)));
					_Tp *pdrow = (_Tp*)(&(dst(cur_pos)));
					int stride = src.step[d] / sizeof(_Tp);
					int this_dim_end = range[d];
					for (int i = 0; i < M; ++i)
					{
						_Tp sum = 0;
						for (int j = 0; j < M; ++j)
						{
							int idx = i + anchor - j;
							if (idx < 0)
							{
								idx += this_dim_end;
							}
							sum += (pkern[j<<1] * psrow[idx*stride]);
						}
						pdrow[i*stride] = sum;

					}
					int N0 = N - M;
					for (int i = M; i < N0; ++i)
					{
						_Tp sum = 0;
						for (int j = 0; j < M; ++j)
						{
							int idx = i + anchor - j;
							sum += (pkern[j<<1] * psrow[idx*stride]);
						}
						pdrow[i*stride] = sum;

					}
					for (int i = N0; i < N; ++i)
					{
						_Tp sum = 0;
						for (int j = 0; j < M; ++j)
						{
							int idx = i + anchor - j;
							if (idx >= this_dim_end)
							{
								idx -= this_dim_end;
							}
							sum += (pkern[j<<1] * psrow[idx*stride]);
						}
						pdrow[i*stride] = sum;

					}
					//--

					cur_dim = src_dims - 1;
					src_cur_pos[cur_dim] += src_step[cur_dim];
				}
			}

			Mat_<Vec<_Tp, 2> >	tmp = src;
			src = dst;
			dst = tmp;
		}
		output = src;
	}

	return 0;
}



//template<typename _Tp>
//int downsample(const Mat_<Vec<_Tp, 2> > &mat, const SmartIntArray &ds_steps, Mat_<Vec<_Tp, 2> > &output)
//{
//	int ndims = mat.dims;
//	SmartIntArray start_pos(ndims);
//	SmartIntArray cur_pos(ndims);
//	SmartIntArray dst_pos(ndims);
////	SmartIntArray step = ds_steps.clone();
//	SmartIntArray range(ndims, mat.size);
//
//	SmartIntArray decimated_size(ndims);
//	for (int i = 0; i < ndims; ++i)
//	{
//		decimated_size[i] = range[i] < 2 ? 1 : range[i] / ds_steps[i];
//	}
//	Mat_<Vec<_Tp, 2> > tmp(ndims, decimated_size, Vec<_Tp, 2>(0,0));
//	{
//		int src_dims;
//		SmartIntArray src_start_pos;
//		SmartIntArray src_cur_pos;
//		SmartIntArray src_step;
//		SmartIntArray src_end_pos;
//
//		//User-Defined initialization
//		src_dims = ndims;
//		src_start_pos = start_pos;
//		src_cur_pos = cur_pos;
//		src_step = ds_steps;
//		src_end_pos = range;
//		//--
//
//		int cur_dim = src_dims - 1;
//		while(true)
//		{
//			while (cur_dim >= 0 && src_cur_pos[cur_dim] >= src_end_pos[cur_dim])
//			{
//				src_cur_pos[cur_dim] = src_start_pos[cur_dim];
//				--cur_dim;
//				if (cur_dim >= 0)
//				{
//					src_cur_pos[cur_dim] += src_step[cur_dim];
//					continue;
//				}
//			}
//			if (cur_dim < 0)
//			{
//				break;
//			}
//
//			//User-Defined actions
//			for (; cur_dim < src_dims; ++cur_dim)
//			{
//				dst_pos[cur_dim] = cur_pos[cur_dim] / ds_steps[cur_dim];
//			}
//			tmp(dst_pos) = mat(cur_pos);
//			//--
//
//			cur_dim = src_dims - 1;
//			src_cur_pos[cur_dim] += src_step[cur_dim];
//		}
//	}
//
//	output = tmp;
//
//	return 0;
//}

template<typename _Tp>
int downsample2(const Mat_<Vec<_Tp, 2> > &mat, const SmartIntArray &ds_steps, Mat_<Vec<_Tp, 2> > &output)
{
	int ndims = mat.dims;
	SmartIntArray start_pos(ndims);
	SmartIntArray cur_pos(ndims);
	SmartIntArray dst_pos(ndims);
	SmartIntArray step = ds_steps.clone();
	SmartIntArray range(ndims, mat.size);

	step[ndims - 1] = range[ndims - 1];
	SmartIntArray decimated_size(ndims);
	for (int i = 0; i < ndims; ++i)
	{
		decimated_size[i] = range[i] < 2 ? 1 : range[i] / ds_steps[i];
	}
	Mat_<Vec<_Tp, 2> > tmp(ndims, decimated_size, Vec<_Tp, 2>(0,0));
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
				dst_pos[cur_dim] = cur_pos[cur_dim] / ds_steps[cur_dim];
			}
			complex<_Tp> *psrow = (complex<_Tp> *)(&mat(cur_pos));
			complex<_Tp> *pdrow = (complex<_Tp> *)(&tmp(dst_pos));
			int n = decimated_size[ndims - 1];
			int last_step = ds_steps[ndims - 1];
			for (int i = 0; i < n; ++i)
			{
				*(pdrow + i) = *(psrow + i * last_step);
			}
			//--

			cur_dim = src_dims - 1;
			src_cur_pos[cur_dim] += src_step[cur_dim];
		}
	}

	output = tmp;

	return 0;
}

template<typename _Tp>
int upsample(const Mat_<Vec<_Tp, 2> > &mat, const SmartIntArray &ds_steps, Mat_<Vec<_Tp, 2> > &output)
{
	int ndims = mat.dims;
	SmartIntArray start_pos(ndims);
	SmartIntArray cur_pos(ndims);
	SmartIntArray dst_pos(ndims);
	SmartIntArray step(ndims,1);
	SmartIntArray range(ndims, mat.size);

	SmartIntArray upsampled_size(ndims);
	for (int i = 0; i < ndims; ++i)
	{
		upsampled_size[i] = range[i] * ds_steps[i];
	}
	Mat_<Vec<_Tp, 2> > tmp(ndims, upsampled_size, Vec<_Tp, 2>(0,0));
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
				dst_pos[cur_dim] = cur_pos[cur_dim] * ds_steps[cur_dim];
			}
			tmp(dst_pos) = mat(cur_pos);
			//--

			cur_dim = src_dims - 1;
			src_cur_pos[cur_dim] += src_step[cur_dim];
		}
	}

	output = tmp;

	return 0;
}

template<typename _Tp>
int conv_by_separable_kernels_and_ds(const Mat_<Vec<_Tp, 2> > &mat, const SmartArray<OneD_TD_Filter<_Tp> > &skerns, const SmartIntArray &ds_step, bool is_real, Mat_<Vec<_Tp, 2> > &output)
{
	int ndims = mat.dims;
	if (ndims != skerns.size())
	{
		return -1;
	}

	if (is_real == false)
	{

		Mat_<Vec<_Tp, 2> > src, dst;

		src = mat.clone();
		dst = Mat_<Vec<_Tp, 2> >(src.dims, src.size, Vec<_Tp, 2>(0,0));

		for (int d = 0; d < ndims; ++d)
		{
			SmartIntArray start_pos(ndims);
			SmartIntArray cur_pos(ndims);
			SmartIntArray this_dim_pos(ndims);
			SmartIntArray step(ndims, 1);
			SmartIntArray range(ndims, mat.size);
			for (int d0 = 0; d0 < d; d0++)
			{
				step[d0] = ds_step[d0];
			}
			step[d] = range[d];
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
					int N = src.size[d];
					const Mat_<Vec<_Tp, 2> > &kern = skerns[d].coefs;
					int M = kern.total();
					cur_pos.copy(this_dim_pos);
					int anchor = skerns[d].anchor;
					complex<_Tp> *pkern = (complex<_Tp>*)(kern.data);
					complex<_Tp> *psrow = (complex<_Tp>*)(&(src(cur_pos)));
					complex<_Tp> *pdrow = (complex<_Tp>*)(&(dst(cur_pos)));
					int stride = src.step[d] / sizeof(complex<_Tp>);
					int this_dim_end = range[d];
					int ds_step_d = ds_step[d];

					for (int i = 0; i < M; i += ds_step_d)
					{
						complex<_Tp> sum = 0;
						for (int j = 0; j < M; ++j)
						{
							int idx = i + anchor - j;
							if (idx < 0)
							{
								idx += this_dim_end;
							}
							sum += (pkern[j] * psrow[idx*stride]);
						}
						pdrow[i*stride] = sum;

					}
					int N0 = N - M;
					for (int i = M; i < N0; i += ds_step_d)
					{
						complex<_Tp> sum = 0;
						for (int j = 0; j < M; ++j)
						{
							int idx = i + anchor - j;
							sum += (pkern[j] * psrow[idx*stride]);
						}
						pdrow[i*stride] = sum;

					}
					for (int i = N0; i < N; i += ds_step_d)
					{
						complex<_Tp> sum = 0;
						for (int j = 0; j < M; ++j)
						{
							int idx = i + anchor - j;
							if (idx >= this_dim_end)
							{
								idx -= this_dim_end;
							}
							sum += (pkern[j] * psrow[idx*stride]);
						}
						pdrow[i*stride] = sum;

					}
					//--

					cur_dim = src_dims - 1;
					src_cur_pos[cur_dim] += src_step[cur_dim];
				}
			}

			Mat_<Vec<_Tp, 2> > tmp = src;
			src = dst;
			dst = tmp;
		}

		downsample2<_Tp>(src, ds_step, output);
	}
	else if (is_real == true)
	{
//		clock_t t0, t1;
//		t0 = tic();
		Mat_<Vec<_Tp, 2> > src, dst;

		src = mat.clone();
		dst = Mat_<Vec<_Tp, 2> >(src.dims, src.size, Vec<_Tp, 2>(0,0));
		for (int d = 0; d < ndims; ++d)
		{
			SmartIntArray start_pos(ndims);
			SmartIntArray cur_pos(ndims);
			SmartIntArray this_dim_pos(ndims);
			SmartIntArray step(ndims, 1);
			SmartIntArray range(ndims, mat.size);
			for (int d0 = 0; d0 < d; d0++)
			{
				step[d0] = ds_step[d0];
			}
			step[d] = range[d];
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
					int N = src.size[d];
					const Mat_<Vec<_Tp, 2> > &kern = skerns[d].coefs;
					int M = kern.total();
					cur_pos.copy(this_dim_pos);
					int anchor = skerns[d].anchor;
					_Tp *pkern = (_Tp*)(kern.data);
					_Tp *psrow = (_Tp*)(&(src(cur_pos)));
					_Tp *pdrow = (_Tp*)(&(dst(cur_pos)));
					int stride = src.step[d] / sizeof(_Tp);
					int this_dim_end = range[d];
					int ds_step_d = ds_step[d];

					for (int i = 0; i < M; i += ds_step_d)
					{
						_Tp sum = 0;
						for (int j = 0; j < M; ++j)
						{
							int idx = i + anchor - j;
							if (idx < 0)
							{
								idx += this_dim_end;
							}
							sum += (pkern[j<<1] * psrow[idx*stride]);
						}
						pdrow[i*stride] = sum;

					}
					int N0 = N - M;
					for (int i = M; i < N0; i += ds_step_d)
					{
						_Tp sum = 0;
						for (int j = 0; j < M; ++j)
						{
							int idx = i + anchor - j;
							sum += (pkern[j<<1] * psrow[idx*stride]);
						}
						pdrow[i*stride] = sum;

					}
					for (int i = N0; i < N; i += ds_step_d)
					{
						_Tp sum = 0;
						for (int j = 0; j < M; ++j)
						{
							int idx = i + anchor - j;
							if (idx >= this_dim_end)
							{
								idx -= this_dim_end;
							}
							sum += (pkern[j<<1] * psrow[idx*stride]);
						}
						pdrow[i*stride] = sum;

					}
					//--

					cur_dim = src_dims - 1;
					src_cur_pos[cur_dim] += src_step[cur_dim];
				}
			}

			Mat_<Vec<_Tp, 2> > tmp = src;
			src = dst;
			dst = tmp;
		}

//		t1 = tic();
//		string msg = show_elapse(t1 - t0);
//		cout << "conv in ds: " << endl << msg << endl;

		downsample2<_Tp>(src, ds_step, output);

//		t0 = tic();
//		msg = show_elapse(t0 - t1);
//		cout << "ds in ds: " << endl << msg << endl;
	}

	return 0;
}

template<typename _Tp>
int conv_by_separable_kernels_and_us(const Mat_<Vec<_Tp, 2> > &mat, const SmartArray<OneD_TD_Filter<_Tp> > &skerns, const SmartIntArray &us_step, bool is_real, Mat_<Vec<_Tp, 2> > &output)
{
	int ndims = mat.dims;
	if (output.empty())
	{
		return -1;
	}
	SmartIntArray start_pos(ndims);
	SmartIntArray cur_pos(ndims);
	SmartIntArray step(ndims, 1);
	SmartIntArray range(ndims, us_step);
	SmartIntArray origin_range(ndims, mat.size);
	SmartIntArray upsampled_range(ndims);
	for (int i = 0; i < ndims; ++i)
	{
		upsampled_range[i] = origin_range[i] * us_step[i];
	}
//	Mat_<Vec<_Tp, 2> > upsampled_mat(ndims, upsampled_range, Vec<_Tp, 2>(0,0));
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
			SmartArray<OneD_TD_Filter<_Tp> > ds_skerns(ndims);
			SmartArray<SmartIntArray> chosen_index(ndims);
			for (int i = 0; i < ndims; ++i)
			{
				int offset = cur_pos[i] + skerns[i].anchor;
				for (int j = offset % us_step[i]; j < (int)skerns[i].coefs.total(); j += us_step[i])
				{
					ds_skerns[i].coefs.push_back(skerns[i].coefs(j));
				}
				ds_skerns[i].anchor = offset / us_step[i];

				chosen_index[i] = SmartIntArray(origin_range[i]);
				for (int j = 0; j < mat.size[i]; j += 1)
				{
					chosen_index[i][j] = cur_pos[i] + j * us_step[i];
				}
			}
			Mat_<Vec<_Tp, 2> >	conv;
//			clock_t t0, t1;
//			t0 = tic();

			conv_by_separable_kernels2<_Tp>(mat, ds_skerns, is_real, conv);
//			t1 = tic();
//			cout << "conv in us: " << endl << show_elapse(t1 - t0) << endl;


//			mat_subadd<_Tp>(output, chosen_index, conv);
			mat_subadd_equaspaced<_Tp>(output, cur_pos, us_step, conv);
//			t0 = tic();
//			cout << "subadd in us: " << endl << show_elapse(t0 - t1) << endl;

			//--

			cur_dim = src_dims - 1;
			src_cur_pos[cur_dim] += src_step[cur_dim];
		}
	}

//	output = upsampled_mat;

	return 0;
}

template<typename _Tp>
int conv_by_separable_kernels_and_us2(const Mat_<Vec<_Tp, 2> > &mat, const SmartArray<OneD_TD_Filter<_Tp> > &skerns, const SmartIntArray &us_step, bool is_real, Mat_<Vec<_Tp, 2> > &output)
{
	int ndims = mat.dims;
	if (output.empty())
	{
		return -1;
	}
	SmartIntArray start_pos(ndims);
	SmartIntArray cur_pos(ndims);
	SmartIntArray step(ndims, 1);
	SmartIntArray range(ndims, us_step);
	SmartIntArray origin_range(ndims, mat.size);
	SmartIntArray upsampled_range(ndims);
	for (int i = 0; i < ndims; ++i)
	{
		upsampled_range[i] = origin_range[i] * us_step[i];
	}

	SmartArray<SmartArray<OneD_TD_Filter<_Tp> > > ds_kerns_at_dim(ndims);
//	SmartArray<SmartArray<SmartIntArray> > filling_index_at_dim[ndims];
	for (int i = 0; i < ndims; ++i)
	{
		ds_kerns_at_dim[i] = SmartArray<OneD_TD_Filter<_Tp> >(us_step[i]);
//		filling_index_at_dim[i] = SmartArray<SmartIntArray>(us_step[i]);
		for (int j = 0; j < us_step[i]; ++j)
		{
			int offset = j + skerns[i].anchor;
			for (int k = offset % us_step[i]; k < (int)skerns[i].coefs.total(); k += us_step[i])
			{
				ds_kerns_at_dim[i][j].coefs.push_back(skerns[i].coefs(k));
			}
			ds_kerns_at_dim[i][j].anchor = offset / us_step[i];

//			filling_index_at_dim[i][j] = SmartIntArray(origin_range[i]);
//			for (int k = 0; k < mat.size[i]; k += 1)
//			{
//				filling_index_at_dim[i][j][k] = j + k * us_step[i];
//			}
		}
	}

	SmartArray<Mat_<Vec<_Tp, 2> > > scanned(ndims);
	SmartIntArray last_pos(ndims);
	last_pos[0] = -1;
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
//			SmartArray<OneD_TD_Filter<_Tp> > ds_skerns(ndims);
//			SmartArray<SmartIntArray> chosen_index(ndims);
//			for (int i = 0; i < ndims; ++i)
//			{
//				int offset = cur_pos[i] + skerns[i].anchor;
//				for (int j = offset % us_step[i]; j < (int)skerns[i].coefs.total(); j += us_step[i])
//				{
//					ds_skerns[i].coefs.push_back(skerns[i].coefs(j));
//				}
//				ds_skerns[i].anchor = offset / us_step[i];
//
//				chosen_index[i] = SmartIntArray(origin_range[i]);
//				for (int j = 0; j < mat.size[i]; j += 1)
//				{
//					chosen_index[i][j] = cur_pos[i] + j * us_step[i];
//				}
//			}

			SmartArray<OneD_TD_Filter<_Tp> > chosen_filter_at_dim(ndims);
			for (int i = 0; i < ndims; ++i)
			{
				chosen_filter_at_dim[i] = ds_kerns_at_dim[i][cur_pos[i]];
			}

			int branch = 0;
			for (;cur_pos[branch] == last_pos[branch]; ++branch);

			for (int i = branch; i < ndims; ++i)
			{
				Mat_<Vec<_Tp, 2> > last;
				if (i == 0)
				{
					last = mat;
				}
				else
				{
					last = scanned[i - 1];
				}
				conv_and_ds_in_one_direction<_Tp>(last, i, chosen_filter_at_dim[i], 1, is_real, scanned[i]);
			}

			Mat_<Vec<_Tp, 2> >	conv;
//			clock_t t0, t1;
//			t0 = tic();

//			conv_by_separable_kernels2<_Tp>(mat, ds_skerns, is_real, conv);
//			t1 = tic();
//			cout << "conv in us: " << endl << show_elapse(t1 - t0) << endl;


//			mat_subadd<_Tp>(output, chosen_index, conv);
			mat_subadd_equaspaced<_Tp>(output, cur_pos, us_step, scanned[ndims - 1]);
//			t0 = tic();
//			cout << "subadd in us: " << endl << show_elapse(t0 - t1) << endl;

			//--

			cur_dim = src_dims - 1;
			src_cur_pos[cur_dim] += src_step[cur_dim];
		}
	}

//	output = upsampled_mat;

	return 0;
}

template<typename _Tp>
int decompose_in_time_domain(const ML_MD_TD_FSystem<_Tp> &ml_md_fs, const Mat_<Vec<_Tp, 2> > &input, bool is_real, typename ML_MC_TD_Filter_Norms_Set<_Tp>::type &norms_set, typename ML_MC_TD_Coefs_Set<_Tp>::type &coefs_set)
{

	if (ml_md_fs.nlevels < 1
		|| input.dims != ml_md_fs.ndims)
	{
		return -1;
	}

	int nlevels = ml_md_fs.nlevels;
	int ndims = ml_md_fs.ndims;

	coefs_set.reserve(nlevels);
	coefs_set.resize(nlevels);
	norms_set.reserve(nlevels);
	norms_set.resize(nlevels);


	for (int cur_lvl = 0; cur_lvl < nlevels; cur_lvl++)
	{
		// This mat is referred to as low-pass channel output last level.
		Mat_<Vec<_Tp, 2> > last_approx;
		if (cur_lvl == 0)
		{
			last_approx = input;
		}
		else
		{
			last_approx = coefs_set[cur_lvl - 1][coefs_set[cur_lvl - 1].size() - 1].coefs;
		}

		// This is the number of filters at each dim at this level.
		SmartIntArray filter_numbers_at_dim(ndims);

		const MD_TD_FSystem<_Tp> &this_level_md_fs = ml_md_fs.md_fs_at_level[cur_lvl];
//			double lowpass_ds_prod = 1.0;
		for (int cur_dim = 0; cur_dim < ndims; ++cur_dim)		// Every dim in this level
		{
			// The last filter is not counted.
			filter_numbers_at_dim[cur_dim] = this_level_md_fs.oned_fs_at_dim[cur_dim].filters.size() - 1;
		}


		int N = filter_numbers_at_dim[ndims - 1];
		SmartIntArray steps(ndims, 1);
		steps[ndims - 1] = 1;
		for (int i = ndims - 2; i >= 0; --i)
		{
			steps[i] = filter_numbers_at_dim[i + 1] * steps[i + 1];
			N *= filter_numbers_at_dim[i];
		}

		for (int n = 0; n < N; ++n)
		{
			// This method is stupid, but clear.
			// Because N would not be too large, typically hundreds, this method would not cost much.
			SmartIntArray cur_pos(ndims);
			for (int i = 0, rem = n; i < ndims; ++i)
			{
				cur_pos[i] = rem / steps[i];
				rem -= (cur_pos[i] * steps[i]);
			}

			//User-Defined actions
			SmartArray<OneD_TD_Filter<_Tp> > chosen_filter_at_dim(ndims);
			SmartIntArray ds_fold_at_dim(ndims);
			bool is_lowpass = true;
			//Should start at 0, since we need to check lowpass for ALL filters
			for (int i = 0; i < ndims; ++i)
			{
				const OneD_TD_Filter<_Tp> &this_filter = this_level_md_fs.oned_fs_at_dim[i].filters[cur_pos[i]];
				chosen_filter_at_dim[i] = this_filter;

				ds_fold_at_dim[i] = this_level_md_fs.oned_fs_at_dim[i].ds_folds[cur_pos[i]];
				is_lowpass = is_lowpass && (this_level_md_fs.oned_fs_at_dim[i].flags[cur_pos[i]] == LOWPASS_FILTER);
			}

			if (is_lowpass)
			{
				continue;
			}

			Coefs_Item<_Tp> item;
			item.is_lowpass = false;
			conv_by_separable_kernels_and_ds<_Tp>(last_approx, chosen_filter_at_dim, ds_fold_at_dim, is_real, item.coefs);
			coefs_set[cur_lvl].push_back(item);

			//-- User Action
		}

		{   // lowpass-channel
		SmartArray<OneD_TD_Filter<_Tp> > lowpass_filter_at_dim(ndims);
		SmartIntArray lowpass_ds_fold_at_dim(ndims);
		for (int i = 0; i < ndims; ++i)
		{
			int index = this_level_md_fs.oned_fs_at_dim[i].filters.size() - 1;
			lowpass_filter_at_dim[i] = this_level_md_fs.oned_fs_at_dim[i].filters[index];
			lowpass_ds_fold_at_dim[i] = this_level_md_fs.oned_fs_at_dim[i].ds_folds[index];
		}

		Coefs_Item<_Tp> item;
		item.is_lowpass = true;
//				conv_by_separable_kernels<_Tp>(last_approx, chosen_filter_at_dim, true, item.coefs);
//				downsample<_Tp>(item.coefs, ds_fold_at_dim, item.coefs);
		conv_by_separable_kernels_and_ds<_Tp>(last_approx, lowpass_filter_at_dim, lowpass_ds_fold_at_dim, is_real, item.coefs);
		coefs_set[cur_lvl].push_back(item);
		}
	}

	return 0;
}

template<typename _Tp>
int decompose_in_time_domain2(const ML_MD_TD_FSystem<_Tp> &ml_md_fs, const Mat_<Vec<_Tp, 2> > &input, bool is_real, typename ML_MC_TD_Filter_Norms_Set<_Tp>::type &norms_set, typename ML_MC_TD_Coefs_Set<_Tp>::type &coefs_set)
{

	if (ml_md_fs.nlevels < 1
		|| input.dims != ml_md_fs.ndims)
	{
		return -1;
	}

	int nlevels = ml_md_fs.nlevels;
	int ndims = ml_md_fs.ndims;

	coefs_set.reserve(nlevels);
	coefs_set.resize(nlevels);
	norms_set.reserve(nlevels);
	norms_set.resize(nlevels);


	for (int cur_lvl = 0; cur_lvl < nlevels; cur_lvl++)
	{
		// This mat is referred to as low-pass channel output last level.
		Mat_<Vec<_Tp, 2> > last_approx;
		if (cur_lvl == 0)
		{
			last_approx = input;
		}
		else
		{
			last_approx = coefs_set[cur_lvl - 1][coefs_set[cur_lvl - 1].size() - 1].coefs;
		}

		// This is the number of filters at each dim at this level.
		SmartIntArray filter_numbers_at_dim(ndims);

		const MD_TD_FSystem<_Tp> &this_level_md_fs = ml_md_fs.md_fs_at_level[cur_lvl];
		for (int cur_dim = 0; cur_dim < ndims; ++cur_dim)		// Every dim in this level
		{
			// The last filter is not counted.
			filter_numbers_at_dim[cur_dim] = this_level_md_fs.oned_fs_at_dim[cur_dim].filters.size() - 1;
		}


		int N = filter_numbers_at_dim[ndims - 1];
		SmartIntArray steps(ndims, 1);
		steps[ndims - 1] = 1;
		for (int i = ndims - 2; i >= 0; --i)
		{
			steps[i] = filter_numbers_at_dim[i + 1] * steps[i + 1];
			N *= filter_numbers_at_dim[i];
		}

		SmartArray<Mat_<Vec<_Tp, 2> > > scanned(ndims);
		SmartIntArray last_pos(ndims);
		last_pos[0] = -1;
		for (int n = 0; n < N; ++n)
		{
			// This method is stupid, but clear.
			// Because N would not be too large, typically hundreds, this method would not cost much.
			SmartIntArray cur_pos(ndims);
			for (int i = 0, rem = n; i < ndims; ++i)
			{
				cur_pos[i] = rem / steps[i];
				rem -= (cur_pos[i] * steps[i]);
			}

			//User-Defined actions
			SmartArray<OneD_TD_Filter<_Tp> > chosen_filter_at_dim(ndims);
			SmartIntArray ds_fold_at_dim(ndims);
			bool is_lowpass = true;
			//Should start at 0, since we need to check lowpass for ALL filters
			for (int i = 0; i < ndims; ++i)
			{
				const OneD_TD_Filter<_Tp> &this_filter = this_level_md_fs.oned_fs_at_dim[i].filters[cur_pos[i]];
				chosen_filter_at_dim[i] = this_filter;

				ds_fold_at_dim[i] = this_level_md_fs.oned_fs_at_dim[i].ds_folds[cur_pos[i]];
				is_lowpass = is_lowpass && (this_level_md_fs.oned_fs_at_dim[i].flags[cur_pos[i]] == LOWPASS_FILTER);
			}

			if (is_lowpass)
			{
				continue;
			}

			int branch = 0;
			for (;cur_pos[branch] == last_pos[branch]; ++branch);

			for (int i = branch; i < ndims; ++i)
			{
				Mat_<Vec<_Tp, 2> > last;
				if (i == 0)
				{
					last = last_approx;
				}
				else
				{
					last = scanned[i - 1];
				}
				conv_and_ds_in_one_direction<_Tp>(last, i, chosen_filter_at_dim[i], ds_fold_at_dim[i], is_real, scanned[i]);
			}

			Coefs_Item<_Tp> item;
			item.is_lowpass = false;
			downsample2<_Tp>(scanned[ndims - 1], ds_fold_at_dim, item.coefs);
			coefs_set[cur_lvl].push_back(item);

//			Coefs_Item<_Tp> item;
//			item.is_lowpass = false;
//			conv_by_separable_kernels_and_ds<_Tp>(last_approx, chosen_filter_at_dim, ds_fold_at_dim, is_real, item.coefs);
//			coefs_set[cur_lvl].push_back(item);

			cur_pos.copy(last_pos);
			//-- User Action
		}

		{   // lowpass-channel
		SmartArray<OneD_TD_Filter<_Tp> > lowpass_filter_at_dim(ndims);
		SmartIntArray lowpass_ds_fold_at_dim(ndims);
		for (int i = 0; i < ndims; ++i)
		{
			int index = this_level_md_fs.oned_fs_at_dim[i].filters.size() - 1;
			lowpass_filter_at_dim[i] = this_level_md_fs.oned_fs_at_dim[i].filters[index];
			lowpass_ds_fold_at_dim[i] = this_level_md_fs.oned_fs_at_dim[i].ds_folds[index];
		}

		Coefs_Item<_Tp> item;
		item.is_lowpass = true;
		conv_by_separable_kernels_and_ds<_Tp>(last_approx, lowpass_filter_at_dim, lowpass_ds_fold_at_dim, is_real, item.coefs);
		coefs_set[cur_lvl].push_back(item);
		}
	}

	return 0;
}


template <typename _Tp>
int reconstruct_in_time_domain(const ML_MD_TD_FSystem<_Tp> &ml_md_fs, const typename ML_MC_TD_Coefs_Set<_Tp>::type &coefs_set, bool is_real, Mat_<Vec<_Tp, 2> > &rec)
{
	int nlevels = ml_md_fs.nlevels;
	int ndims = ml_md_fs.ndims;
	if (nlevels < 1)
	{
		return -1;
	}

	// Every Level
	Mat_<Vec<_Tp, 2> > upper_lowpass_approx;
	for (int cur_lvl = nlevels - 1; cur_lvl >= 0; --cur_lvl)
	{
		const typename MC_TD_Coefs_Set<_Tp>::type &this_level_coefs_set = coefs_set[cur_lvl];
		const MD_TD_FSystem<_Tp> &this_level_md_fs = ml_md_fs.md_fs_at_level[cur_lvl];
		Mat_<Vec<_Tp, 2> > this_level_lowpass_approx;
		Mat_<Vec<_Tp, 2> > this_level_upsampled_sum;
		if (cur_lvl == nlevels - 1)
		{
			this_level_lowpass_approx = this_level_coefs_set[this_level_coefs_set.size() - 1].coefs;
		}
		else
		{
			this_level_lowpass_approx = upper_lowpass_approx;
		}


		SmartIntArray filter_numbers_at_dim(ndims);
		for (int i = 0; i < ndims; ++i)
		{
			filter_numbers_at_dim[i] = this_level_md_fs.oned_fs_at_dim[i].filters.size() - 1;
		}

		// Find all combinations to do tensor product
		int N = filter_numbers_at_dim[ndims - 1];
		SmartIntArray steps(ndims, 1);
		steps[ndims - 1] = 1;
		for (int i = ndims - 2; i >= 0; --i)
		{
			steps[i] = filter_numbers_at_dim[i + 1] * steps[i + 1];
			N *= filter_numbers_at_dim[i];
		}

		SmartIntArray this_level_upsampled_size(ndims);
		for (int n = 0, highpass_coefs_idx = 0; n < N; ++n)
		{
			SmartIntArray cur_pos(ndims);
			for (int i = 0, rem = n; i < ndims; ++i)
			{
				cur_pos[i] = rem / steps[i];
				rem -= (cur_pos[i] * steps[i]);
			}

			//User-Defined actions
			//A combination is found.
			bool is_lowpass = true;
			SmartArray<OneD_TD_Filter<_Tp> > chosen_filter_at_dim(ndims);
			SmartIntArray ds_fold_at_dim(ndims);
			//Start at 0, since we need to check lowpass for all filters
			for (int i = 0; i < ndims; ++i)
			{
				const OneD_TD_Filter<_Tp> &this_filter = this_level_md_fs.oned_fs_at_dim[i].filters[cur_pos[i]];
				chosen_filter_at_dim[i] = this_filter;
				ds_fold_at_dim[i] = this_level_md_fs.oned_fs_at_dim[i].ds_folds[cur_pos[i]];
//				lowpass_ds_fold_at_dim[i] = this_filter.lowpass_ds_fold;
				is_lowpass = is_lowpass && (this_level_md_fs.oned_fs_at_dim[i].flags[cur_pos[i]] == LOWPASS_FILTER);
			}

			Mat_<Vec<_Tp, 2> > cur_coefs;
			if (is_lowpass)
			{
//				ds_fold_at_dim = lowpass_ds_fold_at_dim;
//				cur_coefs = this_level_lowpass_approx;
				continue;
			}
			else
			{
				cur_coefs = this_level_coefs_set[highpass_coefs_idx++].coefs;
			}

			if (this_level_upsampled_sum.empty())
			{
				for (int i = 0; i < ndims; ++i)
				{
					this_level_upsampled_size[i] = cur_coefs.size[i] * ds_fold_at_dim[i];
				}
				this_level_upsampled_sum = Mat_<Vec<_Tp, 2> >(ndims, this_level_upsampled_size, Vec<_Tp,2>(0,0));
			}


			Mat_<Vec<_Tp, 2> > upsampled_coefs;

//			upsample<_Tp>(cur_coefs, ds_fold_at_dim, upsampled_coefs);
//			conv_by_separable_kernels<_Tp>(upsampled_coefs, chosen_filter_at_dim, true, upsampled_coefs);
			conv_by_separable_kernels_and_us2<_Tp>(cur_coefs, chosen_filter_at_dim, ds_fold_at_dim, is_real, this_level_upsampled_sum);
//			this_level_upsampled_sum += upsampled_coefs;

		}

		{   //lowpass-channel
		SmartArray<OneD_TD_Filter<_Tp> > lowpass_filter_at_dim(ndims);
		SmartIntArray lowpass_ds_fold_at_dim(ndims);
		for (int i = 0; i < ndims; ++i)
		{
			int index = this_level_md_fs.oned_fs_at_dim[i].filters.size() - 1;
			lowpass_filter_at_dim[i] = this_level_md_fs.oned_fs_at_dim[i].filters[index];
			lowpass_ds_fold_at_dim[i] = this_level_md_fs.oned_fs_at_dim[i].ds_folds[index];
		}

		Mat_<Vec<_Tp, 2> > upsampled_coefs;
		//			upsample<_Tp>(cur_coefs, ds_fold_at_dim, upsampled_coefs);
		//			conv_by_separable_kernels<_Tp>(upsampled_coefs, chosen_filter_at_dim, true, upsampled_coefs);
		conv_by_separable_kernels_and_us<_Tp>(this_level_lowpass_approx, lowpass_filter_at_dim, lowpass_ds_fold_at_dim, is_real, this_level_upsampled_sum);
//		this_level_upsampled_sum += upsampled_coefs;
		}

		upper_lowpass_approx = this_level_upsampled_sum;

	}

	rec = upper_lowpass_approx;

	return 0;
}


#endif
