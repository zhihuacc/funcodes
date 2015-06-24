#ifndef WAVELETS_TOOLBOX_H
#define WAVELETS_TOOLBOX_H

#include "math_helpers.h"
#include "mat_toolbox.h"
#include "structs.h"


template<typename _Tp>
int linspace(const complex<_Tp> &left, const complex<_Tp> &right, int n, Mat_<Vec<_Tp, 2> > &samples)
{
	if (n < 1)
	{
		return -1;
	}

//	samples.create(2, (int[]){1,n}, DataType<Vec<_Tp, 2> >::type, Scalar(0,0));
//	samples = Mat_<Vec<_Tp, 2> >::zeros(2, (int []){1, n});
	Mat_<Vec<_Tp, 2> > points(2, (int[]){1,n});
	complex<_Tp> delta = (right - left) / (_Tp)n;
	for (int i = 0; i < n; ++i)
	{
		points.template at<complex<_Tp> >(0, i) = left + delta * (_Tp)i;
	}
	samples = points;
	return 0;
}


double sincos_bump(double x, int m);

double sqrt_bump(double x, int m);

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

template <typename _Tp>
int fchi(const Mat_<Vec<_Tp, 2> > &x_pt, const Chi_Ctrl_Param &ctrl_param, const string &opt, Mat_<Vec<_Tp, 2> > &y_val)
{
	Chi_Ctrl_Param param = ctrl_param;

//	Mat_<Vec<_Tp, 2> >  mat = Mat_<Vec<_Tp, 2> >::zeros(x_pt.dims, x_pt.size);
	Mat_<Vec<_Tp, 2> >  mat(x_pt.dims, x_pt.size, Vec<_Tp, 2>(0,0));
	if (opt == "sincos")
	{
		MatConstIterator_<Vec<_Tp, 2> > it1 = x_pt.begin(), end1 = x_pt.end();
		MatIterator_<Vec<_Tp, 2> > it2 = mat.begin();
		for (; it1 != end1; ++it1, ++it2)
		{
			double w = (double)((*it1)[0]);
			double f = 0;
			if (w <= (param.cL - param.epL) || w >= (param.cR + param.epR))
			{
				f = 0;
			}
			else if (w >= (param.cL + param.epL) && w <= (param.cR - param.epR))
			{
				f = 1;
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

			(*it2)[0] = (_Tp)f;
		}
	}
	else if (opt == "sqrt")
	{
		MatConstIterator_<Vec<_Tp, 2> > it1 = x_pt.begin(), end1 = x_pt.end();
		MatIterator_<Vec<_Tp, 2> > it2 = mat.begin();
		for (; it1 != end1; ++it1, ++it2)
		{
			double w = (double)((*it1)[0]);
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

			(*it2)[0] = (_Tp)f;
		}
	}
	else
	{
		return -2;
	}

	y_val = mat;

	return 0;
}

struct OneD_FS_Param
{
	Smart64FArray 	ctrl_points;
	Smart64FArray	epsilons;
	SmartIntArray	highpass_ds_folds;
	int				lowpass_ds_fold;
	int degree;
	string opt;
};

struct MD_FS_Param
{
	SmartArray<OneD_FS_Param>	oned_fs_param_at_dim;
//	SmartIntArray				lowpass_ds_folds;
};

struct ML_MD_FS_Param
{
	SmartArray<MD_FS_Param> md_fs_param_at_level;
	int 					nlevels;
	int 					ndims;
	SmartIntArray			ext_border;
	string					ext_method;
	bool 					isSym;
	ML_MD_FS_Param():nlevels(0), ndims(0), isSym(false){}

	ML_MD_FS_Param(int lvl, int d): nlevels(lvl), ndims(d), ext_border(ndims), isSym(false)
	{
		md_fs_param_at_level.reserve(nlevels);
		for (int i = 0; i < nlevels; ++i)
		{
			md_fs_param_at_level[i].oned_fs_param_at_dim.reserve(ndims);
		}
	}
};

template <typename _Tp>
struct OneD_Filter
{
	Mat_<Vec<_Tp, 2> > coefs;
	Mat_<Vec<_Tp, 2> > folded_coefs;
	int highpass_ds_fold;
	SmartIntArray support_after_ds;
	SmartIntArray sym_support_after_ds;
	bool isLowPass;
};

template <typename _Tp>
struct OneD_FSystem
{
	SmartArray<OneD_Filter<_Tp> >	filters;
	int lowpass_ds_fold;

	OneD_FSystem():lowpass_ds_fold(1){}
	OneD_FSystem(int n):lowpass_ds_fold(1)
	{
		filters.reserve(n);
	}
};

template <typename _Tp>
struct MD_FSystem
{
	SmartArray<OneD_FSystem<_Tp> >	oned_fs_at_dim;
};

template <typename _Tp>
struct ML_MD_FSystem
{
	int nlevels;
	int ndims;
	SmartArray<MD_FSystem<_Tp> > md_fs_at_level;

	ML_MD_FSystem(): nlevels(0), ndims(0) {}
	ML_MD_FSystem(int lvl, int d): nlevels(lvl), ndims(d)
	{
		md_fs_at_level.reserve(nlevels);
		for (int i = 0; i < nlevels; ++i)
		{
			md_fs_at_level[i].oned_fs_at_dim.reserve(ndims);
		}
	}
};



/*
 * This function reduces 'filter' for each dim by folds[cur_dim].
 *  'folded_filter' is the reduced filter. 'support' is the support of 'filter'.
 *
 *  [ 0 0 2 3 0 1 0 0] reduced by 2 to [0 1 2 3]
 *
 */

template<typename _Tp>
int downsample_in_fd_by2(const Mat_<Vec<_Tp, 2> > &filter, SmartIntArray &folds, Mat_<Vec<_Tp, 2> > &folded_filter, SmartArray<SmartIntArray> &support, SmartArray<SmartIntArray> &sym_support)
{
	if (filter.dims != folds.len)
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
	// This holds support indices for corresponding symmetric filter.
	SmartArray<SmartIntArray> sym_supp_set(folded_total);
	Mat_<Vec<_Tp, 2> > folded_mat(dims, folded_range, Vec<_Tp, 2>(0,0));
	int supp_idx = 0;
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

			complex<_Tp> gtzero;
			supp_set[supp_idx] = src_cur_pos.clone();
			SmartIntArray sym_cur_pos(dims);
			for (int i = 0; i < dims; ++i)
			{
				if (src_cur_pos[i] > 0)
				{
					sym_cur_pos[i] = src_end_pos[i] - src_cur_pos[i];
				}
			}
			int sym_supp_idx = (supp_idx > 0) ? (folded_total - supp_idx) : 0;
			sym_supp_set[sym_supp_idx] = sym_cur_pos.clone();
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

					complex<_Tp> a = filter.template at<complex<_Tp> >(src_cur_pos);
					//TODO fabs is not good practice for template programming
					if (fabs(a.real() - 0) > numeric_limits<_Tp>::epsilon()
						|| fabs(a.imag() - 0) > numeric_limits<_Tp>::epsilon())
					{
						supp_set[supp_idx] = src_cur_pos.clone();
						for (int i = 0; i < dims; ++i)
						{
							if (src_cur_pos[i] > 0)
							{
								sym_cur_pos[i] = src_end_pos[i] - src_cur_pos[i];
							}
						}
						sym_supp_set[sym_supp_idx] = sym_cur_pos.clone();
						gtzero = a;
						break;
					}
					// User-defined

					cur_dim = src_dims - 1;
					src_cur_pos[cur_dim] += src_step[cur_dim];
				}

			} // Inner loop end

			folded_mat.template at<complex<_Tp> >(src_cur_pos) = gtzero;
			++supp_idx;
			//--

			cur_dim = src_dims - 1;
			src_cur_pos[cur_dim] += src_step[cur_dim];
		}

	}
	folded_filter = folded_mat;
	support = supp_set;
	sym_support = sym_supp_set;

	return 0;
}

template <typename _Tp>
int construct_1d_filter_system(const Mat_<Vec<_Tp, 2> > &x_pts, const OneD_FS_Param &oned_fs_param, OneD_FSystem<_Tp> &filter_system)
{
	int ctrl_pts_num = oned_fs_param.ctrl_points.len;
	if (ctrl_pts_num < 2 || ctrl_pts_num != oned_fs_param.epsilons.len)
	{
		return -1;
	}
//	OneD_FSystem filter_system(ctrl_pts_num);
	filter_system = OneD_FSystem<_Tp>(ctrl_pts_num);
	const Smart64FArray &ctrl_points = oned_fs_param.ctrl_points;
	const Smart64FArray &epsilons = oned_fs_param.epsilons;
	const string &opt = oned_fs_param.opt;


	Mat_<Vec<_Tp, 2> > shift_right_x = x_pts.clone();
	shift_right_x += Scalar(2 * M_PI, 0);

	Mat_<Vec<_Tp, 2> > shift_left_x = x_pts.clone();
	shift_left_x += Scalar(-2 * M_PI, 0);

	for (int i = 0; i < ctrl_pts_num; ++i)
	{
		OneD_Filter<_Tp> &this_filter = filter_system.filters[i];
		Chi_Ctrl_Param this_filter_param;
		if (i != 0)
		{
			this_filter_param.cL = ctrl_points[i - 1];
			this_filter_param.epL = epsilons[i - 1];
			this_filter_param.cR = ctrl_points[i];
			this_filter_param.epR = epsilons[i];
		}
		else
		{
			this_filter_param.cL = ctrl_points[ctrl_pts_num - 1];
			this_filter_param.epL = epsilons[ctrl_pts_num - 1];
			this_filter_param.cR = ctrl_points[0];
			this_filter_param.epR = epsilons[0];
		}
		this_filter_param.degree = oned_fs_param.degree;

		if (this_filter_param.cR < this_filter_param.cL)
		{
			this_filter_param.cR += 2 * M_PI;
		}

		this_filter.isLowPass = false;
		if (this_filter_param.cL - this_filter_param.epL < 0 && this_filter_param.cR + this_filter_param.epR > 0)
		{
			this_filter.isLowPass = true;
		}

		Mat_<Vec<_Tp, 2> > shift_right_filter;
		Mat_<Vec<_Tp, 2> > shift_left_filter;

		fchi<_Tp>(shift_right_x, this_filter_param, opt, shift_right_filter);
		fchi<_Tp>(x_pts, this_filter_param, opt, this_filter.coefs);
		fchi<_Tp>(shift_left_x, this_filter_param, opt, shift_left_filter);

		this_filter.coefs = this_filter.coefs + shift_right_filter + shift_left_filter;

		// One-D filter is assumed to be 1xn filter.
		SmartIntArray ds_folds(2);
		ds_folds[0] = 1;
		ds_folds[1] = oned_fs_param.highpass_ds_folds[i];
		this_filter.highpass_ds_fold = oned_fs_param.highpass_ds_folds[i];

		SmartArray<SmartIntArray> supp, sym_supp;
		downsample_in_fd_by2<_Tp>(this_filter.coefs, ds_folds, this_filter.folded_coefs, supp, sym_supp);

		//Simplify the support coordinates by removing the first dim.
		SmartIntArray reduced_supp(supp.len);
		SmartIntArray reduced_sym_supp(supp.len);
		for (int i = 0; i < supp.len; ++i)
		{
			reduced_supp[i] = supp[i][1];
			reduced_sym_supp[i] = sym_supp[i][1];
		}
		this_filter.support_after_ds = reduced_supp;
		this_filter.sym_support_after_ds = reduced_sym_supp;
	}
	filter_system.lowpass_ds_fold = oned_fs_param.lowpass_ds_fold;

	return 0;
}

// Here we use vector instead of SmartArray, because it is quite difficult to to know filter number
// before all filters are constructed.
// Need a macro switch between Mat_<Vec<double,2> > and Mat_<Vec<float,2> >.

//template<typename _Tp>
//struct ML_MC_Coefs_Set
//{
//public:
//	typedef vector<vector<Mat_<Vec<_Tp, 2> > > > type;
//};
//
//template<typename _Tp>
//struct MC_Coefs_Set
//{
//	typedef vector<Mat_<Vec<_Tp, 2> > > type;
//};
//
//template<typename _Tp>
//struct ML_MC_Filter_Norms_Set
//{
//	typedef vector<vector<_Tp> > type;
//};
//
//template<typename _Tp>
//struct MC_Filter_Norms_Set
//{
//	typedef vector<_Tp> type;
//};

int figure_good_mat_size(const ML_MD_FS_Param &fs_param, const SmartIntArray &mat_size, const SmartIntArray &border, SmartIntArray &better);

int compose_fs_param(int nlevels, int ndims, const string &fs_param_opt, int ext_size, const string &ext_opt, bool isSym, ML_MD_FS_Param &ml_md_fs_param);

template<typename _Tp>
int check_mat_to_decompose(const ML_MD_FS_Param &fs_param, const Mat_<Vec<_Tp, 2> > &mat)
{
	SmartIntArray mat_size(mat.dims, mat.size);
	SmartIntArray good_mat_size(mat.dims, 1);

	for (int i = 0; i < fs_param.nlevels; ++i)
	{
		for (int j = 0; j < fs_param.ndims; ++j)
		{
			const OneD_FS_Param &this_dim_param = fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j];
			for (int k = 0; k < this_dim_param.highpass_ds_folds.len; ++k)
			{
				if (mat_size[j] % this_dim_param.highpass_ds_folds[k] != 0)
				{
					return -1;
				}
			}
			if (mat_size[j] % (this_dim_param.lowpass_ds_fold * 2) != 0)
			{
				return -2;
			}
			mat_size[j] /= this_dim_param.lowpass_ds_fold;
		}
	}

	return 0;
}


template<typename _Tp>
int decompose_by_ml_md_filter_bank2(const ML_MD_FS_Param &fs_param, const Mat_<Vec<_Tp, 2> > &input, ML_MD_FSystem<_Tp> &ml_md_filter_system, typename ML_MC_Filter_Norms_Set<_Tp>::type &norms_set, typename ML_MC_Coefs_Set<_Tp>::type &coefs_set)
{

	if (fs_param.nlevels < 1
		|| input.dims != fs_param.ndims)
	{
		return -1;
	}

	int nlevels = fs_param.nlevels;
	int ndims = fs_param.ndims;
	double input_size_prod = (double)input.total();

	coefs_set.reserve(nlevels);
	coefs_set.resize(nlevels);
	norms_set.reserve(nlevels);
	norms_set.resize(nlevels);

	// This mat is to store product of low-pass filters of previous levels.
	Mat_<Vec<_Tp, 2> > last_lowpass_product;
	ml_md_filter_system = ML_MD_FSystem<_Tp>(nlevels, ndims);
	for (int cur_lvl = 0; cur_lvl < nlevels; ++cur_lvl)
	{

		// This mat is referred to as low-pass channel output last level.
		Mat_<Vec<_Tp, 2> > last_approx;
		Mat_<Vec<_Tp, 2> > last_approx_center_part;

		if (cur_lvl == 0)	// when last_approx is empty().
		{
			normalized_fft<_Tp>(input, last_approx);
			center_shift<_Tp>(last_approx, last_approx);
		}
		else
		{
			last_approx = coefs_set[cur_lvl - 1][coefs_set[cur_lvl - 1].size() - 1].coefs;
			coefs_set[cur_lvl - 1].pop_back();
		}

		// This is the size of full filters at this level.
		SmartIntArray md_filter_size(ndims, last_approx.size);
		// This is the number of filters at each dim at this level.
		SmartIntArray filter_numbers_at_dim(ndims);
		SmartArray<SmartIntArray> lowpass_center_range_at_dim(ndims);
		const MD_FS_Param &this_level_param = fs_param.md_fs_param_at_level[cur_lvl];
		MD_FSystem<_Tp> &this_level_md_fs = ml_md_filter_system.md_fs_at_level[cur_lvl];
		double lowpass_ds_prod = 1.0;
		for (int cur_dim = 0; cur_dim < ndims; ++cur_dim)		// Every dim in this level
		{

			Mat_<Vec<_Tp, 2> > x_pts;
			linspace<_Tp>(complex<_Tp>(-M_PI, 0), complex<_Tp>(M_PI, 0), md_filter_size[cur_dim], x_pts);

			construct_1d_filter_system<_Tp>(x_pts, this_level_param.oned_fs_param_at_dim[cur_dim],
					                   this_level_md_fs.oned_fs_at_dim[cur_dim]);

			filter_numbers_at_dim[cur_dim] = this_level_md_fs.oned_fs_at_dim[cur_dim].filters.len;

			lowpass_center_range_at_dim[cur_dim].reserve(3);
			lowpass_center_range_at_dim[cur_dim][0] = md_filter_size[cur_dim] / 2
										        - md_filter_size[cur_dim] / this_level_param.oned_fs_param_at_dim[cur_dim].lowpass_ds_fold / 2;
			lowpass_center_range_at_dim[cur_dim][1] = -1;
			lowpass_center_range_at_dim[cur_dim][2] = lowpass_center_range_at_dim[cur_dim][0]
			                                               + md_filter_size[cur_dim] / this_level_param.oned_fs_param_at_dim[cur_dim].lowpass_ds_fold
			                                               - 1;
			lowpass_ds_prod *= this_level_param.oned_fs_param_at_dim[cur_dim].lowpass_ds_fold;
		}

		int N = filter_numbers_at_dim[ndims - 1];
		SmartIntArray steps(ndims, 1);
		steps[ndims - 1] = 1;
		for (int i = ndims - 2; i >= 0; --i)
		{
			steps[i] = filter_numbers_at_dim[i + 1] * steps[i + 1];
			N *= filter_numbers_at_dim[i];
		}

		if (fs_param.isSym)
		{
			N = (N & 1) ? (N / 2 + 1) : (N / 2);
		}

		// This is to store down-sampled filters chosen currently for each dim.
		SmartArray<Mat_<Vec<_Tp, 2> > > chosen_ds_filter_at_dim(ndims);
		// This is to store full filters chosen currently for each dim.
		SmartArray<Mat_<Vec<_Tp, 2> > > chosen_full_filter_at_dim(ndims);
		SmartArray<Mat_<Vec<_Tp, 2> > > lowpass_filter_center_part_at_dim(ndims);
		SmartArray<SmartIntArray> supp_after_ds_at_dim(ndims);
		Mat_<Vec<_Tp, 2> > lowpass_filter;
		bool is_odd_center = false;
		for (int n = 0; n < N; ++n)
		{
			// This method is stupid, but clear.
			// Because N would not be too large, typically hundreds, this method would not cost much.
			SmartIntArray cur_pos(ndims);
			SmartIntArray sym_cur_pos(ndims);
			for (int i = 0, rem = n; i < ndims; ++i)
			{
				cur_pos[i] = rem / steps[i];
				rem -= (cur_pos[i] * steps[i]);
				sym_cur_pos[i] = filter_numbers_at_dim[i] - 1 - cur_pos[i];
			}

			//User-Defined actions
			bool is_lowpass = true;
			double highpass_ds_prod = 1.0;
			//Should start at 0, since we need to check lowpass for ALL filters
			for (int i = 0; i < ndims; ++i)
			{
				const OneD_Filter<_Tp> &this_filter = this_level_md_fs.oned_fs_at_dim[i].filters[cur_pos[i]];
				chosen_ds_filter_at_dim[i] = this_filter.folded_coefs;
				chosen_full_filter_at_dim[i] = this_filter.coefs;
				supp_after_ds_at_dim[i]= this_filter.support_after_ds;
				lowpass_filter_center_part_at_dim[i] = this_filter.coefs.colRange(lowpass_center_range_at_dim[i][0],
																				  lowpass_center_range_at_dim[i][2] + 1);
				is_lowpass = is_lowpass && this_filter.isLowPass;
				highpass_ds_prod *= this_filter.highpass_ds_fold;
			}

			if (is_lowpass)
			{
				// Plan A --
				Mat_<Vec<_Tp, 2> > md_filter_center_part;
				tensor_product<_Tp>(lowpass_filter_center_part_at_dim, md_filter_center_part);
				pw_pow<_Tp>(md_filter_center_part, static_cast<_Tp>(2), md_filter_center_part);
				// -- Plan A

//				stringstream ss;
//				ss << "Test-Data/output/filter_" << cur_lvl << "_" << n << ".txt";
//				print_mat_details_g<_Tp>(md_filter_center_part, 2, ss.str());

				// -- Speed up Plan B
				if (fs_param.isSym && cur_pos == sym_cur_pos)
				{
					// If it is odd center, then there must be only one lowpass filter.
					is_odd_center = true;
				}
				// -- Plan B

				if (lowpass_filter.empty())	// when lowpass_filter is empty().
				{
					lowpass_filter = md_filter_center_part;
				}
				else
				{
					lowpass_filter += md_filter_center_part;
				}
			}
			else
			{
				//  Plan A--
				Mat_<Vec<_Tp, 2> > folded_md_filter;
				Mat_<Vec<_Tp, 2> > last_approx_subarea;
				tensor_product<_Tp>(chosen_ds_filter_at_dim, folded_md_filter);
				mat_select<_Tp>(last_approx, supp_after_ds_at_dim, last_approx_subarea);

				pw_mul<_Tp>(last_approx_subarea, folded_md_filter, last_approx_subarea);

				icenter_shift<_Tp>(last_approx_subarea, last_approx_subarea);
				normalized_ifft<_Tp>(last_approx_subarea, last_approx_subarea);

//				coefs_set[cur_lvl].push_back(last_approx_subarea);
				Coefs_Item<_Tp> item;
				item.is_lowpass = false;
				item.coefs = last_approx_subarea;
				coefs_set[cur_lvl].push_back(item);
				// -- Plan A

				// Compute filters' norms.
				if (cur_lvl == 0)	// when 'last_lowpass_product' is empty().
				{
					// At first level, 'last_lowpass_product' is regarded as I.
					double l2norm = lpnorm<_Tp>(folded_md_filter, static_cast<_Tp>(2));
					norms_set[cur_lvl].push_back(l2norm / sqrt(input_size_prod) * sqrt(highpass_ds_prod));
				}
				else
				{
					Mat_<Vec<_Tp, 2> > last_lowpass_product_subarea;
					mat_select<_Tp>(last_lowpass_product, supp_after_ds_at_dim, last_lowpass_product_subarea);
					pw_mul<_Tp>(last_lowpass_product_subarea, folded_md_filter, last_lowpass_product_subarea);
					double l2norm = lpnorm<_Tp>(last_lowpass_product_subarea, 2);
					norms_set[cur_lvl].push_back(l2norm / sqrt(input_size_prod) * sqrt(highpass_ds_prod));
				}
			}
			//-- User Action
		}

		if (fs_param.isSym && !is_odd_center)
		{
			Mat_<Vec<_Tp, 2> > another_half_lowpass;
			rotate180shift1<_Tp>(lowpass_filter, another_half_lowpass);
			lowpass_filter += another_half_lowpass;
		}

		pw_sqrt<_Tp>(lowpass_filter, lowpass_filter);


		if (true)
		{
			// Plan A --
			mat_select<_Tp>(last_approx, lowpass_center_range_at_dim, last_approx);

			pw_mul<_Tp>(last_approx, lowpass_filter, last_approx);
			// -- Plan A

			if (cur_lvl == nlevels - 1)
			{
				icenter_shift<_Tp>(last_approx, last_approx);
				normalized_ifft<_Tp>(last_approx, last_approx);
//				coefs_set[cur_lvl].push_back(last_approx);

				Coefs_Item<_Tp> item;
				item.is_lowpass = true;
				item.coefs = last_approx;
				coefs_set[cur_lvl].push_back(item);
			}
			else
			{
//				coefs_set[cur_lvl].push_back(last_approx);
				Coefs_Item<_Tp> item;
				item.is_lowpass = true;
				item.coefs = last_approx;
				coefs_set[cur_lvl].push_back(item);
			}

			// Compute filter norms. Here update 'last_lowpass_product'.
			if (cur_lvl == 0)
			{
				last_lowpass_product = lowpass_filter;
			}
			else
			{
				mat_select<_Tp>(last_lowpass_product, lowpass_center_range_at_dim, last_lowpass_product);
				pw_mul<_Tp>(last_lowpass_product, lowpass_filter, last_lowpass_product);
			}
			last_lowpass_product = last_lowpass_product * sqrt(lowpass_ds_prod);
		}
	}

	double l2norm = lpnorm<_Tp>(last_lowpass_product, (_Tp)2);
	norms_set[nlevels - 1].push_back(l2norm / sqrt(input_size_prod));

	return 0;
}


template<typename _Tp>
int reconstruct_by_ml_md_filter_bank2(const ML_MD_FS_Param &fs_param, const ML_MD_FSystem<_Tp> &filter_system, const typename ML_MC_Coefs_Set<_Tp>::type &coefs_set, Mat_<Vec<_Tp, 2> > &rec)
{
	int nlevels = fs_param.nlevels;
	int ndims = fs_param.ndims;
	if (nlevels < 1)
	{
		return -1;
	}


	// Every Level
	Mat_<Vec<_Tp, 2> > upper_level_lowpass_approx;
	Mat_<Vec<_Tp, 2> > this_level_lowpass_approx;
	for (int cur_lvl = nlevels - 1; cur_lvl >= 0; --cur_lvl)
	{
		const typename MC_Coefs_Set<_Tp>::type &this_level_coefs_set = coefs_set[cur_lvl];
		const MD_FSystem<_Tp> &this_level_fs = filter_system.md_fs_at_level[cur_lvl];
		if (cur_lvl == nlevels - 1)
		{
			this_level_lowpass_approx = this_level_coefs_set[this_level_coefs_set.size() - 1].coefs.clone();
			normalized_fft<_Tp>(this_level_lowpass_approx, this_level_lowpass_approx);
			center_shift<_Tp>(this_level_lowpass_approx, this_level_lowpass_approx);
		}
		else
		{
			this_level_lowpass_approx = upper_level_lowpass_approx;
		}

		SmartIntArray full_filter_size_at_dim(ndims);
		SmartIntArray filter_numbers_at_dim(ndims);
		SmartArray<SmartIntArray> lowpass_center_range_at_dim(ndims);
		for (int i = 0; i < ndims; ++i)
		{
			full_filter_size_at_dim[i] = this_level_lowpass_approx.size[i] * this_level_fs.oned_fs_at_dim[i].lowpass_ds_fold;
			filter_numbers_at_dim[i] = this_level_fs.oned_fs_at_dim[i].filters.len;
			lowpass_center_range_at_dim[i].reserve(3);
			lowpass_center_range_at_dim[i][0] = full_filter_size_at_dim[i] / 2
												- full_filter_size_at_dim[i] / this_level_fs.oned_fs_at_dim[i].lowpass_ds_fold / 2;
			lowpass_center_range_at_dim[i][1] = -1;
			lowpass_center_range_at_dim[i][2] = lowpass_center_range_at_dim[i][0]
														   + full_filter_size_at_dim[i] / this_level_fs.oned_fs_at_dim[i].lowpass_ds_fold
														   - 1;
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

		if (fs_param.isSym)
		{
			N = (N & 1) ? (N / 2 + 1) : (N / 2);
		}

		Mat_<Vec<_Tp, 2> > this_level_highpass_sum(ndims, full_filter_size_at_dim, Vec<_Tp, 2>(0,0));
		int highpass_coef_index = 0;
		Mat_<Vec<_Tp, 2> > lowpass_filter;
		bool is_odd_center = false;
		for (int n = 0; n < N; ++n)
		{
			SmartArray<Mat_<Vec<_Tp, 2> > > chosen_ds_filter_at_dim(ndims);
			SmartArray<Mat_<Vec<_Tp, 2> > > lowpass_filter_center_part_at_dim(ndims);
			SmartArray<SmartIntArray> supp_after_ds_at_dim(ndims);
			SmartArray<SmartIntArray> sym_supp_after_ds_at_dim(ndims);
			SmartIntArray cur_pos(ndims);
			SmartIntArray sym_cur_pos(ndims);
			for (int i = 0, rem = n; i < ndims; ++i)
			{
				cur_pos[i] = rem / steps[i];
				rem -= (cur_pos[i] * steps[i]);
				sym_cur_pos[i] = filter_numbers_at_dim[i] - 1 - cur_pos[i];
			}

			//User-Defined actions
			//A combination is found.
			bool is_lowpass = true;
			//Start at 0, since we need to check lowpass for all filters
			for (int i = 0; i < ndims; ++i)
			{
				const OneD_Filter<_Tp> &this_filter = this_level_fs.oned_fs_at_dim[i].filters[cur_pos[i]];
				chosen_ds_filter_at_dim[i] = this_filter.folded_coefs;
				supp_after_ds_at_dim[i]= this_filter.support_after_ds;
				sym_supp_after_ds_at_dim[i] = this_filter.sym_support_after_ds;
				lowpass_filter_center_part_at_dim[i] = this_filter.coefs.colRange(lowpass_center_range_at_dim[i][0],
																				  lowpass_center_range_at_dim[i][2] + 1);
				is_lowpass = is_lowpass && this_filter.isLowPass;
			}

			if (is_lowpass)
			{
				Mat_<Vec<_Tp, 2> > md_filter_center_part;
				tensor_product<_Tp>(lowpass_filter_center_part_at_dim, md_filter_center_part);
				pw_pow<_Tp>(md_filter_center_part, static_cast<_Tp>(2), md_filter_center_part);

				// -- Speed up Plan B
				if (fs_param.isSym && cur_pos == sym_cur_pos)
				{
					is_odd_center = true;
				}
				// -- Plan B

				if (lowpass_filter.empty())
				{
					lowpass_filter = md_filter_center_part;
				}
				else
				{
					lowpass_filter += md_filter_center_part;
				}
			}
			else
			{
				Mat_<Vec<_Tp, 2> > this_channel_coef = this_level_coefs_set[highpass_coef_index].coefs.clone();
				normalized_fft<_Tp>(this_channel_coef, this_channel_coef);
				center_shift<_Tp>(this_channel_coef, this_channel_coef);

				Mat_<Vec<_Tp, 2> > ds_filter;
				tensor_product<_Tp>(chosen_ds_filter_at_dim, ds_filter);
				pw_mul<_Tp>(this_channel_coef, ds_filter, this_channel_coef);
				mat_subadd(this_level_highpass_sum, supp_after_ds_at_dim, this_channel_coef);

//				stringstream ss;
//				ss << "Test-Data/output/rec_mul_" << cur_lvl << "_" << n << ".txt";
//				print_mat_details_g<_Tp>(this_channel_coef, 2, ss.str());

				++highpass_coef_index;
			}
			//--
		}


		// -- Speed up Plan B
		if (fs_param.isSym)
		{
			if (!is_odd_center)
			{
				Mat_<Vec<_Tp, 2> > another_half_lowpass;
				rotate180shift1<_Tp>(lowpass_filter, another_half_lowpass);
				lowpass_filter += another_half_lowpass;
			}

			Mat_<Vec<_Tp, 2> > another_half_highpass_sum = this_level_highpass_sum.clone();
//			conj<_Tp>(another_half_highpass_sum);
			rotate180shift1conj<_Tp>(another_half_highpass_sum, another_half_highpass_sum);
			this_level_highpass_sum += another_half_highpass_sum;
		}
		// -- Plan B

		pw_sqrt<_Tp>(lowpass_filter, lowpass_filter);

		if (true)
		{
			pw_mul<_Tp>(this_level_lowpass_approx, lowpass_filter, this_level_lowpass_approx);

//			stringstream ss;
//			ss << "Test-Data/output/rec_mul_lp_mid_" << cur_lvl << ".txt";
//			print_mat_details_g<_Tp>(this_level_lowpass_approx, 2, ss.str());

			mat_subadd<_Tp>(this_level_highpass_sum, lowpass_center_range_at_dim, this_level_lowpass_approx);
			upper_level_lowpass_approx = this_level_highpass_sum;
		}

	}

	icenter_shift<_Tp>(upper_level_lowpass_approx, upper_level_lowpass_approx);
	normalized_ifft<_Tp>(upper_level_lowpass_approx, upper_level_lowpass_approx);

	rec = upper_level_lowpass_approx;


	return 0;
}




#endif
