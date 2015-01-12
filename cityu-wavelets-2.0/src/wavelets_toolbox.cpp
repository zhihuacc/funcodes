#include "../include/wavelets_toolbox.h"

int figure_good_mat_size(const ML_MD_FS_Param &fs_param, const SmartIntArray &mat_size, const SmartIntArray &border, SmartIntArray &better)
{
	if (fs_param.ndims != mat_size.len || mat_size.len != border.len)
	{
		return -1;
	}

	SmartIntArray good_pad(fs_param.ndims, 1);

	for (int i = 0; i < fs_param.nlevels; ++i)
	{
		for (int j = 0; j < fs_param.ndims; ++j)
		{
			const OneD_FS_Param &this_dim_param = fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j];
			if (this_dim_param.highpass_ds_folds.len <= 0)
			{
				return -1;
			}

			int dummy;
			int this_dim_lcd = this_dim_param.highpass_ds_folds[0];
			for (int k = 0; k < this_dim_param.highpass_ds_folds.len; ++k)
			{
				hcf_lcd(this_dim_lcd, this_dim_param.highpass_ds_folds[k], dummy, this_dim_lcd);
			}

			hcf_lcd(this_dim_lcd, this_dim_param.lowpass_ds_fold * 2, dummy, this_dim_lcd);
			good_pad[j] *= this_dim_lcd;
		}
	}

	SmartIntArray better_border(mat_size.len);
	for (int i = 0; i < mat_size.len; ++i)
	{
		int r = (mat_size[i] + border[i]) % good_pad[i];
		if (r != 0)
		{
			good_pad[i] = good_pad[i] - r;
		}
		else
		{
			good_pad[i] = 0;
		}
		better_border[i] = border[i] + good_pad[i];
	}
	better = better_border;

	return 0;
}

double sincos_bump(double x, int m)
{
	double f;
	if (x <= 0 || x >= 1)
	{
		f = 0;
	}
	else
	{
		f = 0;
		for (int j = 0; j < m; ++j)
		{
			f += nchoosek(m - 1 + j, j) * pow(x, j);
		}

		f *= pow(1 - x, m);
		f = sin(0.5 * M_PI * f);
	}

	return f;
}

double sqrt_bump(double x, int m)
{
	double f;
	if (x <= 0 || x >= 1)
	{
		f = 0;
	}
	else
	{
		f = 0;
		for (int k = 0; k < m; ++k)
		{
			f += (k&1 ? -1 : 1) * nchoosek(2 * m - 1, m - 1 - k) * nchoosek(m - 1 + k, k) * pow(x, k);
		}

		f *= pow(x, m);
		f = sqrt(f);
	}

	return f;
}

int compose_fs_param(int nlevels, int ndims, const string &fs_param_opt, int ext_size, const string &ext_opt, ML_MD_FS_Param &ml_md_fs_param)
{
	ML_MD_FS_Param fs_param(nlevels, ndims);
	for (int i = 0; i < fs_param.nlevels; ++i)
	{
		for (int j = 0; j < fs_param.ndims; ++j)
		{
			if (fs_param_opt == "CTF3")
			{
				fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].ctrl_points = Smart64FArray(3, (double[]){-33.0/32.0, 33.0/32.0, M_PI});
				fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].epsilons = Smart64FArray(3, (double[]){69.0/128.0, 69.0/128.0, 51.0/512.0});
				fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].degree = 1;
				fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].highpass_ds_folds = SmartIntArray(3, (int[]){2,2,2});
				fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].lowpass_ds_fold = 2;
				fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].opt = "sincos";
			}
			else if (fs_param_opt == "CTF6D4")
			{
				fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].ctrl_points = Smart64FArray(6, (double[]){-2, -1.145796, 0, 1.145796, 2, M_PI});
				fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].epsilons = Smart64FArray(6, (double[]){0.35, 0.3, 0.125, 0.3, 0.35, 0.0778});
				fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].degree = 1;
				fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].highpass_ds_folds = SmartIntArray(6, (int[]){4,4,4,4,4,4});
				fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].lowpass_ds_fold = 2;
				fs_param.md_fs_param_at_level[i].oned_fs_param_at_dim[j].opt = "sincos";
			}
			else if (fs_param_opt == "CTF6D2")
			{

			}
		}
	}
	fs_param.ext_border = SmartIntArray(ndims, ext_size);
	fs_param.ext_method = ext_opt;

	ml_md_fs_param = fs_param;
	return 0;
}
