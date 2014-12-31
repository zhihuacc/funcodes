#include "../include/wavelets_toolbox.h"

int figure_good_mat_size(const ML_MD_FS_Param &fs_param, const SmartIntArray &mat_size, SmartIntArray &border)
{
	if (fs_param.ndims != mat_size.len || mat_size.len != border.len)
	{
		return -1;
	}

	SmartIntArray good_border(fs_param.ndims, 1);

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

			hcf_lcd(this_dim_lcd, this_dim_param.lowpass_ds_fold, dummy, this_dim_lcd);
			good_border[j] *= this_dim_lcd;
		}
	}

	for (int i = 0; i < mat_size.len; ++i)
	{
		int r = (mat_size[i] + border[i]) % good_border[i];
		if (r != 0)
		{
			good_border[i] = good_border[i] - r;
		}
		else
		{
			good_border[i] = 0;
		}
		border[i] += good_border[i];
	}

	return 0;
}
