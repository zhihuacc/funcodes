#include "../include/dynamical_nested_loops.h"

//int dynamical_nested_loops(const vector<iter_param> &iter_params
//		                   , int level, factor_func func, vector<int> curr_config
//		                   , void *observed_data
//		                   , float &sum)
//{
//	if (level < 0)
//	{
//		return 0;
//	}
//
//	// Get in the innerest loop.
//	if (level >= iter_params.size())
//	{
//        sum += func(curr_config, observed_data);
//	}
//
//    iter_param this_level_loop_param = iter_params[level];
//
//    for (int i = this_level_loop_param.start
//    	 ; i < this_level_loop_param.end
//    	 ; i += this_level_loop_param.step)
//    {
//    	curr_config.push_back(i);
//
//        dynamical_nested_loops(iter_params, level + 1, func, curr_config, observed_data, sum);
//
//        curr_config.pop_back();
//    }
//
//	return 0;
//}
