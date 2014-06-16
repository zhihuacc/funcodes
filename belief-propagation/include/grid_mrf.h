#ifndef GRID_MRF_H
#define GRID_MRF_H

#include "factor_graph.h"

#define LABEL_SET_SIZE  32
enum
{
	BP_SUM_PRODUCT = 0,
	BP_MIN_SUM
};

class grid_mrf
{
private:
	int rows_;
	int cols_;
	factor_graph factor_graph_;
    int min_sum_method(int iter, bool is_sync);
    int sum_product_method(int iter, bool is_sync);

public:
	grid_mrf(int rows, int cols, void *observed_data);
	int inference(int method, int iter, bool is_sync);
    int draw_graph();

};

#endif
