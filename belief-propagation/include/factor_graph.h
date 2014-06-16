#ifndef FACTOR_GRAPH_H
#define FACTOR_GRAPH_H

#include "graph_struct.h"



class factor_graph: public graph_topology
{
private:

	struct iter_param
	{
		int start;
		int step;
		int end;
		iter_param():start(0), step(1), end(0) {}
		iter_param(int start0, int step0, int end0): start(start0), step(step0), end(end0){}
	};

	int sum_product_factor_single(const vector<iter_param> &iter_params
			        , graph_node::factor_func func
			        , void *observed_data
			        , const vector<tr1::shared_ptr<message> > &receivd_messages
			        , int level
			        , vector<int> curr_labeling
			        , float &sum);
	int min_sum_factor_single(const vector<iter_param> &iter_params
				        , graph_node::factor_func func
				        , void *observed_data
				        , const vector<tr1::shared_ptr<message> > &receivd_messages
				        , int level
				        , vector<int> curr_labeling
				        , float &minimum);

public:
	factor_graph(int var_cnt, int factor_cnt);
	int do_loopy_min_sum();
	int sum_product_factor_message_pass(int v, int w, bool is_sync);
	int sum_product_variable_message_pass(int v, int w, bool is_sync);
	int min_sum_factor_message_pass(int v, int w, bool is_sync);
	int min_sum_variable_message_pass(int v, int w, bool is_sync);
	int set_factor_func(int node_id, graph_node::factor_func func, void *data, graph_node::destructor free_data);
    int set_label_set_size(int node_id, int size);
    int set_node_type(int node_id, int node_type);

	int min_sum_messages_product_on_node(int node_id);
	int sum_product_messages_product_on_node(int node_id);
	int best_label_on_node(int node_id);
    int draw_graph();
};

#endif
