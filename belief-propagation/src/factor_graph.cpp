#include "../include/factor_graph.h"
#include <iostream>
#include <limits>

using namespace std;

#define FLOAT_PRECISION 0.000001f

factor_graph::factor_graph(int var_cnt, int factor_cnt):graph_topology(var_cnt + factor_cnt, false)
{}

int factor_graph::draw_graph()
{
	for (int v = 0; v < nodes_count(-1); v++)
	{
        adj_iterator it(*this, v);
        cout.width(2);
        cout << v << ":";
        for (int adj = it.begin(); !it.end(); adj = it.next())
        {
            cout.width(2);
            cout << adj << " ";
        }
        cout << endl;
	}

	return 0;
}

int factor_graph::do_loopy_min_sum()
{

	return 0;
}

int factor_graph::sum_product_factor_message_pass(int v, int w, bool is_sync)
{
//	if (v < 0 || v >= nodes_total_count_
//		|| w < 0 || w >= nodes_total_count_)
//	{
//		return -1;
//	}


    tr1::shared_ptr<graph_node> this_node = nodes_[v]; //factor node
    tr1::shared_ptr<graph_node> next_node = nodes_[w];

//    if (this_node->node_type != FACTOR_NODE || next_node->node_type != VAR_NODE)
//    {
//    	return -2;
//    }

    //Sometimes different variables have different label sets.
    int new_message_dim = next_node->label_set_size;
//    if (new_message_dim < 1)
//    {
//    	return -3;
//    }

    int fixed_variable_idx = 0;
    int idx = 0;
    vector<iter_param> iter_params;
    vector<tr1::shared_ptr<message> > received_messages;
    for (list<tr1::shared_ptr<graph_edge> >::const_iterator it = this_node->incident_edges.begin()
    	 ; it != this_node->incident_edges.end()
    	 ; it++)
    {
        if ((*it)->v->node_id != next_node->node_id)
        {
        	if (!((*it)->cool_message))
			{
				(*it)->cool_message = tr1::shared_ptr<message>(new message((*it)->v->label_set_size, 1.0f));
			}

			if (!((*it)->hot_message))
			{
				(*it)->hot_message = tr1::shared_ptr<message>(new message((*it)->v->label_set_size, 1.0f));
			}
        	received_messages.push_back(is_sync ? (*it)->cool_message : (*it)->hot_message);
        }
        else
        {
        	fixed_variable_idx = idx;
        	received_messages.push_back(tr1::shared_ptr<message>(new message(new_message_dim, 1.0f)));
        }
        idx++;
        iter_params.push_back(iter_param(0, 1, (*it)->v->label_set_size));
    }

    tr1::shared_ptr<message> new_message(new message(new_message_dim, 0));
    float minb = numeric_limits<float>::max(), maxb = numeric_limits<float>::min();
    for (int i = 0; i < new_message_dim; i++)
    {
    	iter_params[fixed_variable_idx].start = i;
    	iter_params[fixed_variable_idx].step = 1;
    	iter_params[fixed_variable_idx].end = iter_params[fixed_variable_idx].start + 1;

    	vector<int> curr_config;
    	sum_product_factor_single(iter_params, this_node->cost_func
    			    , this_node->observed_data, received_messages
    			    , 0, curr_config, new_message->at(i));

    	if (new_message->at(i) > maxb)
    	{
    		maxb = new_message->at(i);
    	}
    	else if (new_message->at(i) < minb)
    	{
    		minb = new_message->at(i);
    	}
    }

//	if (sum < 0.000001)
//	{
//		sum = 0.000001;
//	}
//
//	for (int i = 0; i < new_message_dim; i++)
//	{
//		new_message->at(i) /= sum;
//	}

    float sum = 0, range = maxb - minb;
    if (range <= 0.0f)
    {
    	for (int i = 0; i < new_message_dim; i++)
    	{
    		new_message->at(i) = 1;
    		sum += new_message->at(i);
    	}
    }
    else
    {
        range = max(FLOAT_PRECISION, range);
		for (int i = 0; i < new_message_dim; i++)
		{
			new_message->at(i) = (new_message->at(i) - minb) / range;
			sum += new_message->at(i);
		}
    }

	sum = max(sum, FLOAT_PRECISION);
	for (int i = 0; i < new_message_dim; i++)
	{
		new_message->at(i) /= sum;
	}

    tr1::shared_ptr<graph_edge> outgoing_edge;
    for (list<tr1::shared_ptr<graph_edge> >::const_iterator it = next_node->incident_edges.begin()
    	 ; it != next_node->incident_edges.end()
    	 ; it++)
    {
    	if ((*it)->v->node_id == this_node->node_id)
    	{
    		outgoing_edge = *it;
    	}
    }

    outgoing_edge->cool_message = outgoing_edge->hot_message;
    outgoing_edge->hot_message = new_message;

	return 0;
}





int factor_graph::sum_product_factor_single(const vector<iter_param> &iter_params
        , graph_node::factor_func cost_func
        , void *observed_data
        , const vector<tr1::shared_ptr<message> > &received_messages
        , int level
        , vector<int> curr_config
        , float &sum)
{
//	if (level < 0 || iter_params.size() != received_messages.size())
//	{
//		return -1;
//	}

	// Get in the innerest loop.
	if (level >= static_cast<int>(iter_params.size()))
	{
		float product = 1.0;

		for (int i = 0; i < static_cast<int>(received_messages.size()); i++)
		{
			product *= received_messages[i]->at(curr_config[i]);
		}
        sum += cost_func(curr_config, observed_data) * product;
        return 0;
	}

    iter_param this_level_loop_param = iter_params[level];

    for (int i = this_level_loop_param.start
    	 ; i < this_level_loop_param.end
    	 ; i += this_level_loop_param.step)
    {
    	curr_config.push_back(i);

    	sum_product_factor_single(iter_params, cost_func, observed_data, received_messages, level + 1, curr_config, sum);

        curr_config.pop_back();
    }

    return 0;
}

int factor_graph::sum_product_variable_message_pass(int v, int w, bool is_sync)
{
//	if (v < 0 || v >= nodes_total_count_
//		|| w < 0 || w >= nodes_total_count_)
//	{
//		return -1;
//	}


    tr1::shared_ptr<graph_node> this_node = nodes_[v]; //variable node
    tr1::shared_ptr<graph_node> next_node = nodes_[w];

//    if (this_node->node_type != VAR_NODE || next_node->node_type != FACTOR_NODE)
//    {
//    	return -2;
//    }

    //Sometimes different variables have different label sets.
    int new_message_dim = this_node->label_set_size;
//    if (new_message_dim < 1)
//    {
//    	return -3;
//    }

    vector<tr1::shared_ptr<message> > received_messages;
    for (list<tr1::shared_ptr<graph_edge> >::const_iterator it = this_node->incident_edges.begin()
    	 ; it != this_node->incident_edges.end()
    	 ; it++)
    {
        if ((*it)->v->node_id != next_node->node_id)
        {
        	if (!((*it)->cool_message))
			{
				(*it)->cool_message = tr1::shared_ptr<message>(new message((*it)->v->label_set_size, 1.0f));
			}

			if (!((*it)->hot_message))
			{
				(*it)->hot_message = tr1::shared_ptr<message>(new message((*it)->v->label_set_size, 1.0f));
			}
        	received_messages.push_back(is_sync ? (*it)->cool_message : (*it)->hot_message);
        }
        else
        {
        	received_messages.push_back(tr1::shared_ptr<message>(new message(new_message_dim, 1.0)));
        }
    }

    tr1::shared_ptr<message> new_message(new message(new_message_dim, 1.0));
    float minb = numeric_limits<float>::max(), maxb = numeric_limits<float>::min();
    for (int i = 0; i < new_message_dim; i++)
    {
    	for (vector<tr1::shared_ptr<message> >::const_iterator it = received_messages.begin()
    		 ; it != received_messages.end()
    		 ; it++)
    	{
    		new_message->at(i) *= (*it)->at(i);
    	}

    	if (new_message->at(i) > maxb)
    	{
    		maxb = new_message->at(i);
    	}
    	else if (new_message->at(i) < minb)
    	{
    		minb = new_message->at(i);
    	}
    }

    float sum = 0, range = maxb - minb;
    if (range <= 0.0f)
    {
    	for (int i = 0; i < new_message_dim; i++)
    	{
    		new_message->at(i) = 1;
    		sum += new_message->at(i);
    	}
    }
    else
    {
        range = max(FLOAT_PRECISION, range);
		for (int i = 0; i < new_message_dim; i++)
		{
			new_message->at(i) = (new_message->at(i) - minb) / range;
			sum += new_message->at(i);
		}
    }

	sum = max(sum, FLOAT_PRECISION);
	for (int i = 0; i < new_message_dim; i++)
	{
		new_message->at(i) /= sum;
	}

    tr1::shared_ptr<graph_edge> outgoing_edge;
    for (list<tr1::shared_ptr<graph_edge> >::const_iterator it = next_node->incident_edges.begin()
    	 ; it != next_node->incident_edges.end()
    	 ; it++)
    {
    	if ((*it)->v->node_id == this_node->node_id)
    	{
    		outgoing_edge = *it;
    		break;
    	}
    }

    outgoing_edge->cool_message = outgoing_edge->hot_message;
    outgoing_edge->hot_message = new_message;

	return 0;
}

int factor_graph::min_sum_factor_message_pass(int v, int w, bool is_sync)
{
//	if (v < 0 || v >= nodes_total_count_
//		|| w < 0 || w >= nodes_total_count_)
//	{
//		return -1;
//	}


    tr1::shared_ptr<graph_node> this_node = nodes_[v]; //factor node
    tr1::shared_ptr<graph_node> next_node = nodes_[w];

//    if (this_node->node_type != FACTOR_NODE || next_node->node_type != VAR_NODE)
//    {
//    	return -2;
//    }

    //Sometimes different variables have different label sets.
    int new_message_dim = next_node->label_set_size;
//    if (new_message_dim < 1)
//    {
//    	return -3;
//    }

    int fixed_variable_idx = 0;
    int idx = 0;
    vector<iter_param> iter_params;
    vector<tr1::shared_ptr<message> > received_messages;
    for (list<tr1::shared_ptr<graph_edge> >::const_iterator it = this_node->incident_edges.begin()
    	 ; it != this_node->incident_edges.end()
    	 ; it++)
    {
        if ((*it)->v->node_id != next_node->node_id)
        {
        	if (!((*it)->cool_message))
			{
				(*it)->cool_message = tr1::shared_ptr<message>(new message((*it)->v->label_set_size, 0));
			}

			if (!((*it)->hot_message))
			{
				(*it)->hot_message = tr1::shared_ptr<message>(new message((*it)->v->label_set_size, 0));
			}
        	received_messages.push_back(is_sync ? (*it)->cool_message : (*it)->hot_message);
        }
        else
        {
        	fixed_variable_idx = idx;
        	received_messages.push_back(tr1::shared_ptr<message>(new message(new_message_dim, 1.0f / new_message_dim)));
        }
        idx++;
        iter_params.push_back(iter_param(0, 1, (*it)->v->label_set_size));
    }

    tr1::shared_ptr<message> new_message(new message(new_message_dim, numeric_limits<float>::max()));
    float minb = numeric_limits<float>::max(), maxb = numeric_limits<float>::min();
    for (int i = 0; i < new_message_dim; i++)
    {
    	iter_params[fixed_variable_idx].start = i;
    	iter_params[fixed_variable_idx].step = 1;
    	iter_params[fixed_variable_idx].end = iter_params[fixed_variable_idx].start + 1;

    	vector<int> curr_config;
    	min_sum_factor_single(iter_params, this_node->cost_func
    			    , this_node->observed_data, received_messages
    			    , 0, curr_config, new_message->at(i));

    	if (new_message->at(i) > maxb)
    	{
    		maxb = new_message->at(i);
    	}
    	else if (new_message->at(i) < minb)
    	{
    		minb = new_message->at(i);
    	}
    }

    float sum = 0, range = maxb - minb;
    if (range <= 0.0f)
    {
    	for (int i = 0; i < new_message_dim; i++)
    	{
    		new_message->at(i) = 1;
    		sum += new_message->at(i);
    	}
    }
    else
    {
        range = max(FLOAT_PRECISION, range);
		for (int i = 0; i < new_message_dim; i++)
		{
			new_message->at(i) = (new_message->at(i) - minb) / range;
			sum += new_message->at(i);
		}
    }

	sum = max(sum, FLOAT_PRECISION);
	for (int i = 0; i < new_message_dim; i++)
	{
		new_message->at(i) /= sum;
	}


    tr1::shared_ptr<graph_edge> outgoing_edge;
    for (list<tr1::shared_ptr<graph_edge> >::const_iterator it = next_node->incident_edges.begin()
    	 ; it != next_node->incident_edges.end()
    	 ; it++)
    {
    	if ((*it)->v->node_id == this_node->node_id)
    	{
    		outgoing_edge = *it;
    	}
    }

    outgoing_edge->cool_message = outgoing_edge->hot_message;
    outgoing_edge->hot_message = new_message;

	return 0;
}



int factor_graph::min_sum_factor_single(const vector<iter_param> &iter_params
        , graph_node::factor_func cost_func
        , void *observed_data
        , const vector<tr1::shared_ptr<message> > &received_messages
        , int level
        , vector<int> curr_config
        , float &minimum)
{
	//	if (level < 0 || iter_params.size() != received_messages.size())
	//	{
	//		return -1;
	//	}

	// Get in the innerest loop.
	if (level >= static_cast<int>(iter_params.size()))
	{
		float sum = 0;

		for (int i = 0; i < static_cast<int>(received_messages.size()); i++)
		{
			sum += received_messages[i]->at(curr_config[i]);
		}

		sum += cost_func(curr_config, observed_data);
        minimum = sum < minimum ? sum : minimum;
        return 0;
	}

    iter_param this_level_loop_param = iter_params[level];

    for (int i = this_level_loop_param.start
    	 ; i < this_level_loop_param.end
    	 ; i += this_level_loop_param.step)
    {
    	curr_config.push_back(i);

    	min_sum_factor_single(iter_params, cost_func, observed_data, received_messages, level + 1, curr_config, minimum);

        curr_config.pop_back();
    }

    return 0;
}

int factor_graph::min_sum_variable_message_pass(int v, int w, bool is_sync)
{
//	if (v < 0 || v >= nodes_total_count_
//		|| w < 0 || w >= nodes_total_count_)
//	{
//		return -1;
//	}


	tr1::shared_ptr<graph_node> this_node = nodes_[v]; //variable node
	tr1::shared_ptr<graph_node> next_node = nodes_[w];

//    if (this_node->node_type != VAR_NODE || next_node->node_type != FACTOR_NODE)
//    {
//    	return -2;
//    }

	//Sometimes different variables have different label sets.
	int new_message_dim = this_node->label_set_size;
//    if (new_message_dim < 1)
//    {
//    	return -3;
//    }


	vector<tr1::shared_ptr<message> > received_messages;
	for (list<tr1::shared_ptr<graph_edge> >::const_iterator it = this_node->incident_edges.begin()
		 ; it != this_node->incident_edges.end()
		 ; it++)
	{
		if ((*it)->v->node_id != next_node->node_id)
		{
        	if (!((*it)->cool_message))
			{
				(*it)->cool_message = tr1::shared_ptr<message>(new message((*it)->w->label_set_size, 1.0f / (*it)->w->label_set_size));
			}

			if (!((*it)->hot_message))
			{
				(*it)->hot_message = tr1::shared_ptr<message>(new message((*it)->w->label_set_size, 1.0f / (*it)->w->label_set_size));
			}

			received_messages.push_back(is_sync ? (*it)->cool_message : (*it)->hot_message);
		}
		else
		{
			received_messages.push_back(tr1::shared_ptr<message>(new message(new_message_dim, 0)));
		}
	}

	tr1::shared_ptr<message> new_message(new message(new_message_dim, 0));
	float minb = numeric_limits<float>::max(), maxb = numeric_limits<float>::min();
	float sum = 0;
	for (int i = 0; i < new_message_dim; i++)
	{
		for (vector<tr1::shared_ptr<message> >::const_iterator it = received_messages.begin()
			 ; it != received_messages.end()
			 ; it++)
		{
			new_message->at(i) += (*it)->at(i);
		}

    	if (new_message->at(i) > maxb)
    	{
    		maxb = new_message->at(i);
    	}
    	else if (new_message->at(i) < minb)
    	{
    		minb = new_message->at(i);
    	}
	}

	sum = max(sum, FLOAT_PRECISION);
	for (int i = 0; i < new_message_dim; i++)
	{
		new_message->at(i) /= sum;
	}

//    float sum = 0, range = maxb - minb;
//    if (range <= 0.0f)
//    {
//    	for (int i = 0; i < new_message_dim; i++)
//    	{
//    		new_message->at(i) = 1;
//    		sum += new_message->at(i);
//    	}
//    }
//    else
//    {
//        range = max(FLOAT_PRECISION, range);
//		for (int i = 0; i < new_message_dim; i++)
//		{
//			new_message->at(i) = (new_message->at(i) - minb) / range;
//			sum += new_message->at(i);
//		}
//    }
//
//	sum = max(sum, FLOAT_PRECISION);
//	for (int i = 0; i < new_message_dim; i++)
//	{
//		new_message->at(i) /= sum;
//	}

    tr1::shared_ptr<graph_edge> outgoing_edge;
    for (list<tr1::shared_ptr<graph_edge> >::const_iterator it = next_node->incident_edges.begin()
    	 ; it != next_node->incident_edges.end()
    	 ; it++)
    {
    	if ((*it)->v->node_id == this_node->node_id)
    	{
    		outgoing_edge = *it;
    	}
    }

	outgoing_edge->cool_message = outgoing_edge->hot_message;
	outgoing_edge->hot_message = new_message;

	return 0;
}

int factor_graph::set_node_type(int node_id, int node_type)
{
	if (node_id < 0 || node_id >= nodes_total_count_)
	{
		return -1;
	}

	nodes_[node_id]->node_type = node_type;

	return 0;
}

int factor_graph::set_factor_func(int node_id, graph_node::factor_func func, void *data, graph_node::destructor free_observed_data_dtr)
{
    if (node_id < 0 || node_id >= static_cast<int>(nodes_.size()) || (data != NULL && free_observed_data_dtr == NULL))
    {
    	return -1;
    }


    tr1::shared_ptr<graph_node> this_node = nodes_[node_id];
    this_node->cost_func = func;

    if (this_node->observed_data != NULL)
    {
    	this_node->free_observed_data_destructor(this_node->observed_data);
    	this_node->observed_data = NULL;
    }
    this_node->observed_data = data;
    this_node->free_observed_data_destructor = free_observed_data_dtr;

    return 0;
}

int factor_graph::min_sum_messages_product_on_node(int node_id)
{
    if (node_id < 0 || node_id >= static_cast<int>(nodes_.size()))
    {
    	return -1;
    }

    tr1::shared_ptr<graph_node> this_node = nodes_[node_id];
    if (this_node->node_type != VAR_NODE)
    {
    	return -2;
    }

    this_node->product_of_messages = tr1::shared_ptr<message>(new message(this_node->label_set_size, 0));

	float minb = numeric_limits<float>::max(), maxb = numeric_limits<float>::min();
	for (int i = 0; i < this_node->label_set_size; i++)
	{
		this_node->product_of_messages->at(i) = 0;
		for (list<tr1::shared_ptr<graph_edge> >::iterator it = this_node->incident_edges.begin()
			 ; it != this_node->incident_edges.end()
			 ; it++)
		{
		    this_node->product_of_messages->at(i) += (*it)->hot_message->at(i);
		}

    	if (this_node->product_of_messages->at(i) > maxb)
    	{
    		maxb = this_node->product_of_messages->at(i);
    	}
    	else if (this_node->product_of_messages->at(i) < minb)
    	{
    		minb = this_node->product_of_messages->at(i);
    	}
	}

    float sum = 0, range = maxb - minb;
    if (range <= 0.0f)
    {
    	for (int i = 0; i < this_node->label_set_size; i++)
    	{
    		this_node->product_of_messages->at(i) = 1;
    		sum += this_node->product_of_messages->at(i);
    	}
    }
    else
    {
        range = max(FLOAT_PRECISION, range);
		for (int i = 0; i < this_node->label_set_size; i++)
		{
			this_node->product_of_messages->at(i) = (this_node->product_of_messages->at(i) - minb) / range;
			sum += this_node->product_of_messages->at(i);
		}
    }

	sum = max(sum, FLOAT_PRECISION);

	float min_val = numeric_limits<float>::max();
	for (int i = 0; i < this_node->label_set_size; i++)
	{
		this_node->product_of_messages->at(i) /= sum;
		if (this_node->product_of_messages->at(i) < min_val)
		{
			min_val = this_node->product_of_messages->at(i);
			this_node->best_label = i;
		}
	}


	return 0;
}

int factor_graph::sum_product_messages_product_on_node(int node_id)
{
    if (node_id < 0 || node_id >= static_cast<int>(nodes_.size()))
    {
    	return -1;
    }

    tr1::shared_ptr<graph_node> this_node = nodes_[node_id];
    if (this_node->node_type != VAR_NODE)
    {
    	return -2;
    }

    this_node->product_of_messages = tr1::shared_ptr<message>(new message(this_node->label_set_size, 1.0));

	float minb = numeric_limits<float>::max(), maxb = numeric_limits<float>::min();
	for (int i = 0; i < this_node->label_set_size; i++)
	{
		this_node->product_of_messages->at(i) = 1.0;
		for (list<tr1::shared_ptr<graph_edge> >::iterator it = this_node->incident_edges.begin()
			 ; it != this_node->incident_edges.end()
			 ; it++)
		{
		    this_node->product_of_messages->at(i) *= (*it)->hot_message->at(i);
		}

    	if (this_node->product_of_messages->at(i) > maxb)
    	{
    		maxb = this_node->product_of_messages->at(i);
    	}
    	else if (this_node->product_of_messages->at(i) < minb)
    	{
    		minb = this_node->product_of_messages->at(i);
    	}
	}

    float sum = 0, range = maxb - minb;
    if (range <= 0.0f)
    {
    	for (int i = 0; i < this_node->label_set_size; i++)
    	{
    		this_node->product_of_messages->at(i) = 1;
    		sum += this_node->product_of_messages->at(i);
    	}
    }
    else
    {
        range = max(FLOAT_PRECISION, range);
		for (int i = 0; i < this_node->label_set_size; i++)
		{
			this_node->product_of_messages->at(i) = (this_node->product_of_messages->at(i) - minb) / range;
			sum += this_node->product_of_messages->at(i);
		}
    }

	sum = max(sum, FLOAT_PRECISION);

	float max_val = numeric_limits<float>::min();
	for (int i = 0; i < this_node->label_set_size; i++)
	{
		this_node->product_of_messages->at(i) /= sum;
		if (this_node->product_of_messages->at(i) > max_val)
		{
			max_val = this_node->product_of_messages->at(i);
			this_node->best_label = i;
		}
	}

	return 0;
}

int factor_graph::best_label_on_node(int node_id)
{
    if (node_id < 0 || node_id >= static_cast<int>(nodes_.size()))
    {
    	return -1;
    }

    return nodes_[node_id]->best_label;
}

int factor_graph::set_label_set_size(int node_id, int size)
{
    if (node_id < 0 || node_id >= static_cast<int>(nodes_.size()))
    {
    	return -1;
    }

    tr1::shared_ptr<graph_node> this_node = nodes_[node_id];
    this_node->label_set_size = size;

    return 0;
}



