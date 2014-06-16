#include "../include/graph_struct.h"

sparse_graph::sparse_graph(int vn, int is_directed)
: is_directed_(is_directed), nodes_total_count_(vn), edges_total_count_(0)
{
	nodes_.reserve(nodes_total_count_);
	nodes_.resize(nodes_total_count_);

	for (int i = 0; i < nodes_total_count_; i++)
	{
        nodes_[i] = tr1::shared_ptr<graph_node>(new graph_node());
        nodes_[i]->node_id = i;
	}
}

int sparse_graph::insert_edge(int v, int w)
{
	if (v < 0 || v >= nodes_total_count_
		|| w < 0 || w >= nodes_total_count_
		|| v == w)
	{
		return -1;
	}

    tr1::shared_ptr<graph_edge> edge(new graph_edge());
    edge->v = nodes_[v];
    edge->w = nodes_[w];

    nodes_[w]->incident_edges.insert(nodes_[w]->incident_edges.end(), edge); //push_back

    if (!is_directed_)
    {
    	tr1::shared_ptr<graph_edge> edge(new graph_edge());
		edge->v = nodes_[w];
		edge->w = nodes_[v];

		nodes_[v]->incident_edges.insert(nodes_[v]->incident_edges.end(), edge); //push_back
    }

    edges_total_count_++;

	return 0;
}

int sparse_graph::remove_edge(int v, int w)
{
	if (v < 0 || v >= nodes_total_count_
		|| w < 0 || w >= nodes_total_count_
		|| v == w)
	{
		return -1;
	}

	tr1::shared_ptr<graph_node> v_node = nodes_[v];

	for (list<tr1::shared_ptr<graph_edge> >::iterator it = v_node->incident_edges.begin()
		; it != v_node->incident_edges.end()
		;)
	{
        if ((*it)->w->node_id == w)
        {
        	it = v_node->incident_edges.erase(it);
        }
        else
        {
        	it++;
        }
	}

	if (!is_directed_)
	{
		tr1::shared_ptr<graph_node> v_node = nodes_[w];

		for (list<tr1::shared_ptr<graph_edge> >::iterator it = v_node->incident_edges.begin()
			; it != v_node->incident_edges.end()
			;)
		{
	        if ((*it)->w->node_id == v)
	        {
	        	it = v_node->incident_edges.erase(it);
	        }
	        else
	        {
	        	it++;
	        }
		}
	}

	edges_total_count_--;

	return 0;
}

int sparse_graph::nodes_count(int node_type)
{
	int cnt = 0;
	if (node_type == -1)
	{
		return cnt = nodes_total_count_;
	}

	for (vector<tr1::shared_ptr<graph_node> >::iterator it = nodes_.begin()
		; it != nodes_.end()
		; it++)
	{
		if (node_type != -1 && (*it)->node_type == node_type)
        {
        	cnt++;
        }
	}

	return cnt;
}

int sparse_graph::edges_count()
{
	return edges_total_count_;
}

bool sparse_graph::edge_exists(int v, int w)
{
	bool ret = false;
	if (v < 0 || v >= nodes_total_count_
		|| w < 0 || w >= nodes_total_count_
		|| v == w)
	{
		return ret;
	}


	tr1::shared_ptr<graph_node> v_node = nodes_[w];

	for (list<tr1::shared_ptr<graph_edge> >::iterator it = v_node->incident_edges.begin()
		; it != v_node->incident_edges.end()
		;)
	{
        if ((*it)->w->node_id == v)
        {
        	ret = true;
        	break;
        }
	}

	return ret;
}



sparse_graph::adj_iterator::adj_iterator(const sparse_graph &graph, int w): graph_(graph)
, w_(w), it_(graph_.nodes_[w_]->incident_edges.begin())
{
}

int sparse_graph::adj_iterator::begin()
{
    it_ = graph_.nodes_[w_]->incident_edges.begin();

	if (it_ != graph_.nodes_[w_]->incident_edges.end())
	{
        return (*it_)->v->node_id;
	}

    return -1;
}

int sparse_graph::adj_iterator::next()
{
	it_++;
	if (it_ != graph_.nodes_[w_]->incident_edges.end())
	{
        return (*it_)->v->node_id;
	}

	return -1;
}

bool sparse_graph::adj_iterator::end()
{
	return it_ == graph_.nodes_[w_]->incident_edges.end() ? true : false;
}

