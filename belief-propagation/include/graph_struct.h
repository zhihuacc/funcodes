#ifndef GRAPH_STRUCT_H
#define GRAPH_STRUCT_H

#include <tr1/memory>
#include <list>
#include <vector>

using namespace std;


struct graph_node;

typedef vector<float> message;

//edge type specified for factor graph
struct graph_edge
{
tr1::shared_ptr<graph_node> v;
tr1::shared_ptr<graph_node> w;
int order;
tr1::shared_ptr<message> hot_message;
tr1::shared_ptr<message> cool_message;
};

enum
{
	VAR_NODE = 0,
	FACTOR_NODE,
};


//node specified for factor graph
struct graph_node
{
int node_id;
list<tr1::shared_ptr<graph_edge> > incident_edges;

typedef float (*factor_func)(const vector<int> &labels, void *observed_data);
factor_func cost_func; // data-cost, smoothness cost, higher-order cost

void *observed_data;   // used by data-cost; NULL for other cost
typedef void (*destructor)(void *data);
destructor free_observed_data_destructor;

int node_type;         // 0-variable node; 1-factor node
int label_set_size;    // label set size for variable noe; 0 for factor node

tr1::shared_ptr<message> product_of_messages;
int best_label;

graph_node():node_id(-1), cost_func(NULL), observed_data(NULL)
             , free_observed_data_destructor(NULL), node_type(-1), label_set_size(0), best_label(-1) {}
~graph_node(){ if (observed_data != NULL)  free_observed_data_destructor(observed_data); }


};

//template <typename node_type, typename edge_type>
class sparse_graph
{
protected:
	vector<tr1::shared_ptr<graph_node> > nodes_;
	int is_directed_;
	int nodes_total_count_;
	int edges_total_count_;

public:
	sparse_graph(int vn, int is_directed);
    int insert_edge(int v, int w);
    int remove_edge(int v, int w);
    bool edge_exists(int v, int w);
    int nodes_count(int node_type);
    int edges_count();
    tr1::shared_ptr<graph_edge> edge_of(int v, int w);

    class adj_iterator
    {
    private:
    	const sparse_graph &graph_;
    	int w_;
    	list<tr1::shared_ptr<graph_edge> >::iterator it_;
    public:
    	adj_iterator(const sparse_graph &graph, int v);
    	int begin();
    	int next();
    	bool end();
    };
};

//typedef sparse_graph<graph_node, graph_edge> graph_topology;
typedef sparse_graph graph_topology;

#endif
