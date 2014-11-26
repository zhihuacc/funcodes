#include "../include/tensor.h"

#include <iostream>
#include <iomanip>
using namespace std;

void print(const Tensor &tensor, int n, int *idx)
{
	cout << "[";
	for (int i = 0; i < n; ++i)
	{
		cout << idx[i];
		if (i != n - 1)
		{
			cout << ", ";
		}
	}
	cout << "] = " << tensor._data_mat.at<Vec2d>(idx)[0]
	               << ", "
	               << tensor._data_mat.at<Vec2d>(idx)[1] << endl;
}

void print(const Tensor &tensor, const string &filename = "cout")
{
	int max_term = 8;
	Mat mat = tensor._data_mat;
	int *pos = new int[mat.dims];
	const int *range = mat.size;;
	int dims = mat.dims;

	for (int i = 0; i < dims; i++)
	{
		pos[i] = 0;
	}

	cout << setiosflags(ios::fixed) << setprecision(4);
	int i = dims - 1;
	int t = 0;
	while(true)
	{
		while (i >= 0 && pos[i] >= range[i])
		{
			pos[i] = 0;
			--i;

			if (i >= 0)
			{
				++pos[i];
				continue;
			}
		}

		if (i < 0)
		{
			cout << endl;
			break;
		}

		if (pos[dims - 1] == 0){
			cout << endl << "[";
			for (int idx = 0; idx < dims; ++idx)
			{
				cout << pos[idx];
				if (idx < dims - 1)
				{
					cout << ",";
				}
				else
				{
					cout << "]";
				}
			}
			cout << endl;
		}


		cout << "(" << mat.at<Vec2d>(pos)[0] << ","
				    << mat.at<Vec2d>(pos)[1] << ")";
		if (t < max_term - 1)
		{
			cout << " ";
		}
		else if(t == max_term - 1)
		{
			cout << endl;
		}
		++t;
		t %= max_term;

		i = mat.dims - 1;
		++pos[i];
	}

	delete [] pos;

}
