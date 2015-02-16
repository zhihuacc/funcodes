#include "../include/mat_toolbox.h"

bool sameSize(const Mat &left, const Mat &right)
{
	SmartIntArray sl(left.dims, left.size), sr(right.dims, right.size);
	bool ret = sl == sr;
	if (ret == false)
	{
		cout << "Failed in sameSize " << endl;
	}

	return ret;
}

bool isGoodMat(const Mat &domain)
{
	if (domain.empty() || !domain.isContinuous()
		|| (domain.depth() != CV_64F && domain.depth() != CV_32F))
	{
		cout << "Failed in isGoodMat " << endl;
		return false;
	}

	return true;
}

//double mat_error(const Mat &left, const Mat &right, const Mat &mask)
//{
//	Mat diff = left - right;
//
//	multiply(diff, mask, diff);
//	double err = norm(diff, NORM_L2);
////	err /= diff.total();
//	return err;
//}
int log10space(double left, double right, int n, vector<double> &points)
{
	if (n < 2)
	{
		return -1;
	}
	double interval = (right - left) / (n - 1);
	points.reserve(n);
	points.resize(n);
	for (int i = 0; i < n; ++i)
	{
		points[i] = pow(10, left + i * interval);
	}
	return 0;
}
