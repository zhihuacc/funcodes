#include "../include/dt_cwt.h"
#include "../include/utils.h"

double Pm(double x, int m)
{
	if ( x <= 0.0 || x >= 1.0 || m < 1)
	{
		return -1;
	}

	double sum = 0.0;
	for (int j = 0; j < m; ++j)
	{
		sum += nchoosek(m - 1 + j, j) * pow(x, j);
	}

	sum *= pow(1 - x, m);
	return sum;
}

int construct_dt_cwt_filter(const DT_CWT_Param &param, Tensor &filter0, Tensor &filter1)
{
	DT_CWT_Param param0 = param;

	if (param0.cl > param0.cr)
	{
		param0.cl -= param0.T;
	}

	if (param0.cr - param0.cl < param0.epl + param0.epr)
	{
		return -1;
	}

	Mat mat(2, (int[]){1, param0.n}, CV_64FC2, Scalar(0, 0));
	Mat mat2 = mat.clone();

	double delta = param0.T / param0.n;
	double w0 = param0.lend - param0.T;
	for (int i = 0; i < 3 * param0.n; ++i)
	{
		double w = w0 + i * delta;
		double f = 0.0;
		if (w <= (param0.cl - param0.epl) || w >= (param0.cr + param0.epr))
		{
			f = 0.0;
		}
		else if (w >= (param0.cl + param0.epl) && w <= (param0.cr - param0.epr))
		{
			f = 1.0;
		}
		else if (w > (param0.cl - param0.epl) && w < (param0.cl + param0.epl))
		{
			double r = (param0.cl + param0.epl - w) / (2 * param0.epl);
			f = sin(0.5 * M_PI * Pm(r, param0.m));
		}
		else if (w > (param0.cr - param0.epr) && w < (param0.cr + param0.epr))
		{
			double r = (w - param0.cr + param0.epr) / (2 * param0.epr);
			f = sin(0.5 * M_PI * Pm(r, param0.m));
		}

		mat.at<Vec2d>(0, i % param0.n)[0] += f;
		mat.at<Vec2d>(0, i % param0.n)[1] += 0;

	}

	for (int i = 0; i < param0.n; ++i)
	{
		complex<double> a(mat.at<Vec2d>(0, i)[0], mat.at<Vec2d>(0, i)[1]);
		double w = -M_PI + i * 2 * M_PI / param0.n;
		complex<double> b(cos(-w), sin(-w));
		a *= b;
		if (i < param0.n / 2)
		{
			mat2.at<Vec2d>(0, i + param0.n / 2)[0] = a.real();
			mat2.at<Vec2d>(0, i + param0.n / 2)[1] = a.imag();
		}
		else
		{
			mat2.at<Vec2d>(0, i - param0.n / 2)[0] = a.real();
			mat2.at<Vec2d>(0, i - param0.n / 2)[1] = a.imag();
		}
	}

	filter0 = Tensor(1, mat);
	filter1 = Tensor(1, mat2);

	return 0;
}
