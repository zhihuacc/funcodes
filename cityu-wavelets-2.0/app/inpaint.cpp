#include "../include/inpaint.h"
#include <cmath>
#include <algorithm>

using namespace std;

int figure_good_sigmas(double est_sigma, double ratio, int phase1_num, int phase2_num, vector<double> &sigmas)
{
	double lambda = std::max<double>(est_sigma - (1 - ratio) * (1 - ratio) * est_sigma / 2.0, 1);
	double left = log10(512), right = log10(std::max<double>(2 * lambda + 10, 20));

	vector<double> pts;
	log10space(left, right, phase1_num, pts);
	sigmas = pts;

	left = log10(std::max<double>(15, 2 * lambda));
	right = log10(lambda);
	pts.clear();
	log10space(left, right, phase2_num, pts);
	sigmas.insert(sigmas.end(), pts.begin(), pts.end());

	return 0;
}
