#include "../include/math_helpers.h"

int nchoosek(int n, int k)
{
	if (n < 0 || k < 0 || k > n)
	{
		return -1;
	}

	if (n == k)
	{
		return 1;
	}

	int n0 = n;
	int num = 1;
	for (int i = 0; i < k; ++i)
	{
		num *= n0;
		--n0;
	}

	for (int k0 = k; k0 > 0; --k0)
	{
		num /= k0;
	}

	return num;
}
