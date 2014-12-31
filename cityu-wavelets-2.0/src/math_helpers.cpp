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

int hcf_lcd(int n, int m, int &hcf, int &lcd)
{
	if (n < m)
	{
		int tmp = n;
		n = m;
		m = tmp;
	}

	int p = n * m;
	while (m != 0)
	{
		int r = n % m;
		n = m;
		m = r;
	}

	hcf = n;
	lcd = p / n;

	return 0;
}
