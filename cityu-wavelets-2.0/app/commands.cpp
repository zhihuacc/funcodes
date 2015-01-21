#include "../include/mat_toolbox.h"

using namespace std;

int cvtxml_entry(const string &fn)
{
	typedef double FLOAT_TYPE;

	Media_Format mfmt;
	Mat_<Vec<FLOAT_TYPE, 2> > mat;
	load_as_tensor<FLOAT_TYPE>(fn, mat, &mfmt);

	string xml = fn + ".xml";
	save_as_media<FLOAT_TYPE>(xml, mat, &mfmt);

	return 0;
}

int psnr_entry(const string &left, const string &right)
{
	typedef double FLOAT_TYPE;

	Mat_<Vec<FLOAT_TYPE, 2> > left_mat, right_mat;
	Media_Format mfmt;
	load_as_tensor<FLOAT_TYPE>(left, left_mat, &mfmt);
	load_as_tensor<FLOAT_TYPE>(right, right_mat, &mfmt);

	double score, msr;
	psnr<FLOAT_TYPE>(left_mat, right_mat, score, msr);

	cout << "PSNR: " << score << ", MSR: " << msr << endl;

	return 0;
}
