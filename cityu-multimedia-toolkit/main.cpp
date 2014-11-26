#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "include/tensor.h"
#include "include/loaders.h"
#include "include/filter_banks.h"
#include "include/dt_cwt.h"
#include "include/debug.h"
#include "include/utils.h"

#include "c-style/include/wavelets_tools.h"
#include "unit_test.h"

#include <iostream>
#include <sstream>

using namespace cv;

double Pm(double x, int m);


int main(int argc, char **argv)
{

	Unit_Test unit;
	unit.mat_extension(argc, argv);

	return 0;
	//
	double d1 = Pm(0.1341, 3);
	double d2 = Pm(1 - 0.1341, 3);
	cout << d1 << ", " << d2 << ", " << d1 + d2 << endl;

	// Case 7
	Tensor before(2, (int[]){1, 64});
	before.set_value((int[]){0, 0}, (double[]){1,0});
	before.set_value((int[]){0, 1}, (double[]){2,0});
	before.set_value((int[]){0, 2}, (double[]){1,0});
	before.set_value((int[]){0, 3}, (double[]){2,0});
	before.set_value((int[]){0, 4}, (double[]){3,0});
	before.set_value((int[]){0, 5}, (double[]){2,0});
	before.set_value((int[]){0, 6}, (double[]){1,0});
	before.set_value((int[]){0, 7}, (double[]){2,0});
	before.set_value((int[]){0, 8}, (double[]){7,0});
	before.set_value((int[]){0, 9}, (double[]){3,0});
	before.set_value((int[]){0, 10}, (double[]){1,0});
	before.set_value((int[]){0, 11}, (double[]){5,0});
	before.set_value((int[]){0, 12}, (double[]){8,0});
	before.set_value((int[]){0, 13}, (double[]){9,0});
	before.set_value((int[]){0, 14}, (double[]){13,0});
	before.set_value((int[]){0, 15}, (double[]){12,0});
	before.set_value((int[]){0, 16}, (double[]){1,0});
	before.set_value((int[]){0, 17}, (double[]){2,0});
	before.set_value((int[]){0, 18}, (double[]){1,0});
	before.set_value((int[]){0, 19}, (double[]){2,0});
	before.set_value((int[]){0, 20}, (double[]){3,0});
	before.set_value((int[]){0, 21}, (double[]){2,0});
	before.set_value((int[]){0, 22}, (double[]){1,0});
	before.set_value((int[]){0, 23}, (double[]){2,0});
	before.set_value((int[]){0, 24}, (double[]){1,0});
	before.set_value((int[]){0, 25}, (double[]){2,0});
	before.set_value((int[]){0, 26}, (double[]){1,0});
	before.set_value((int[]){0, 27}, (double[]){2,0});
	before.set_value((int[]){0, 28}, (double[]){3,0});
	before.set_value((int[]){0, 29}, (double[]){2,0});
	before.set_value((int[]){0, 30}, (double[]){1,0});
	before.set_value((int[]){0, 31}, (double[]){2,0});
	before.set_value((int[]){0, 32}, (double[]){1,0});
	before.set_value((int[]){0, 33}, (double[]){2,0});
	before.set_value((int[]){0, 34}, (double[]){1,0});
	before.set_value((int[]){0, 35}, (double[]){2,0});
	before.set_value((int[]){0, 36}, (double[]){3,0});
	before.set_value((int[]){0, 37}, (double[]){2,0});
	before.set_value((int[]){0, 38}, (double[]){1,0});
	before.set_value((int[]){0, 39}, (double[]){2,0});
	before.set_value((int[]){0, 40}, (double[]){1,0});
	before.set_value((int[]){0, 41}, (double[]){2,0});
	before.set_value((int[]){0, 42}, (double[]){1,0});
	before.set_value((int[]){0, 43}, (double[]){2,0});
	before.set_value((int[]){0, 44}, (double[]){3,0});
	before.set_value((int[]){0, 45}, (double[]){2,0});
	before.set_value((int[]){0, 46}, (double[]){1,0});
	before.set_value((int[]){0, 47}, (double[]){2,0});
	before.set_value((int[]){0, 48}, (double[]){1,0});
	before.set_value((int[]){0, 49}, (double[]){2,0});
	before.set_value((int[]){0, 50}, (double[]){1,0});
	before.set_value((int[]){0, 51}, (double[]){2,0});
	before.set_value((int[]){0, 52}, (double[]){3,0});
	before.set_value((int[]){0, 53}, (double[]){2,0});
	before.set_value((int[]){0, 54}, (double[]){1,0});
	before.set_value((int[]){0, 55}, (double[]){2,0});
	before.set_value((int[]){0, 56}, (double[]){1,0});
	before.set_value((int[]){0, 57}, (double[]){2,0});
	before.set_value((int[]){0, 58}, (double[]){1,0});
	before.set_value((int[]){0, 59}, (double[]){2,0});
	before.set_value((int[]){0, 60}, (double[]){3,0});
	before.set_value((int[]){0, 61}, (double[]){2,0});
	before.set_value((int[]){0, 62}, (double[]){1,0});
	before.set_value((int[]){0, 63}, (double[]){2,0});

	Coef_Set coefs;
	double rt = sqrt(2);
	Tensor lo_de(2, (int[]){1, 64}), hi_de(2, (int[]){1, 64}), lo_re(2, (int[]){1, 64}), hi_re(2, (int[]){1, 64});
	lo_de.set_value((int[]){0, 0}, (double[]){1 / rt, 0});
	lo_de.set_value((int[]){0, 1}, (double[]){1 / rt, 0});
	hi_de.set_value((int[]){0, 0}, (double[]){1 / rt, 0});
	hi_de.set_value((int[]){0, 1}, (double[]){-1 / rt, 0});

	Tensor ihaarlow, ihaarhigh;
	lo_de.dft(ihaarlow);
	hi_de.dft(ihaarhigh);
	cout << "Haar Freq: --" << endl;
	print(ihaarlow);
	print(ihaarhigh);


//	lo_de.set_value((int[]){0, 0}, (double[]){0.0352, 0});
//	lo_de.set_value((int[]){0, 1}, (double[]){-0.0854, 0});
//	lo_de.set_value((int[]){0, 2}, (double[]){-0.1350, 0});
//	lo_de.set_value((int[]){0, 3}, (double[]){0.4599, 0});
//	lo_de.set_value((int[]){0, 4}, (double[]){0.8069, 0});
//	lo_de.set_value((int[]){0, 5}, (double[]){0.3327, 0});
//	hi_de.set_value((int[]){0, 0}, (double[]){-0.3327, 0});
//	hi_de.set_value((int[]){0, 1}, (double[]){0.8069, 0});
//	hi_de.set_value((int[]){0, 2}, (double[]){-0.4599, 0});
//	hi_de.set_value((int[]){0, 3}, (double[]){-0.1350, 0});
//	hi_de.set_value((int[]){0, 4}, (double[]){0.0854, 0});
//	hi_de.set_value((int[]){0, 5}, (double[]){0.0352, 0});



	Tensor filters[2] = {lo_de, hi_de};
	Filter_Bank laar_bank(2, filters);

	laar_bank.decompose(before, coefs, 2);


	Tensor rec;
	laar_bank.reconstruct(coefs, rec, 2);

	cout << "Haar Result: --" << endl;
	rec.psnr(before);
	cout << "Before: --" << endl;
	print(before);
	cout << "Rec: --" << endl;
	print(rec);
	before.psnr(rec);




	DT_CWT_Param param, param0, param1;
//	param.cl = -0.3;
//	param.cr = 0.9;
//	param.epl = 0.2;
//	param.epr = 0.2;
//	param.lend = -1;
//	param.rend = 1;
//	param.T = 2 * 1;
//	param.m = 1;
//	param.n = 20;
//
//	param.cl = 33.0/32.0;
//	param.cr = M_PI;
//	param.epl = 69.0 / 128.0;
//	param.epr = 51.0 / 512.0;
//	param.lend = -M_PI;
//	param.rend = M_PI;
//	param.T = 2 * M_PI;
//	param.m = 1;
//	param.n = 20;

	param0.cl = -M_PI / 3;
	param0.cr = M_PI / 3;
	param0.epl = M_PI / 8;
	param0.epr = M_PI / 8;
	param0.lend = -M_PI;
	param0.rend = M_PI;
	param0.T = 2 * M_PI;
	param0.m = 1;
	param0.n = 64;

	param1.cl = M_PI / 3;
	param1.cr = 5 * M_PI / 3;
	param1.epl = M_PI / 8;
	param1.epr = M_PI / 8;
	param1.lend = -M_PI;
	param1.rend = M_PI;
	param1.T = 2 * M_PI;
	param1.m = 1;
	param1.n = 64;

	cout << "Self-construct wavelet: --" << endl;
	Tensor f0, f1, ilow, ihigh;
	construct_dt_cwt_filter(param0, ilow, f1);
	construct_dt_cwt_filter(param1, ihigh, f1);
	cout << "Freq-domain: " << endl;
	print(ilow);
	print(ihigh);

	Tensor low, high;
	ilow.scale(complex<double>(rt, 0), complex<double>(0,0), ilow);
	ihigh.scale(complex<double>(rt, 0), complex<double>(0,0), ihigh);
	cout << "Scaled: --" << endl;
	print(ilow);
	print(ihigh);


	ilow.idft(low);
	low.dft(ilow);
	ihigh.idft(high);
	high.dft(ihigh);
	cout << "Show Filter Freq after IDFT->DFT: --" << endl;
	print(ilow);
	print(ihigh);

	cout << "Show Time-Domain Filter: --" << endl;
	print(low);
	cout << "low norm: " << low.l2norm() << endl;
	print(high);
	cout << "high norm: " << high.l2norm() << endl;




	Filter_Bank novel(2, (Tensor[]){low, high});

	cout << "Before2: --" <<endl;
	print(before);
	coefs.clear();
	novel.decompose(before, coefs, 2);

	Tensor rec2;
	novel.reconstruct(coefs, rec2, 2);

	cout << "Rec2: ----" << endl;
	print(rec2);

	return 0;


//	// Case 6
//	Tensor dec_lo0, dec_hi1, dec_hi2, dec_hi3;
//	dec_lo0 = Tensor(2, (int[]){256, 256});
//	dec_hi1 = Tensor(2, (int[]){256, 256});
//	dec_hi2 = Tensor(2, (int[]){256, 256});
//	dec_hi3 = Tensor(2, (int[]){256, 256});
//
//	dec_lo0.set_value((int[]){0, 0}, (double[]){1.0 / 2, 0});
//	dec_lo0.set_value((int[]){0, 1}, (double[]){1.0 / 2, 0});
//	dec_lo0.set_value((int[]){1, 0}, (double[]){1.0 / 2, 0});
//	dec_lo0.set_value((int[]){1, 1}, (double[]){1.0 / 2, 0});
//
//	dec_hi1.set_value((int[]){0, 0}, (double[]){1.0 / 2, 0});
//	dec_hi1.set_value((int[]){0, 1}, (double[]){-1.0 / 2, 0});
//	dec_hi1.set_value((int[]){1, 0}, (double[]){1.0 / 2, 0});
//	dec_hi1.set_value((int[]){1, 1}, (double[]){-1.0 / 2, 0});
//
//	dec_hi2.set_value((int[]){0, 0}, (double[]){1.0 / 2, 0});
//	dec_hi2.set_value((int[]){0, 1}, (double[]){1.0 / 2, 0});
//	dec_hi2.set_value((int[]){1, 0}, (double[]){-1.0 / 2, 0});
//	dec_hi2.set_value((int[]){1, 1}, (double[]){-1.0 / 2, 0});
//
//	dec_hi3.set_value((int[]){0, 0}, (double[]){1.0 / 2, 0});
//	dec_hi3.set_value((int[]){0, 1}, (double[]){-1.0 / 2, 0});
//	dec_hi3.set_value((int[]){1, 0}, (double[]){-1.0 / 2, 0});
//	dec_hi3.set_value((int[]){1, 1}, (double[]){1.0 / 2, 0});
//
////	Tensor rec_lo0, rec_hi1, rec_hi2, rec_hi3;
////	rec_lo0 = Tensor(2, (int[]){256, 256});
////	rec_hi1 = Tensor(2, (int[]){256, 256});
////	rec_hi2 = Tensor(2, (int[]){256, 256});
////	rec_hi3 = Tensor(2, (int[]){256, 256});
////
////	rec_lo0.set_value((int[]){0, 0}, (double[]){1.0 / 2, 0});
////	rec_lo0.set_value((int[]){0, 255}, (double[]){1.0 / 2, 0});
////	rec_lo0.set_value((int[]){255, 0}, (double[]){1.0 / 2, 0});
////	rec_lo0.set_value((int[]){255, 255}, (double[]){1.0 / 2, 0});
////
////	rec_hi1.set_value((int[]){0, 0}, (double[]){1.0 / 2, 0});
////	rec_hi1.set_value((int[]){0, 255}, (double[]){-1.0 / 2, 0});
////	rec_hi1.set_value((int[]){255, 0}, (double[]){1.0 / 2, 0});
////	rec_hi1.set_value((int[]){255, 255}, (double[]){-1.0 / 2, 0});
////
////	rec_hi2.set_value((int[]){0, 0}, (double[]){1.0 / 2, 0});
////	rec_hi2.set_value((int[]){0, 255}, (double[]){1.0 / 2, 0});
////	rec_hi2.set_value((int[]){255, 0}, (double[]){-1.0 / 2, 0});
////	rec_hi2.set_value((int[]){255, 255}, (double[]){-1.0 / 2, 0});
////
////	rec_hi3.set_value((int[]){0, 0}, (double[]){1.0 / 2, 0});
////	rec_hi3.set_value((int[]){0, 255}, (double[]){-1.0 / 2, 0});
////	rec_hi3.set_value((int[]){255, 0}, (double[]){-1.0 / 2, 0});
////	rec_hi3.set_value((int[]){255, 255}, (double[]){1.0 / 2, 0});
//
//	Tensor filters[8] = {dec_hi1, dec_hi2, dec_hi3, dec_lo0};
//	Filter_Bank laar_bank(4, filters);
//
//	Tensor before;
//	Media_Format mfmt;
//	load_as_tensor("./test2.jpg", before, &mfmt);
//	save_as_media("./test2-gray.jpg", before, &mfmt);
//
//	Coef_Set coefs;
//	laar_bank.decompose(before, coefs, 3);
//
//
//	Tensor rec;
//	laar_bank.reconstruct(coefs, rec, 3);
//	save_as_media("./test2-reconstructed.jpg", rec, &mfmt);
//
//	rec.psnr(before);
//
//	return 0;

//	// Case 5
//	Tensor before(1, (int[]){8});
//	before.set_value((int[]){0, 0}, (double[]){1,0});
//	before.set_value((int[]){0, 1}, (double[]){2,0});
//	before.set_value((int[]){0, 2}, (double[]){1,0});
//	before.set_value((int[]){0, 3}, (double[]){2,0});
//	before.set_value((int[]){0, 4}, (double[]){3,0});
//	before.set_value((int[]){0, 5}, (double[]){2,0});
//	before.set_value((int[]){0, 6}, (double[]){1,0});
//	before.set_value((int[]){0, 7}, (double[]){2,0});
//
//	Tensor lo_de(1, (int[]){8}), hi_de(1, (int[]){8}), lo_re(1, (int[]){8}), hi_re(1, (int[]){8});
//	lo_de.set_value((int[]){0, 0}, (double[]){1, 0});
//	lo_de.set_value((int[]){0, 1}, (double[]){1, 0});
//	hi_de.set_value((int[]){0, 0}, (double[]){1, 0});
//	hi_de.set_value((int[]){0, 1}, (double[]){-1, 0});
//
//
//	lo_re.set_value((int[]){0, 0}, (double[]){1, 0});
//	lo_re.set_value((int[]){0, 7}, (double[]){1, 0});
//	hi_re.set_value((int[]){0, 0}, (double[]){1, 0});
//	hi_re.set_value((int[]){0, 7}, (double[]){-1, 0});
//
//	Tensor ch1, ch2;
//	before.conv(lo_de, ch1);
//	before.conv(hi_de, ch2);
//	std::cout << "Lo channel: " << std::endl;
//	ch1.print((int[]){0, 0});
//	ch1.print((int[]){0, 1});
//	ch1.print((int[]){0, 2});
//	ch1.print((int[]){0, 3});
//	ch1.print((int[]){0, 4});
//	ch1.print((int[]){0, 5});
//	ch1.print((int[]){0, 6});
//	ch1.print((int[]){0, 7});
//	std::cout << "Hi channel: " << std::endl;
//	ch2.print((int[]){0, 0});
//	ch2.print((int[]){0, 1});
//	ch2.print((int[]){0, 2});
//	ch2.print((int[]){0, 3});
//	ch2.print((int[]){0, 4});
//	ch2.print((int[]){0, 5});
//	ch2.print((int[]){0, 6});
//	ch2.print((int[]){0, 7});
//
//	int size[2];
//	ch1.downsample(0, size, ch1);
//	ch1.upsample(0, size, ch1);
//	ch2.downsample(0, size, ch2);
//	ch2.upsample(0, size, ch2);
//
//	std::cout << "UD Lo channel: " << std::endl;
//	ch1.print((int[]){0, 0});
//	ch1.print((int[]){0, 1});
//	ch1.print((int[]){0, 2});
//	ch1.print((int[]){0, 3});
//	ch1.print((int[]){0, 4});
//	ch1.print((int[]){0, 5});
//	ch1.print((int[]){0, 6});
//	ch1.print((int[]){0, 7});
//	std::cout << "UD Hi channel: " << std::endl;
//	ch2.print((int[]){0, 0});
//	ch2.print((int[]){0, 1});
//	ch2.print((int[]){0, 2});
//	ch2.print((int[]){0, 3});
//	ch2.print((int[]){0, 4});
//	ch2.print((int[]){0, 5});
//	ch2.print((int[]){0, 6});
//	ch2.print((int[]){0, 7});
//
//	Tensor ch1r, ch2r;
//	ch1.conv(lo_re, ch1r);
//	ch2.conv(hi_re, ch2r);
//
//	std::cout << "re Lo channel: " << std::endl;
//	ch1r.print((int[]){0, 0});
//	ch1r.print((int[]){0, 1});
//	ch1r.print((int[]){0, 2});
//	ch1r.print((int[]){0, 3});
//	ch1r.print((int[]){0, 4});
//	ch1r.print((int[]){0, 5});
//	ch1r.print((int[]){0, 6});
//	ch1r.print((int[]){0, 7});
//	std::cout << "re Hi channel: " << std::endl;
//	ch2r.print((int[]){0, 0});
//	ch2r.print((int[]){0, 1});
//	ch2r.print((int[]){0, 2});
//	ch2r.print((int[]){0, 3});
//	ch2r.print((int[]){0, 4});
//	ch2r.print((int[]){0, 5});
//	ch2r.print((int[]){0, 6});
//	ch2r.print((int[]){0, 7});
//
//	ch1r.pw_add(ch2r, ch1r);
//	std::cout << "Rec: " << std::endl;
//	ch1r.print((int[]){0, 0});
//	ch1r.print((int[]){0, 1});
//	ch1r.print((int[]){0, 2});
//	ch1r.print((int[]){0, 3});
//	ch1r.print((int[]){0, 4});
//	ch1r.print((int[]){0, 5});
//	ch1r.print((int[]){0, 6});
//	ch1r.print((int[]){0, 7});
//	return 0;




	// Case 4
//	Tensor before;
//	Media_Format mfmt;
//	load_as_tensor("./test2.jpg", before, &mfmt);
//	save_as_media("./test2-gray.jpg", before, &mfmt);
//
//
//	before.print((int[]){128,128});
//	before.print((int[]){129,128});
//	before.print((int[]){130,128});
//	before.print((int[]){131,128});
//	before.print((int[]){128,128});
//	before.print((int[]){128,129});
//	before.print((int[]){128,130});
//	before.print((int[]){128,131});
//	before.print((int[]){200,128});
//	before.print((int[]){200,129});
//	before.print((int[]){200,130});
//	before.print((int[]){200,131});
//
//	Tensor lo0, hi1, hi2, hi3, lo0_res, hi1_res, hi2_res, hi3_res;
//	lo0 = Tensor(2, (int[]){256, 256});
//	hi1 = Tensor(2, (int[]){256, 256});
//	hi2 = Tensor(2, (int[]){256, 256});
//	hi3 = Tensor(2, (int[]){256, 256});
//
//	lo0.set_value((int[]){0, 0}, (double[]){1.0 / 2, 0});
//	lo0.set_value((int[]){0, 1}, (double[]){1.0 / 2, 0});
//	lo0.set_value((int[]){1, 0}, (double[]){1.0 / 2, 0});
//	lo0.set_value((int[]){1, 1}, (double[]){1.0 / 2, 0});
//
//	hi1.set_value((int[]){0, 0}, (double[]){1.0 / 2, 0});
//	hi1.set_value((int[]){0, 1}, (double[]){-1.0 / 2, 0});
//	hi1.set_value((int[]){1, 0}, (double[]){1.0 / 2, 0});
//	hi1.set_value((int[]){1, 1}, (double[]){-1.0 / 2, 0});
//
//	hi2.set_value((int[]){0, 0}, (double[]){1.0 / 2, 0});
//	hi2.set_value((int[]){0, 1}, (double[]){1.0 / 2, 0});
//	hi2.set_value((int[]){1, 0}, (double[]){-1.0 / 2, 0});
//	hi2.set_value((int[]){1, 1}, (double[]){-1.0 / 2, 0});
//
//	hi3.set_value((int[]){0, 0}, (double[]){1.0 / 2, 0});
//	hi3.set_value((int[]){0, 1}, (double[]){-1.0 / 2, 0});
//	hi3.set_value((int[]){1, 0}, (double[]){-1.0 / 2, 0});
//	hi3.set_value((int[]){1, 1}, (double[]){1.0 / 2, 0});
//
//	int size[2];
//	before.conv(lo0, lo0_res);
//	lo0_res.downsample_by2(0, size, lo0_res);
//	lo0_res.upsample_by2(0, size, lo0_res);
//	save_as_media("./test2-lo0.jpg", lo0_res, &mfmt);
//
//	before.conv(hi1, hi1_res);
//	hi1_res.downsample_by2(0, size, hi1_res);
//	hi1_res.upsample_by2(0, size, hi1_res);
//	save_as_media("./test2-hi1.jpg", hi1_res, &mfmt);
//
//	before.conv(hi2, hi2_res);
//	hi2_res.downsample_by2(0, size, hi2_res);
//	hi2_res.upsample_by2(0, size, hi2_res);
//	save_as_media("./test2-hi2.jpg", hi2_res, &mfmt);
//
//	before.conv(hi3, hi3_res);
//	hi3_res.downsample_by2(0, size, hi3_res);
//	hi3_res.upsample_by2(0, size, hi3_res);
//	save_as_media("./test2-hi3.jpg", hi3_res, &mfmt);
//
//	Tensor lo0r, hi1r, hi2r, hi3r, lo0r_res, hi1r_res, hi2r_res, hi3r_res;
//	lo0r = Tensor(2, (int[]){256, 256});
//	hi1r = Tensor(2, (int[]){256, 256});
//	hi2r = Tensor(2, (int[]){256, 256});
//	hi3r = Tensor(2, (int[]){256, 256});
//
//	lo0r.set_value((int[]){0, 0}, (double[]){1.0 / 2, 0});
//	lo0r.set_value((int[]){0, 255}, (double[]){1.0 / 2, 0});
//	lo0r.set_value((int[]){255, 0}, (double[]){1.0 / 2, 0});
//	lo0r.set_value((int[]){255, 255}, (double[]){1.0 / 2, 0});
//
//	hi1r.set_value((int[]){0, 0}, (double[]){1.0 / 2, 0});
//	hi1r.set_value((int[]){0, 255}, (double[]){-1.0 / 2, 0});
//	hi1r.set_value((int[]){255, 0}, (double[]){1.0 / 2, 0});
//	hi1r.set_value((int[]){255, 255}, (double[]){-1.0 / 2, 0});
//
//	hi2r.set_value((int[]){0, 0}, (double[]){1.0 / 2, 0});
//	hi2r.set_value((int[]){0, 255}, (double[]){1.0 / 2, 0});
//	hi2r.set_value((int[]){255, 0}, (double[]){-1.0 / 2, 0});
//	hi2r.set_value((int[]){255, 255}, (double[]){-1.0 / 2, 0});
//
//	hi3r.set_value((int[]){0, 0}, (double[]){1.0 / 2, 0});
//	hi3r.set_value((int[]){0, 255}, (double[]){-1.0 / 2, 0});
//	hi3r.set_value((int[]){255, 0}, (double[]){-1.0 / 2, 0});
//	hi3r.set_value((int[]){255, 255}, (double[]){1.0 / 2, 0});
//
//	save_as_media("./test2-lo0-up.jpg", lo0_res, &mfmt);
//	save_as_media("./test2-hi1-up.jpg", hi1_res, &mfmt);
//	save_as_media("./test2-hi2-up.jpg", hi2_res, &mfmt);
//	save_as_media("./test2-hi3-up.jpg", hi3_res, &mfmt);
//
//	lo0_res.conv(lo0r, lo0r_res);
//	hi1_res.conv(hi1r, hi1r_res);
//	hi2_res.conv(hi2r, hi2r_res);
//	hi3_res.conv(hi3r, hi3r_res);
//
//	lo0r_res.pw_add(hi1r_res, lo0r_res);
//	lo0r_res.pw_add(hi2r_res, lo0r_res);
//	lo0r_res.pw_add(hi3r_res, lo0r_res);
//
//	std::cout << "Rec: " << std::endl;
//	lo0r_res.print((int[]){128,128});
//	lo0r_res.print((int[]){129,128});
//	lo0r_res.print((int[]){130,128});
//	lo0r_res.print((int[]){131,128});
//	lo0r_res.print((int[]){128,128});
//	lo0r_res.print((int[]){128,129});
//	lo0r_res.print((int[]){128,130});
//	lo0r_res.print((int[]){128,131});
//	lo0r_res.print((int[]){200,128});
//	lo0r_res.print((int[]){200,129});
//	lo0r_res.print((int[]){200,130});
//	lo0r_res.print((int[]){200,131});
//
//	save_as_media("./test2-reconstructed.jpg", lo0r_res, &mfmt);
//
//	lo0r_res.psnr(before);
//
//	return 0;

//	// Case 3
//	Tensor before;
//	Media_Format mfmt;
//	load_as_tensor("./test2.jpg", before, &mfmt);
//	save_as_media("./test2-gray.jpg", before, &mfmt);
//
//	Tensor filter, result;
//	filter = Tensor(2, (int[]){256, 256});
//	filter.set_value((int[]){0, 0}, (double[]){1.0 / 9, 0});
//	filter.set_value((int[]){0, 1}, (double[]){1.0 / 9, 0});
//	filter.set_value((int[]){1, 0}, (double[]){1.0 / 9, 0});
//	filter.set_value((int[]){1, 1}, (double[]){1.0 / 9, 0});
//	filter.set_value((int[]){0, 2}, (double[]){1.0 / 9, 0});
//	filter.set_value((int[]){1, 2}, (double[]){1.0 / 9, 0});
//	filter.set_value((int[]){2, 0}, (double[]){1.0 / 9, 0});
//	filter.set_value((int[]){2, 1}, (double[]){1.0 / 9, 0});
//	filter.set_value((int[]){2, 2}, (double[]){1.0 / 9, 0});
//
//	before.conv(filter, result);
//	save_as_media("./test2-filtered.jpg", result, &mfmt);
//
//	Tensor decimated;
//	result.decimate(0, decimated);
//	save_as_media("./test2-decimated.jpg", decimated, &mfmt);
//
//	return 0;

//	// Case 1
//	Tensor before;
//
//	Media_Format mfmt;
//	load_as_tensor("./test.jpg", before, &mfmt);
//
//    Tensor shifted;
//    before.center_shift(shifted);
//    save_as_media("./test-shifted.jpg", shifted, &mfmt);
//
//    Tensor shifted_back;
//    shifted.icenter_shift(shifted_back);
//    save_as_media("./test-shifted-back.jpg", shifted_back, &mfmt);
//    return 0;

//  // Case 2
//	Tensor before;
//	Media_Format mfmt;
//	load_as_tensor("./test.avi", before, &mfmt);
//
//	Tensor after;
//	before.dft(after);
//
//	Tensor recovered;
//	after.idft(recovered);
//
//	save_as_media("./test-after.avi", after, &mfmt);
//	save_as_media("./test-recovered.avi", recovered, &mfmt);
//
//	return 0;

}
