#include <config4cpp/Configuration.h>


#include "include/mat_toolbox.h"
#include "include/wavelets_toolbox.h"
#include "unit_test.h"
#include "include/denoising.h"
#include "include/commands.h"

using namespace config4cpp;

int cmd_config_parse(int argc, char **argv);

int main(int argc, char **argv)
{
//	setlocale(LC_ALL, "");
//	cmd_config_parse(argc, argv);
//
//	return 0;

	Unit_Test unit_test;
//	unit_test.psnr_test(argc, argv);

	unit_test.denoising(argc, argv);

//	unit_test.mat_select_test(argc, argv);

//	unit_test.decomposition_test(argc, argv);

//	unit_test.reconstruction_test(argc, argv);

//	unit_test.construct_1d_filter_test(argc, argv);

//	unit_test.fft_center_shift_test(argc, argv);

//	unit_test.test_any(argc, argv);
}

//int denoise_entry(const Configuration *cfg, const string noisy_file)
//{
//typedef double FLOAT_TYPE;
//
//	string param_scope("default");
//
//	int nlevels = cfg->lookupInt(param_scope.c_str(), "nlevels");
//	string fs_opt( cfg->lookupString(param_scope.c_str(), "fs") );
//	int ext_size = cfg->lookupInt(param_scope.c_str(), "ext_size");
//	string ext_method( cfg->lookupString(param_scope.c_str(), "ext_method") );
////	string mat_file ( cfg->lookupString(param_scope.c_str(), "f") );
//	int ndims = 0;
//	bool is_sym = cfg->lookupBoolean(param_scope.c_str(), "is_sym");
//
//	Mat_<Vec<FLOAT_TYPE, 2> > noisy_mat;
//	Media_Format mfmt;
//	load_as_tensor<FLOAT_TYPE>(noisy_file, noisy_mat, &mfmt);
//	ndims = noisy_mat.dims;
//
//	ML_MD_FS_Param ml_md_fs_param;
//	int ret = compose_fs_param(nlevels, ndims, fs_opt, ext_size, ext_method, is_sym, ml_md_fs_param);
//	if (ret)
//	{
//		cout << "Error in FS param. " << endl;
//		return 0;
//	}
//
//	cout << "Dec-Rec Paramters: " << endl;
//	cout << "  nlevels: " << nlevels << endl;
//	cout << "  ndims: " << ndims << endl;
//	cout << "  fs_opt: " << fs_opt << endl;
//	cout << "  ext_size: " << ext_size << endl;
//	cout << "  ext_method: " << ext_method << endl;
//	cout << "  is_sym: " << is_sym << endl;
//
//	double mean = static_cast<double>(cfg->lookupFloat(param_scope.c_str(), "mean"));
//	double stdev = static_cast<double>(cfg->lookupFloat(param_scope.c_str(), "stdev"));
//	double c = static_cast<double>(cfg->lookupFloat(param_scope.c_str(), "c"));
//	int wwidth = cfg->lookupInt(param_scope.c_str(), "wwidth");
//	bool doNorm = cfg->lookupBoolean(param_scope.c_str(), "doNorm");
//	string thr_method( cfg->lookupString(param_scope.c_str(), "thr_method") );
//
//	Thresholding_Param thr_param;
//	ret = compose_thr_param(mean, stdev, c, wwidth, doNorm, thr_method, thr_param);
//	if (ret)
//	{
//		cout << "Error in Thr param. " << endl;
//		return 0;
//	}
//
//	cout << endl << "Thresholding Parameters: " << endl;
//	cout << "  mean: " << mean << endl;
//	cout << "  stdev: " << stdev << endl;
//	cout << "  c: " << c << endl;
//	cout << "  wwidth: " << wwidth << endl;
//	cout << "  doNorm: " << doNorm << endl;
//	cout << "  thr_method: " << thr_method << endl;
//
//	Mat_<Vec<FLOAT_TYPE, 2> > denoised;
//	thresholding_denoise<FLOAT_TYPE>(noisy_mat, ml_md_fs_param, thr_param, denoised);
//
//	double score, msr;
//	psnr<FLOAT_TYPE>(noisy_mat, denoised, score, msr);
//	cout << "Denoised PSNR score: " << score << ", msr: " << msr << endl;
//
//	return 0;
//}

int prepopulate_config(Configuration *cfg, int argc, char **argv)
{
	string param_scope("default");
//	cfg->insertString(param_scope.c_str(), "cfg", "./example_cfg.txt");
	for (int i = 0; i < argc;)
	{
		string this_arg(argv[i]);
		if (this_arg == "-set")
		{
			if (i + 2 >= argc)
			{
				cout << "Cmd Arguments Wrong!" << endl;
				exit(0);
			}

			cfg->insertString(param_scope.c_str(), argv[i + 1], argv[i + 2]);
			i += 3;
		}
	}

	string cfg_file(cfg->lookupString(param_scope.c_str(), "cfg"));

	if (!cfg_file.empty())
	{
		cfg->parse(cfg_file.c_str());
	}

	return 0;
}

int cmd_config_parse(int argc, char **argv)
{
	Configuration *cfg = Configuration::create();
	try
	{
		if (argc < 3)
		{
			cout << "Too Few arguments!" << endl;
			exit(0);
		}

		string sub_cmd(argv[1]);

		if (sub_cmd == "denoise")
		{

			string noisy_file(argv[2]);
			prepopulate_config(cfg, argc - 3, argv + 3);
			denoise_entry(cfg, noisy_file);
		}
		else if (sub_cmd == "psnr")
		{
			string left_file(argv[2]);
			string right_file(argv[3]);
//			prepopulate_config(cfg, argc - 4, argv + 4);
			psnr_entry(left_file, right_file);
		}
		else if (sub_cmd == "cvtxml")
		{
			string fn(argv[2]);
//			prepopulate_config(cfg, argc - 3, argv + 3);
			cvtxml_entry(fn);
		}
		else
		{
			cout << "Sub-command unrecognized!" << endl;
			exit(0);
		}

	}
	catch (const ConfigurationException &ex)
	{
		cout << ex.c_str() << endl;
	}
	cfg->destroy();
	return 0;
}
