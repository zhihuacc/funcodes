#include <config4cpp/Configuration.h>


#include "include/mat_toolbox.h"
#include "include/wavelets_toolbox.h"
#include "unit_test.h"
#include "include/denoising.h"
#include "include/commands.h"
#include "include/inpaint.h"

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

//	unit_test.denoising(argc, argv);

//	unit_test.mat_select_test(argc, argv);

//	unit_test.decomposition_test(argc, argv);

//	unit_test.reconstruction_test(argc, argv);

//	unit_test.construct_1d_filter_test(argc, argv);

//	unit_test.fft_center_shift_test(argc, argv);

//	unit_test.conv_and_ds_test(argc, argv);

//	unit_test.conv_and_us_test(argc, argv);

	unit_test.comp_supp_test(argc, argv);

//	unit_test.performance_test(argc, argv);

//	unit_test.test_any(argc, argv);
}


int prepopulate_config(Configuration *cfg, const string &top_scope, int argc, char **argv)
{
	string cfg_file;
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

			cfg->insertString(top_scope.c_str(), argv[i + 1], argv[i + 2]);
			i += 3;
		}
		else if (this_arg == "-cfg")
		{
			cfg_file = argv[i + 1];
			i += 2;
		}
	}

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
			prepopulate_config(cfg, "", argc - 2, argv + 2);
			batch_denoise<double>(cfg, "");
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
//		else if (sub_cmd == "denodemo")
//		{
//			string fn(argv[2]);
//			denoising_demo(fn);
//		}
		else if (sub_cmd == "inpaint")
		{
			prepopulate_config(cfg, "", argc - 2, argv + 2);
			batch_inpaint<double>(cfg, "");

//			string img_name(argv[2]);
//			string mask_name(argv[3]);
//			inpaint(cfg, img_name, mask_name);
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
