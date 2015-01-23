#ifndef _COMMANDS_H
#define _COMMANDS_H

#include <string>
#include <config4cpp/Configuration.h>

using namespace std;
using namespace config4cpp;

int cvtxml_entry(const string &fn);
int psnr_entry(const string &left, const string &right);
int inpaint(const Configuration *cfg, const string &img_name, const string &mask_name);


#endif
