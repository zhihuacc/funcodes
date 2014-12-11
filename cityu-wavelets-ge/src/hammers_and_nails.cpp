#include "../include/hammers_and_nails.h"

string trim_head_and_tail(const string &str)
{
	if (str.empty())
	{
		return str;
	}

	string cpy = str;

	cpy.erase(0, str.find_first_not_of(" "));
	cpy.erase(str.find_last_not_of(" ") + 1);

	return cpy;
}

int string_options_parser(const string &opt_str, map<string, string> opt_map)
{
//	string::iterator it = opt_str.begin(), end = opt_str.end();
	int start = 0, found = 0, end = opt_str.length();
	while(found != end)
	{
		start = found;

		found =  opt_str.find(':', start);
		if (found == end)
		{
			break;
		}

		string param = opt_str.substr(start, found - start - 1);
		param = trim_head_and_tail(param);

		start = found;
		found = opt_str.find(';', start);
		if (found == end)
		{
			break;
		}
		string val = opt_str.substr(start, found - start - 1);
		val = trim_head_and_tail(val);

		if (val.empty())
		{
			continue;
		}

		opt_map[param] = val;
	}

	return 0;
}
