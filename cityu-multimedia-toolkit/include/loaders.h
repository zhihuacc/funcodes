#ifndef _LOADERS_H
#define _LOADERS_H

#include "tensor.h"

struct Media_Format
{
	union {
		struct
		{
			int height;
			int width;
		} imgage_prop;
		struct
		{
			int fps;
			int fourcc;
			int frame_count;
			int frame_height;
			int frame_width;
		} video_prop;
	};
};

// Reader for images, videos etc
int load_as_tensor(const string &filename, Tensor &tensor, Media_Format *media_file_fmt);
int save_as_media(const string &filename, const Tensor &tensor, const Media_Format *media_file_fmt);

int print_tensor(const Tensor &t, const string &filename);

#endif
