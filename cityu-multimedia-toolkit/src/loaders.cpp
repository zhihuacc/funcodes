#include "../include/loaders.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;


// Reader for images, videos etc
int load_as_tensor(const string &filename, Tensor &tensor, Media_Format *media_file_fmt)
{
	size_t pos = filename.find_last_of('.');
	if (pos == std::string::npos)
	{
		return -1;
	}

	string suffix = filename.substr(pos);
	Mat mat;
	// images
	if (suffix == string(".jpg") || suffix == string(".bmp") || suffix == string(".jpeg"))
	{
		Mat img = imread(filename);
		if (img.empty())
		{
			return -2;
		}

		switch (img.type())
		{
		case CV_8UC3:
			cvtColor(img, img, CV_BGR2GRAY);
			break;
		case CV_8UC1:
			break;
		default:
			return -3;
		}

		mat.create(img.size(), CV_64FC2);
		for (int r = 0; r < img.rows; ++r)
		{
			for (int c = 0; c < img.cols; ++c)
			{
				mat.at<Vec2d>(r, c)[0] = img.at<uchar>(r, c);
				mat.at<Vec2d>(r, c)[1] = 0;
			}
		}

		tensor = Tensor(2, mat);

		return 0;
	}

	// videos
	if (suffix == string(".avi") || suffix == string(".wav"))
	{
		VideoCapture cap(filename);
		if (!cap.isOpened())
		{
			return -2;
		}


		int sz[3];
		sz[0] = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_COUNT));
		sz[1] = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_HEIGHT));
		sz[2] = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_WIDTH));


		if (media_file_fmt != NULL)
		{
			media_file_fmt->video_prop.frame_count = sz[0];
			media_file_fmt->video_prop.frame_height = sz[1];
			media_file_fmt->video_prop.frame_width = sz[2];
			media_file_fmt->video_prop.fps = static_cast<int>(cap.get(CV_CAP_PROP_FPS));
			media_file_fmt->video_prop.fourcc = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));
		}


		mat.create(3, sz, CV_64FC2);
		for (int f = 0; f < sz[0]; f++)
		{
			Mat frame;
			bool bret = cap.read(frame);
			if (!bret)
			{
				break;
			}

			// Handle only 8UC1.
			switch(frame.type())
			{
			case CV_8UC3:
				cvtColor(frame, frame, CV_BGR2GRAY);
				break;
			case CV_8UC1:
				break;
			default:

				return -3;
			}

//			imshow("Current Frame", frame);
//			waitKey();


			Range ranges[3];
			ranges[0] = Range(f, f + 1);
			ranges[1] = Range::all();
			ranges[2] = Range::all();

			Mat plane = mat.row(f);
			for (int r = 0; r < sz[1]; ++r)
			{
				for (int c = 0; c < sz[2]; ++c)
				{
					plane.at<Vec2d>(0, r, c)[0] = frame.at<uchar>(r, c);
					plane.at<Vec2d>(0, r, c)[1] = 0;
				}
			}
		}

		tensor = Tensor(3, mat);

		return 0;
	}

	return -4;
}

int save_as_media(const string &filename, const Tensor &tensor, const Media_Format *media_file_fmt)
{
	Mat mat = tensor._data_mat;
    if (tensor._order == 2)
    {
         Mat img(mat.size(), CV_64FC1);

         for (int i = 0; i < mat.rows; i++)
         {
         	for (int j = 0; j < mat.cols; j++)
         	{
         		double d = mat.at<Vec2d>(i, j)[0] * mat.at<Vec2d>(i, j)[0]
         		         + mat.at<Vec2d>(i, j)[1] * mat.at<Vec2d>(i, j)[1];

         		img.at<double>(i, j) = sqrt(d);
         	}
         }

         img.convertTo(img, CV_8UC1, 1, 0);

         imwrite(filename, img);

         return 0;
    }

    if (tensor._order == 3)
    {
    	if (media_file_fmt == NULL)
    	{
    		return -3;
    	}

    	Mat mat = tensor._data_mat;
    	VideoWriter writer(filename, CV_FOURCC('P','I','M','1')
    			         , media_file_fmt->video_prop.fps
    			         , Size(mat.size[2], mat.size[1]), false);

    	if (!writer.isOpened())
    	{
    		return -2;
    	}

    	for (int f = 0; f < mat.size[0]; f++)
    	{
    		Mat plane = mat.row(f);
            Mat frame(Size(plane.size[2], plane.size[1]), CV_64FC1);

            for (int i = 0; i < mat.size[1]; i++)
            {
            	for (int j = 0; j < mat.size[2]; j++)
            	{
            		double d = plane.at<Vec2d>(0, i, j)[0] * plane.at<Vec2d>(0, i, j)[0]
            		         + plane.at<Vec2d>(0, i, j)[1] * plane.at<Vec2d>(0, i, j)[1];

            		frame.at<double>(i, j) = sqrt(d);
            	}
            }

            frame.convertTo(frame, CV_8UC1, 1, 0);
            writer << frame;
    	}
    }
	return 0;
}

int print_tensor(const Tensor &t, const string &filename)
{

	return 0;
}
