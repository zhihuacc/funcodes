#include "include/graph_struct.h"
#include "include/grid_mrf.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	if (argc < 3)
	{
		cout << "Parameters too few." << endl;
		return 1;
	}

	Mat left_img = imread(string(argv[1]), CV_LOAD_IMAGE_GRAYSCALE);
	Mat right_img = imread(string(argv[2]), CV_LOAD_IMAGE_GRAYSCALE);

	if (left_img.rows != right_img.rows || left_img.cols != right_img.cols)
	{
		return 2;
	}

	GaussianBlur(left_img, left_img, Size(3,3), 0.7);
	GaussianBlur(left_img, left_img, Size(3,3), 0.7);

//	Mat_<Vec3b> &left_img_ = (Mat_<Vec3b> &)left_img;
//	Mat_<Vec3b> &right_img_ = (Mat_<Vec3b> &)left_img;

	int rows = left_img.rows;
	int cols = left_img.cols;
	float **observed_data = new float*[rows * cols];
	for (int i = 0; i < rows * cols; i++)
	{
		observed_data[i] = new float[LABEL_SET_SIZE];
	}

	for (int y = 0; y < rows; y++)
	{
		for (int x = 0; x < cols; x++)
		{
			int loc = y * cols + x;
			for (int k = 0; k < LABEL_SET_SIZE; k++)
			{
			    int x0 = x, y0 = y;
			    int x1 = x0 - k, y1 = y;
			    float diff = 0;
			    for (int i = -1; i <= 1; i++)
			    {
			    	for (int j = -1; j <= 1; j++)
			    	{
//			    		float diff0 = 0;
//			    		for (int c = 0; c < 3; c++)
//			    		{
//			    			diff0 += abs(left_img_(min(max(y0 + i, 0), rows - 1)
//			    				                               , min(max(x0 + j, 0), cols - 1))[c]
//			    				    - right_img_(min(max(y1 + i, 0), rows - 1)
//				                                                  , min(max(x1 + j, 0), cols - 1))[c]);
//			    		}
//			    		diff += (diff0 / 3);

			    		diff += abs(left_img.at<unsigned char>(min(max(y0 + i, 0), rows - 1)
			    				                               , min(max(x0 + j, 0), cols - 1))
			    				    - right_img.at<unsigned char>(min(max(y1 + i, 0), rows - 1)
				                                                  , min(max(x1 + j, 0), cols - 1)));
			    		diff /= 9;
			    	}
			    }
			    observed_data[loc][k] = diff;
			}
		}
	}


    grid_mrf mrf(rows, cols, observed_data);
    mrf.draw_graph();

	for (int i = 0; i < rows * cols; i++)
	{
		delete [] observed_data[i];
	}

	delete [] observed_data;

    mrf.inference(BP_MIN_SUM, 70, false);
	return 0;
}
