#include<iostream>
#include<opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include<opencv2/dnn.hpp>
#include<opencv2/cudaarithm.hpp>
#include<opencv2/cudaoptflow.hpp>
#include<opencv2/cudaimgproc.hpp>
#include<opencv2/cudafeatures2d.hpp>
#include<opencv2/cudaobjdetect.hpp>
#include<opencv2/cudawarping.hpp>
#include<opencv2/cudafilters.hpp>
using namespace cv;
using namespace std;
using namespace cv::cuda;
using namespace cv::dnn;

int test()
{
	Mat src_host = imread("F:/pic.jpg");
	GpuMat src, gray;
	src.upload(src_host);
	cuda::cvtColor(src, gray, COLOR_BGR2GRAY);
	Mat gray_host;
	gray.download(gray_host);
	imshow("src", src_host);
	imshow("gray", gray_host);
	waitKey(0);
	return 0;
}