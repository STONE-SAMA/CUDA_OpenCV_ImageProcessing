#include<iostream>
#include<algorithm>
#include<stdio.h>
#include<stdlib.h>
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
#include<vector>
#include<string>
using namespace cv;
using namespace std;
using namespace cv::cuda;
using namespace cv::dnn;

void drawHistogram(Mat& image, string str);//绘制直方图

//灰度直方图均衡
void grayProcess()
{
	Mat h_img = imread("F:/cuda_pictures/girl.jpg",0);
	GpuMat d_img, d_result;
	d_img.upload(h_img);
	cv::cuda::equalizeHist(d_img, d_result);
	Mat h_result;
	d_result.download(h_result);
	imshow("原始图像", h_img);
	//原始图像直方图
	drawHistogram(h_img, "原始图像直方图");
	imshow("直方图均衡", h_result);
	//图像均衡后直方图
	drawHistogram(h_result, "图像均衡后直方图");
	waitKey(0);
}

//彩色直方图均衡
int myColorProcess()
{
	cv::Mat src_host = imread("F:/cuda_pictures/sea.jpg");
	if (!src_host.data)
	{
		cout << "读取图片错误，请重新输入正确路径！\n";
		system("pause");
		return -1;
	}
	GpuMat src, h_result,g_result;
	src.upload(src_host);
	cuda::cvtColor(src, h_result, cv::COLOR_BGR2HSV);
	std::vector<GpuMat> vec_channels;
	cuda::split(h_result, vec_channels);
	cuda::equalizeHist(vec_channels[2], vec_channels[2]);
	cuda::merge(vec_channels, h_result);
	cuda::cvtColor(h_result, g_result, cv::COLOR_HSV2BGR);
	Mat result;
	g_result.download(result);
	imshow("原始图像",src_host);
	//原始图像直方图
	drawHistogram(src_host,"原始图像直方图");
	imshow("直方图均衡", result);
	//图像均衡后直方图
	drawHistogram(result,"图像均衡后直方图");
	waitKey(0);
	return 0;
}

void colorProcess()
{
	cv::Mat h_img1 = imread("F:/cuda_pictures/sea.jpg");
	cv::Mat h_img2, h_result;
	cv::cvtColor(h_img1, h_img2, cv::COLOR_BGR2HSV);//BGR转为HSV
	//拆分成三通道,分别计算
	std::vector<cv::Mat> vec_channels;
	cv::split(h_img2, vec_channels);
	//色调与饱和度通道包含颜色信息，无需均衡
	//只需对值通道进行均衡
	cv::equalizeHist(vec_channels[2], vec_channels[2]);
	cv::merge(vec_channels, h_img2);
	cv::cvtColor(h_img2, h_result, cv::COLOR_HSV2BGR);
	cv::imshow("原始图像",h_img1);
	cv::imshow("直方图均衡", h_result);
	waitKey(0);
}

//图片大小调整
void changeSize()
{
	Mat host_img = imread("F:/cuda_pictures/girl.jpg", 0);
	GpuMat d_img, d_result1, d_result2;
	d_img.upload(host_img);
	int width = d_img.cols;
	int height = d_img.size().height;
	cuda::resize(d_img, d_result1, cv::Size(300,300), cv::INTER_CUBIC);//双三次
	cuda::resize(d_img, d_result2, cv::Size(0.5 * width, 0.5 * height), cv::INTER_LINEAR);//双线性
	Mat h_result1, h_result2;
	d_result1.download(h_result1);
	d_result2.download(h_result2);
	cv::imshow("原始图像", host_img);
	cv::imshow("固定大小", h_result1);
	cv::imshow("尺寸减半", h_result2);
	waitKey(0);
}

void myChangeSize()
{
	Mat host_img = imread("F:/cuda_pictures/girl.jpg", 0);
	int width, height;
	cout << "请输入需要调整的宽度：" << endl;
	cin >> width;
	cout << "请输入需要调整的高度：" << endl;
	cin >> height;
	GpuMat device_img, device_result;
	device_img.upload(host_img);
	cout << "原始宽度:" << device_img.cols;
	cout << "原始高度:" << device_img.size().height << endl;
	cuda::resize(device_img, device_result, cv::Size(width, height), cv::INTER_CUBIC);
	Mat host_result;
	device_result.download(host_result);
	cv::imshow("原始图像", host_img);
	cv::imshow("调整大小", host_result);
	waitKey(0);
}

//绘制直方图
void drawHistogram(Mat& srcImage, string str)
{
	int dims = srcImage.channels();//图片通道数
	if (dims == 3)//彩色
	{
		int bins = 256;
		int histsize[] = { bins };
		float range[] = { 0, 256 };
		const float* histRange = { range };
		Mat  b_Hist, g_Hist, r_Hist;
		//图像通道的分离，3个通道B、G、R
		vector<Mat> rgb_channel;
		split(srcImage, rgb_channel);
		//计算各个通道的直方图
		//B-通道
		calcHist(&rgb_channel[0], 1, 0, Mat(), b_Hist, 1, histsize, &histRange, true, false); 
		//G-通道
		calcHist(&rgb_channel[1], 1, 0, Mat(), g_Hist, 1, histsize, &histRange, true, false); 
		//R-通道
		calcHist(&rgb_channel[2], 1, 0, Mat(), r_Hist, 1, histsize, &histRange, true, false); 
		//设置直方图绘图参数
		int hist_h = 360;
		int hist_w = bins * 3;
		int bin_w = cvRound((double)hist_w / bins);
		//创建黑底图像
		Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
		//直方图归一化到[0,histImage.rows]
		//B-通道
		cv::normalize(b_Hist, b_Hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		//G-通道
		cv::normalize(g_Hist, g_Hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		//R-通道
		cv::normalize(r_Hist, r_Hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		//绘制图像
		for (int i = 1; i < bins; i++)
		{
			//绘制B通道的直方图信息
			line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_Hist.at<float>(i - 1))),
				Point(bin_w * (i), hist_h - cvRound(b_Hist.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
			//绘制G通道的直方图信息
			line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_Hist.at<float>(i - 1))),
				Point(bin_w * (i), hist_h - cvRound(g_Hist.at<float>(i))), Scalar(0, 255, 0), 2, 8, 0);
			 //绘制R通道的直方图信息
			line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_Hist.at<float>(i - 1))), 
				Point(bin_w * (i), hist_h - cvRound(r_Hist.at<float>(i))), Scalar(0, 0, 255), 2, 8, 0);
		}
		imshow(str, histImage);
	}
	else//灰度图
	{
		const int channels[1] = { 0 };
		const int bins[1] = { 256 };
		float hranges[2] = { 0,255 };
		const float* ranges[1] = { hranges };
		Mat hist;
		// 计算Blue, Green, Red通道的直方图
		calcHist(&srcImage, 1, 0, Mat(), hist, 1, bins, ranges);
		//设置直方图绘图参数
		int hist_h = 360;
		int hist_w = bins[0] * 3;
		int bin_w = cvRound((double)hist_w / bins[0]);
		Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);
		// 归一化直方图数据
		cv::normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		// 绘制直方图曲线
		for (int i = 1; i < bins[0]; i++) {
			line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
				Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))), Scalar(0, 255, 0), 2, 8, 0);
		}
		imshow(str, histImage);
	}
}

//平均滤波器
void myAverageFilter()
{
	Mat host_img = imread("F:/cuda_pictures/girl.jpg", 0);
	GpuMat d_img, d_result3,d_result5;
	d_img.upload(host_img);
	cv::Ptr<cuda::Filter> filter3, filter5;
	//平均滤波3X3
	filter3 = cuda::createBoxFilter(CV_8UC1, CV_8UC1, cv::Size(3, 3));
	filter3->apply(d_img, d_result3);
	//平均滤波5X5
	filter5 = cuda::createBoxFilter(CV_8UC1, CV_8UC1, cv::Size(5, 5));
	filter5->apply(d_img, d_result5);

	Mat h_result3, h_result5;
	d_result3.download(h_result3);
	d_result5.download(h_result5);
	imshow("原始图像", host_img);
	imshow("3X3平均滤波", h_result3);
	imshow("5X5平均滤波", h_result5);

	waitKey(0);
}

//高斯滤波器
void myGaussFilter()
{
	Mat host_img = imread("F:/cuda_pictures/girl.jpg", 0);
	GpuMat d_img, d_result3, d_result5;
	d_img.upload(host_img);
	cv::Ptr<cuda::Filter> filter3, filter5;
	filter3 = cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(3, 3), 1);
	filter3->apply(d_img, d_result3);
	filter5 = cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(5, 5), 1);
	filter5->apply(d_img, d_result5);
	
	Mat h_result3, h_result5;
	d_result3.download(h_result3);
	d_result5.download(h_result5);
	imshow("原始图像", host_img);
	imshow("3X3高斯滤波", h_result3);
	imshow("5X5高斯滤波", h_result5);

	waitKey(0);
}

//中值过滤
int myMedianFilter()
{
	Mat host_img = imread("F:/cuda_pictures/salt_pepper.jpg",0);
	if (!host_img.data)
	{
		cout << "读取图片错误，请重新输入正确路径！\n";
		system("pause");
		return -1;
	}
	Mat host_result;
	cv::medianBlur(host_img, host_result, 3);
	imshow("原始图像", host_img);
	imshow("中值滤波", host_result);

	waitKey(0);
}
//腐蚀
int myPicErode()
{
	Mat host_img = imread("F:/cuda_pictures/blobs.png", 0);
	if (!host_img.data)
	{
		cout << "读取图片错误，请重新输入正确路径！\n";
		system("pause");
		return -1;
	}
	GpuMat d_img, d_result;
	//定义形态操作的结构元素
	Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	d_img.upload(host_img);
	cv::Ptr<cuda::Filter> filter;
	filter = cuda::createMorphologyFilter(cv::MORPH_ERODE, CV_8UC1, element);
	filter->apply(d_img, d_result);
	Mat h_result;
	d_result.download(h_result);
	
	imshow("原始图像", host_img);
	imshow("腐蚀图像", h_result);

	waitKey(0);
}
//膨胀
int myPicDilate()
{
	Mat host_img = imread("F:/cuda_pictures/blobs.png", 0);
	if (!host_img.data)
	{
		cout << "读取图片错误，请重新输入正确路径！\n";
		system("pause");
		return -1;
	}
	GpuMat d_img, d_result;
	//定义形态操作的结构元素
	Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	d_img.upload(host_img);
	cv::Ptr<cuda::Filter> filter;
	filter = cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1, element);
	filter->apply(d_img, d_result);
	Mat h_result;
	d_result.download(h_result);

	imshow("原始图像", host_img);
	imshow("膨胀图像", h_result);

	waitKey(0);
}

void showMenu()//显示菜单
{
	cout << "========== 选 择 ==========" << endl;
	cout << "===1、灰度图像直方图均衡===" << endl;
	cout << "===2、彩色图像直方图均衡===" << endl;
	cout << "===3、彩色图像CUDA      ===" << endl;
	cout << "===4、调整大小          ===" << endl;
	cout << "===5、输入指定调整大小  ===" << endl;
	cout << "===6、使用平均滤波器    ===" << endl;
	cout << "===7、使用高斯滤波器    ===" << endl;
	cout << "===8、使用中值滤波器    ===" << endl;
	cout << "===9、对图像腐蚀操作    ===" << endl;
	cout << "===10、对图像膨胀操作   ===" << endl;
	cout << "===========================" << endl;
}


int main()
{
	
	int choose;
	while (true)
	{
		showMenu();
		cin >> choose;
	
		switch (choose)
		{
		case 1:
			grayProcess();
			break;
		case 2:
			colorProcess();
			break;
		case 3:
			myColorProcess();
			break;
		case 4:
			changeSize();
			break;
		case 5:
			myChangeSize();
			break;
		case 6:
			myAverageFilter();
			break;
		case 7:
			myGaussFilter();
			break;
		case 8:
			myMedianFilter();
			break;
		case 9:
			myPicErode();
			break;
		case 10:
			myPicDilate();
			break;
		default:
			cout << "不存在此操作" << endl;
			system("pause");
		}
		system("cls");
	}

	system("pause");
	return 0;
}