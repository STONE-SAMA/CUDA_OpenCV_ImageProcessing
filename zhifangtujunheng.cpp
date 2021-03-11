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

void drawHistogram(Mat& image, string str);//����ֱ��ͼ

//�Ҷ�ֱ��ͼ����
void grayProcess()
{
	Mat h_img = imread("F:/cuda_pictures/girl.jpg",0);
	GpuMat d_img, d_result;
	d_img.upload(h_img);
	cv::cuda::equalizeHist(d_img, d_result);
	Mat h_result;
	d_result.download(h_result);
	imshow("ԭʼͼ��", h_img);
	//ԭʼͼ��ֱ��ͼ
	drawHistogram(h_img, "ԭʼͼ��ֱ��ͼ");
	imshow("ֱ��ͼ����", h_result);
	//ͼ������ֱ��ͼ
	drawHistogram(h_result, "ͼ������ֱ��ͼ");
	waitKey(0);
}

//��ɫֱ��ͼ����
int myColorProcess()
{
	cv::Mat src_host = imread("F:/cuda_pictures/sea.jpg");
	if (!src_host.data)
	{
		cout << "��ȡͼƬ����������������ȷ·����\n";
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
	imshow("ԭʼͼ��",src_host);
	//ԭʼͼ��ֱ��ͼ
	drawHistogram(src_host,"ԭʼͼ��ֱ��ͼ");
	imshow("ֱ��ͼ����", result);
	//ͼ������ֱ��ͼ
	drawHistogram(result,"ͼ������ֱ��ͼ");
	waitKey(0);
	return 0;
}

void colorProcess()
{
	cv::Mat h_img1 = imread("F:/cuda_pictures/sea.jpg");
	cv::Mat h_img2, h_result;
	cv::cvtColor(h_img1, h_img2, cv::COLOR_BGR2HSV);//BGRתΪHSV
	//��ֳ���ͨ��,�ֱ����
	std::vector<cv::Mat> vec_channels;
	cv::split(h_img2, vec_channels);
	//ɫ���뱥�Ͷ�ͨ��������ɫ��Ϣ���������
	//ֻ���ֵͨ�����о���
	cv::equalizeHist(vec_channels[2], vec_channels[2]);
	cv::merge(vec_channels, h_img2);
	cv::cvtColor(h_img2, h_result, cv::COLOR_HSV2BGR);
	cv::imshow("ԭʼͼ��",h_img1);
	cv::imshow("ֱ��ͼ����", h_result);
	waitKey(0);
}

//ͼƬ��С����
void changeSize()
{
	Mat host_img = imread("F:/cuda_pictures/girl.jpg", 0);
	GpuMat d_img, d_result1, d_result2;
	d_img.upload(host_img);
	int width = d_img.cols;
	int height = d_img.size().height;
	cuda::resize(d_img, d_result1, cv::Size(300,300), cv::INTER_CUBIC);//˫����
	cuda::resize(d_img, d_result2, cv::Size(0.5 * width, 0.5 * height), cv::INTER_LINEAR);//˫����
	Mat h_result1, h_result2;
	d_result1.download(h_result1);
	d_result2.download(h_result2);
	cv::imshow("ԭʼͼ��", host_img);
	cv::imshow("�̶���С", h_result1);
	cv::imshow("�ߴ����", h_result2);
	waitKey(0);
}

void myChangeSize()
{
	Mat host_img = imread("F:/cuda_pictures/girl.jpg", 0);
	int width, height;
	cout << "��������Ҫ�����Ŀ�ȣ�" << endl;
	cin >> width;
	cout << "��������Ҫ�����ĸ߶ȣ�" << endl;
	cin >> height;
	GpuMat device_img, device_result;
	device_img.upload(host_img);
	cout << "ԭʼ���:" << device_img.cols;
	cout << "ԭʼ�߶�:" << device_img.size().height << endl;
	cuda::resize(device_img, device_result, cv::Size(width, height), cv::INTER_CUBIC);
	Mat host_result;
	device_result.download(host_result);
	cv::imshow("ԭʼͼ��", host_img);
	cv::imshow("������С", host_result);
	waitKey(0);
}

//����ֱ��ͼ
void drawHistogram(Mat& srcImage, string str)
{
	int dims = srcImage.channels();//ͼƬͨ����
	if (dims == 3)//��ɫ
	{
		int bins = 256;
		int histsize[] = { bins };
		float range[] = { 0, 256 };
		const float* histRange = { range };
		Mat  b_Hist, g_Hist, r_Hist;
		//ͼ��ͨ���ķ��룬3��ͨ��B��G��R
		vector<Mat> rgb_channel;
		split(srcImage, rgb_channel);
		//�������ͨ����ֱ��ͼ
		//B-ͨ��
		calcHist(&rgb_channel[0], 1, 0, Mat(), b_Hist, 1, histsize, &histRange, true, false); 
		//G-ͨ��
		calcHist(&rgb_channel[1], 1, 0, Mat(), g_Hist, 1, histsize, &histRange, true, false); 
		//R-ͨ��
		calcHist(&rgb_channel[2], 1, 0, Mat(), r_Hist, 1, histsize, &histRange, true, false); 
		//����ֱ��ͼ��ͼ����
		int hist_h = 360;
		int hist_w = bins * 3;
		int bin_w = cvRound((double)hist_w / bins);
		//�����ڵ�ͼ��
		Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
		//ֱ��ͼ��һ����[0,histImage.rows]
		//B-ͨ��
		cv::normalize(b_Hist, b_Hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		//G-ͨ��
		cv::normalize(g_Hist, g_Hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		//R-ͨ��
		cv::normalize(r_Hist, r_Hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		//����ͼ��
		for (int i = 1; i < bins; i++)
		{
			//����Bͨ����ֱ��ͼ��Ϣ
			line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_Hist.at<float>(i - 1))),
				Point(bin_w * (i), hist_h - cvRound(b_Hist.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
			//����Gͨ����ֱ��ͼ��Ϣ
			line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_Hist.at<float>(i - 1))),
				Point(bin_w * (i), hist_h - cvRound(g_Hist.at<float>(i))), Scalar(0, 255, 0), 2, 8, 0);
			 //����Rͨ����ֱ��ͼ��Ϣ
			line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_Hist.at<float>(i - 1))), 
				Point(bin_w * (i), hist_h - cvRound(r_Hist.at<float>(i))), Scalar(0, 0, 255), 2, 8, 0);
		}
		imshow(str, histImage);
	}
	else//�Ҷ�ͼ
	{
		const int channels[1] = { 0 };
		const int bins[1] = { 256 };
		float hranges[2] = { 0,255 };
		const float* ranges[1] = { hranges };
		Mat hist;
		// ����Blue, Green, Redͨ����ֱ��ͼ
		calcHist(&srcImage, 1, 0, Mat(), hist, 1, bins, ranges);
		//����ֱ��ͼ��ͼ����
		int hist_h = 360;
		int hist_w = bins[0] * 3;
		int bin_w = cvRound((double)hist_w / bins[0]);
		Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);
		// ��һ��ֱ��ͼ����
		cv::normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		// ����ֱ��ͼ����
		for (int i = 1; i < bins[0]; i++) {
			line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
				Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))), Scalar(0, 255, 0), 2, 8, 0);
		}
		imshow(str, histImage);
	}
}

//ƽ���˲���
void myAverageFilter()
{
	Mat host_img = imread("F:/cuda_pictures/girl.jpg", 0);
	GpuMat d_img, d_result3,d_result5;
	d_img.upload(host_img);
	cv::Ptr<cuda::Filter> filter3, filter5;
	//ƽ���˲�3X3
	filter3 = cuda::createBoxFilter(CV_8UC1, CV_8UC1, cv::Size(3, 3));
	filter3->apply(d_img, d_result3);
	//ƽ���˲�5X5
	filter5 = cuda::createBoxFilter(CV_8UC1, CV_8UC1, cv::Size(5, 5));
	filter5->apply(d_img, d_result5);

	Mat h_result3, h_result5;
	d_result3.download(h_result3);
	d_result5.download(h_result5);
	imshow("ԭʼͼ��", host_img);
	imshow("3X3ƽ���˲�", h_result3);
	imshow("5X5ƽ���˲�", h_result5);

	waitKey(0);
}

//��˹�˲���
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
	imshow("ԭʼͼ��", host_img);
	imshow("3X3��˹�˲�", h_result3);
	imshow("5X5��˹�˲�", h_result5);

	waitKey(0);
}

//��ֵ����
int myMedianFilter()
{
	Mat host_img = imread("F:/cuda_pictures/salt_pepper.jpg",0);
	if (!host_img.data)
	{
		cout << "��ȡͼƬ����������������ȷ·����\n";
		system("pause");
		return -1;
	}
	Mat host_result;
	cv::medianBlur(host_img, host_result, 3);
	imshow("ԭʼͼ��", host_img);
	imshow("��ֵ�˲�", host_result);

	waitKey(0);
}
//��ʴ
int myPicErode()
{
	Mat host_img = imread("F:/cuda_pictures/blobs.png", 0);
	if (!host_img.data)
	{
		cout << "��ȡͼƬ����������������ȷ·����\n";
		system("pause");
		return -1;
	}
	GpuMat d_img, d_result;
	//������̬�����ĽṹԪ��
	Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	d_img.upload(host_img);
	cv::Ptr<cuda::Filter> filter;
	filter = cuda::createMorphologyFilter(cv::MORPH_ERODE, CV_8UC1, element);
	filter->apply(d_img, d_result);
	Mat h_result;
	d_result.download(h_result);
	
	imshow("ԭʼͼ��", host_img);
	imshow("��ʴͼ��", h_result);

	waitKey(0);
}
//����
int myPicDilate()
{
	Mat host_img = imread("F:/cuda_pictures/blobs.png", 0);
	if (!host_img.data)
	{
		cout << "��ȡͼƬ����������������ȷ·����\n";
		system("pause");
		return -1;
	}
	GpuMat d_img, d_result;
	//������̬�����ĽṹԪ��
	Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	d_img.upload(host_img);
	cv::Ptr<cuda::Filter> filter;
	filter = cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1, element);
	filter->apply(d_img, d_result);
	Mat h_result;
	d_result.download(h_result);

	imshow("ԭʼͼ��", host_img);
	imshow("����ͼ��", h_result);

	waitKey(0);
}

void showMenu()//��ʾ�˵�
{
	cout << "========== ѡ �� ==========" << endl;
	cout << "===1���Ҷ�ͼ��ֱ��ͼ����===" << endl;
	cout << "===2����ɫͼ��ֱ��ͼ����===" << endl;
	cout << "===3����ɫͼ��CUDA      ===" << endl;
	cout << "===4��������С          ===" << endl;
	cout << "===5������ָ��������С  ===" << endl;
	cout << "===6��ʹ��ƽ���˲���    ===" << endl;
	cout << "===7��ʹ�ø�˹�˲���    ===" << endl;
	cout << "===8��ʹ����ֵ�˲���    ===" << endl;
	cout << "===9����ͼ��ʴ����    ===" << endl;
	cout << "===10����ͼ�����Ͳ���   ===" << endl;
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
			cout << "�����ڴ˲���" << endl;
			system("pause");
		}
		system("cls");
	}

	system("pause");
	return 0;
}