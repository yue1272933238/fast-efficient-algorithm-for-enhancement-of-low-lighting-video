#include <iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;

//输出一个数组的所有元素
void print_arr(int arr[], int len)
{
	for (int i = 0; i < len; i++)
		cout << arr[i] << " ";
	cout << endl;
}

//对图像进行反色操作
void get_inverted_img(Mat img, Mat inverted)
{
	for (int row = 0; row < inverted.rows; row++)
		for (int col = 0; col < inverted.cols; col++)
			for (int channel = 0; channel < inverted.channels(); channel++)
				inverted.at<Vec3b>(row, col)[channel] = 255 - img.at<Vec3b>(row, col)[channel];
}

//根据反色图获得参数A的值（R,G,B对应的有三个值）
void get_A_value(Mat M, int A[])
{
	//MM这个Mat存放的是M中，每个像素值R,G,B三个通道中的最小的一个，所以是一维的。
	Mat MM = Mat::zeros(M.rows, M.cols, CV_8UC1);
	for (int row = 0; row < M.rows; row++)
	{
		for (int col = 0; col < M.cols; col++)
		{
			int min_intensity = 255;
			for (int channel = 0; channel < M.channels(); channel++)
				if (min_intensity > M.at<Vec3b>(row, col)[channel])
					min_intensity = M.at<Vec3b>(row, col)[channel];
			MM.at<uchar>(row, col) = min_intensity;
		}
	}

	//top_k存放的是MM这个Mat中最大的前100个数所对应的位置（行列值转编码一个整数）。
	int top_k[100] = { 0 };
	int len = 100;
	int max_value = 0;
	int index = -1;

	for (int i = 0; i < len; i++)
	{
		max_value = 0;
		index = -1;
		for (int row = 0; row < MM.rows; row++)
		{
			for (int col = 0; col < MM.cols; col++)
			{
				if (MM.at<uchar>(row, col) > max_value)
				{
					max_value = MM.at<uchar>(row, col);
					index = row * MM.cols + col;
				}
			}
		}
		top_k[i] = index;
		MM.at<uchar>(index / MM.cols, index % MM.cols) = 0;
	}

	//找出这一百个像素中RGB总和最大的那个像素
	max_value = 0;
	index = -1;
	int sum = 0;
	for (int i = 0; i < len; i++)
	{
		sum = 0;
		for (int channel = 0; channel < M.channels(); channel++)
			sum = sum + M.at<Vec3b>(top_k[i] / M.cols, top_k[i] % M.cols)[channel];
		if (sum >= max_value)
		{
			max_value = sum;
			index = i;
		}
	}

	//修改全球大气值A
	for (int channel = 0; channel < M.channels(); channel++)
		A[channel] = M.at<Vec3b>(top_k[index] / M.cols, top_k[index] % M.cols)[channel];
	return;
}

//以M[row][col][channel]为中心，尺寸为kernel_size的最小值滤波结果
int minimum_filter(Mat M, int row, int col, int kernel_size, int channel)
{
	int temp = 255;
	int offset = (kernel_size - 1) / 2;

	for (int i = (row - offset); i <= (row + offset); i++)
		for (int j = (col - offset); j <= (col + offset); j++)
			if (M.at<Vec3b>(i, j)[channel] < temp)
				temp = M.at<Vec3b>(i, j)[channel];
	return temp;
}

//对于每个像素值，估计t(x)的值
void t_for_each_pixel(Mat T, Mat R, int A[])
{
	//对R进行滤波时，先进行padding，最小值滤波因此补255，padding之后存储在RR中
	//(论文中滤波器的尺寸为9)

	double omega_param = 0.8;
	int kernel_size = 9;
	int offset = (kernel_size - 1) / 2;

	Mat RR(R.rows + 2 * offset, R.cols + 2 * offset, CV_8UC3, Scalar(255, 255, 255));

	for (int row = 0; row < R.rows; row++)
		for (int col = 0; col < R.cols; col++)
			for (int channel = 0; channel < R.channels(); channel++)
				RR.at<Vec3b>(row + offset, col + offset)[channel] = R.at<Vec3b>(row, col)[channel];

	for (int row = offset; row < (R.rows + offset); row++)
	{
		for (int col = offset; col < (R.cols + offset); col++)
		{
			double temp = 255.0;
			for (int channel = 0; channel < RR.channels(); channel++)
			{
				double filtered_val = (double)minimum_filter(RR, row, col, kernel_size, channel)*1.0;
				filtered_val = filtered_val / ((double)A[channel]);

				if (filtered_val < temp)
					temp = filtered_val;
			}
			T.at<double>(row - offset, col - offset) = 1.0 - omega_param *temp;
		}
	}
	return;
}

void recovery_img(Mat J, Mat R, Mat T, int A[])
{
	for (int row = 0; row < J.rows; row++)
	{
		for (int col = 0; col < J.cols; col++)
		{
			double p = 0;
			if (T.at<double>(row, col) < 0.5 && T.at<double>(row, col) > 0)
				p = 2 * T.at<double>(row, col);
			else
				p = 1;
			for (int channel = 0; channel < J.channels(); channel++)
			{
				double temp = ((double)(R.at<Vec3b>(row, col)[channel]) - A[channel]) / (p*T.at<double>(row, col));
				temp = temp + A[channel];
				J.at<Vec3b>(row, col)[channel] = (uchar)temp;
			}
		}
	}
	return;
}

//导向滤波器
Mat guidedfilter(Mat &srcImage, Mat &srcClone, int r, double eps)
{
	//转换源图像信息	
	srcImage.convertTo(srcImage, CV_64FC1);
	srcClone.convertTo(srcClone, CV_64FC1);
	int nRows = srcImage.rows;
	int nCols = srcImage.cols;
	Mat boxResult;

	//步骤一：计算均值	
	boxFilter(Mat::ones(nRows, nCols, srcImage.type()), boxResult, CV_64FC1, Size(r, r));

	//生成导向均值mean_I	
	Mat mean_I;
	boxFilter(srcImage, mean_I, CV_64FC1, Size(r, r));

	//生成原始均值mean_p	
	Mat mean_p;
	boxFilter(srcClone, mean_p, CV_64FC1, Size(r, r));

	//生成互相关均值mean_Ip	
	Mat mean_Ip;
	boxFilter(srcImage.mul(srcClone), mean_Ip, CV_64FC1, Size(r, r));
	Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);

	//生成自相关均值mean_II	
	Mat mean_II;

	//应用盒滤波器计算相关的值	
	boxFilter(srcImage.mul(srcImage), mean_II, CV_64FC1, Size(r, r));

	//步骤二：计算相关系数	
	Mat var_I = mean_II - mean_I.mul(mean_I);
	Mat var_Ip = mean_Ip - mean_I.mul(mean_p);

	//步骤三：计算参数系数a,b	
	Mat a = cov_Ip / (var_I + eps);
	Mat b = mean_p - a.mul(mean_I);

	//步骤四：计算系数a\b的均值	
	Mat mean_a;	
	boxFilter(a, mean_a, CV_64FC1, Size(r, r));
	mean_a = mean_a / boxResult;
	Mat mean_b;	
	boxFilter(b, mean_b, CV_64FC1, Size(r, r));
	mean_b = mean_b / boxResult;

	//步骤五：生成输出矩阵	
	Mat resultMat = mean_a.mul(srcImage) + mean_b;
	return resultMat;
}

//进行最后的滤波操作
void filter(Mat M, int radius)
{
	vector<Mat> vSrcImage, vResultImage;
	split(M, vSrcImage);
	
	Mat resultMat;
	for (int i = 0; i < 3; i++)
	{
		//分通道转换成浮点型		
		Mat tempImage;
		vSrcImage[i].convertTo(tempImage, CV_64FC1, 1.0 / 255.0);
		Mat p = tempImage.clone();

		//分别进行导向滤波		
		Mat resultImage = guidedfilter(tempImage, p, radius, 0.01);
		vResultImage.push_back(resultImage);
	}
	//通道结果合并	
	merge(vResultImage, resultMat);
	resultMat.convertTo(resultMat, CV_8UC1, 255, 0);

	imshow("result", resultMat);
	imwrite("result.bmp", resultMat);
}

int main()
{
	Mat origion = imread("test_1.bmp");
	if (!origion.data)
	{
		cout << "载入图像失败" << endl;
		return 0;
	}
	cout << "输入图像的尺寸为：" << origion.rows << " " << origion.cols << endl;
	namedWindow("origion", CV_WINDOW_AUTOSIZE);
	imshow("origion", origion);

	//R代表反色之后的图
	Mat R = Mat::zeros(origion.rows, origion.cols, CV_8UC3);
	get_inverted_img(origion, R);

	//A代表了global atmosphere light
	int A[] = { 0,0,0 };
	get_A_value(R, A);
	cout << "A的值为(opencv:b,g,r顺序)" << " ";
	print_arr(A, sizeof(A) / sizeof(int));

	//对于每一个像素点x，计算t(x)的值，T的类型为double
	Mat T = Mat::zeros(origion.rows, origion.cols, CV_64FC1);
	t_for_each_pixel(T, R, A);

	//由反色图R和T，以及A，得到去雾后的图J
	Mat J = Mat::zeros(origion.rows, origion.cols, CV_8UC3);
	recovery_img(J, R, T, A);

	//将J再次反色，得到最终的输出图E
	Mat E = Mat::zeros(origion.rows, origion.cols, CV_8UC3);
	get_inverted_img(J, E);
	imwrite("enhancement_without_guidedfilter.bmp", E);

	//对最终的输出图进行导向滤波，得到增强后的结果E
	int radius = 11;
	filter(E, radius);

	cvWaitKey(0);
	return 0;
}
