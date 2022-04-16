
#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>

#define W_FILT 3
#define INIT 0.25
#define LR 0.0001
#define ITER 500000

using namespace std;
using namespace cv;

inline double func(double* w, double* x) {
	double sum = 0;
	for (int i = 0; i < W_FILT * W_FILT; i++)
		sum += w[i] * x[i];
	if (sum > 255) sum = 255;
	else if (sum < 0) sum = 0;
	return sum;
}

inline double lossFunc(double* w, double** pts, double* Y, int N) {
	double sum = 0;
	for (int i = 0; i < N; i++) {
		double _y = func(w, pts[i]);
		double y = Y[i];
		sum += pow(y - _y, 2);
	}
	return sum;
}

int main() {
	Mat input_src = imread("D:/FILTER_ML/SDR_Compressed.png");
	Mat output_dst = imread("D:/FILTER_ML/HDR_Compressed.png");
	int inputVectsCount = (input_src.rows - (W_FILT - 1)) * (input_src.cols - (W_FILT - 1));
	int matSize = input_src.rows * input_src.cols;
	vector<Mat> input_channels, output_channels;
	double* flattened_input_y_channel = new double[matSize];
	double* flattened_output_y_channel = new double[inputVectsCount];
	split(input_src, input_channels);
	split(output_dst, output_channels);
	for (int i = 0; i < input_src.rows; i++)
		for (int j = 0; j < input_src.cols; j++)
			flattened_input_y_channel[i * input_src.cols + j] = (double)input_channels[0].at<uchar>(i, j);
	for (int i = (W_FILT - 1) / 2; i < output_dst.rows - (W_FILT - 1) / 2; i++)
		for (int j = (W_FILT - 1) / 2; j < output_dst.cols - (W_FILT - 1) / 2; j++)
			flattened_output_y_channel[(i - (W_FILT - 1) / 2) * (output_dst.cols - (W_FILT - 1)) + (j - (W_FILT - 1) / 2)] = (double)output_channels[0].at<uchar>(i, j);
	double** inputVects = new double* [inputVectsCount];
	for (int i = 0; i < inputVectsCount; i++)
		inputVects[i] = new double[W_FILT * W_FILT];
	for (int i = 0; i < inputVectsCount; i++)
		for (int j = 0; j < W_FILT * W_FILT; j++) {
			inputVects[i][j] = flattened_input_y_channel[i + j];
		}
	double* flattened_filter = new double[W_FILT * W_FILT];
	double begin = clock();
	for (int i = 0; i < W_FILT * W_FILT; i++)
		flattened_filter[i] = INIT;
	double lr = LR;
	double* grad = new double[W_FILT * W_FILT];
	int flag = 0;
	for (int i = 0; i < ITER; i++) {
		for (int j = 0; j < W_FILT * W_FILT; j++) {
			double* filter_copy = new double[W_FILT * W_FILT];
			for (int k = 0; k < W_FILT * W_FILT; k++)
				filter_copy[k] = flattened_filter[k];
			filter_copy[j] += lr;
			grad[j] = (lossFunc(filter_copy, inputVects, flattened_output_y_channel, inputVectsCount) - lossFunc(flattened_filter, inputVects, flattened_output_y_channel, inputVectsCount)) / 2;
			delete[] filter_copy;
		}
		for (int j = 0; j < W_FILT * W_FILT; j++)
			flattened_filter[j] -= lr * grad[j];
		for (int j = 0; j < W_FILT * W_FILT; j++)
			if (grad[j] <= lr) flag++;
		if (flag == W_FILT * W_FILT) break;
		cout << "Loss = " << lossFunc(flattened_filter, inputVects, flattened_output_y_channel, inputVectsCount) << endl;
	}
	double end = clock();
	double dur = (end - begin) / CLOCKS_PER_SEC;
	cout << endl << "Duration: " << dur << endl;
	cout << endl << "Filter: " << endl;
	for (int i = 0; i < W_FILT * W_FILT; i++) {
		cout << flattened_filter[i] << ' ';
		if ((i + 1) % W_FILT == 0) cout << endl;
	}
	cout << endl << "Loss = " << lossFunc(flattened_filter, inputVects, flattened_output_y_channel, inputVectsCount) << endl;
}
