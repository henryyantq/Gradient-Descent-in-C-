#include <iostream>
#include <cmath>
#define D 500	//vector dimensions
#define N 500	//numbers of vectors
#define ITER 500000		//max. iterations

using namespace std;

inline double lossF(double* w, double b, double** pts);
inline double fitCurve(double* w, double b, double* x);
inline double random(double min, double max);

int main() {
	double** points = new double* [N];
	for (int i = 0; i < N; i++)
		points[i] = new double[D];
	for (int i = 0; i < N; i++)
		for (int j = 0; j < D; j++)
			points[i][j] == random(0, 255);
	double* weights = new double[D - 1];
	for (int i = 0; i < D - 1; i++)
		weights[i] = random(0, 1000);
	double bias = random(0, 1000);
	double* grad = new double[D];
	double lr = 0.0001;
	for (int i = 0; i < ITER; i++) {
		for (int j = 0; j < D - 1; j++) {
			double* weights_tmp = weights;
			weights_tmp[j] += lr;
			grad[j] = (lossF(weights_tmp, bias, points) - lossF(weights, bias, points)) / lr;
		}
		grad[D - 1] = (lossF(weights, bias + lr, points) - lossF(weights, bias, points)) / lr;
		for (int j = 0; j < D - 1; j++) {
			weights[j] -= lr * grad[j];
		}
		bias -= lr * grad[D - 1];
		int flag = 0;
		for (int j = 0; j < D; j++)
			if (grad[j] < lr) flag++;
		if (flag == D) break;
	}
	cout << "Loss = " << lossF(weights, bias, points) << endl;
}

inline double fitCurve(double* w, double b, double* x) {
	double sum = 0;
	for (int i = 0; i < D - 1; i++)
		sum += w[i] * x[i];
	sum += b;
	return sum;
}

inline double lossF(double* w, double b, double** pts) {
	double _y;
	double sum = 0;
	for (int i = 0; i < N; i++) {
		_y = fitCurve(w, b, pts[i]);
		double loss = pts[i][D - 1] - _y;
		sum += pow(loss, 2);
	}
	return sum;
}

inline double random(double min, double max) {
	return (double)(rand() % (int)(max - min + 1)) + min;
}
