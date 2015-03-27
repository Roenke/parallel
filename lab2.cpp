#include <omp.h>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <math.h>
#include <cstdlib>
#include <locale>
using namespace std;

const double THETA = 0.1;
const int MAXITER = 10000;
const double EPS = 1e-7;

void generate(double* &matrix, double* &vector, int size)
{
	int max = size * size;
	double rowSum;
	matrix = new double[max];
	vector = new double[size];
	for (int i = 0; i < size; i++)
	{
		rowSum = 0;
		for (int j = 0; j < size; j++)
		{
			matrix[i * size + j] = i == j ? 1.0 : (rand() % size / static_cast<double>(max));
			rowSum += matrix[i *size + j];
		}
		vector[i] = rowSum;
	}
}

void showUsage()
{
	cerr << "Usage: ./lab2 num_omp_thread task_size\r\n";
	cerr << "(int)num_omp_thread - OMP threads count\r\n";
	cerr << "(int)task_size - Matrix size\r\n";
}

int main(int argc, char** argv)
{
	int num_threads, taskSize;
	int iteration;
	double timeStart;
	double error;
	double rightNorm;
	double* matrix;
	double* result;
	double* vector;
	double* prev;
	double duration;

	if (argc != 3 || sscanf(argv[1], "%d", &num_threads) != 1 || sscanf(argv[2], "%d", &taskSize) != 1)
	{
		showUsage();
		return 1;
	}

	std::cerr << "Start generation.\r\n";
	generate(matrix, vector, taskSize);
	std::cerr << "Generation complete\r\n";

	result = new double[taskSize];
	prev = new double[taskSize];

	rightNorm = 0;
	for (int i = 0; i < taskSize; i++)
	{
		rightNorm += vector[i] * vector[i];
		result[i] = prev[i] = 0.0;
	}

	rightNorm = sqrt(rightNorm);

	error = 1.0;
	iteration = 1;
	timeStart = omp_get_wtime();

	omp_set_num_threads(num_threads);

	#pragma omp parallel
	while (iteration < MAXITER && error > EPS)
	{
		error = 0.0;
		#pragma omp parallel for
		for (int i = 0; i < taskSize; i++)
		{
			result[i] = 0.0;
			#pragma omp parallel for
			for (int j = 0; j < taskSize; j++)
			{
				result[i] += prev[j] * matrix[i * taskSize + j];
			}
			result[i] -= vector[i];
			error += result[i] * result[i];
		}
		error = sqrt(error) / rightNorm;
		#pragma omp parallel for
		for (int i = 0; i < taskSize; i++)
		{
			result[i] = prev[i] - THETA * result[i];
		}

		//std::cerr << "Iteration " << iteration <<". Error = " << error << "\r\n";
		swap(prev, result);
		iteration++;
	}

	swap(prev, result);
	duration = (omp_get_wtime() - timeStart);

	error = 0.0;
	for (int i = 0; i < taskSize; i++)
	{
		error += (result[i] - 1.0) * (result[i] - 1.0);
	} 

	cerr << "Calculation complete.\r\n";
	cerr << "Iterations: " << iteration << "\r\n";
	cerr << "Error: " << error << "\r\n";
	cerr << "Time: " << duration << "\r\n";
	cerr << "[" << result[0] << "," << result[1] << "," << result[2] << ",...," << result[taskSize - 1] << "]\r\n";
}
