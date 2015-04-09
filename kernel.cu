
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <conio.h>
#include <iostream>
#include <algorithm> 
#include <time.h>
#include <stdio.h>
using namespace std;

const int MAXITER = 100000;
const int BLOCK_SIZE = 32;
const int N = 32 * 32;					// Количество внутренних узлов сетки
const int SIZE = N + 2;					// Общее количество узлов в сетке
float LENGTH = 10;						// Длина интервала расчета
const float h = LENGTH / (SIZE - 1);	// Величина шага сетки
const float a = 1.0;					// Параметр дифф. уравнения
float F[SIZE][SIZE];					// Матрица значений функции на заданной сетке

const float h_sq = h * h;
const float c = 4.0 / h_sq + a;

__constant__ float constants[3];

__device__
float r(float x, float y)
{
	return - (x + y);
}

__global__ void kernel(float* prev, float* current, int* end)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int i = blockIdx.y * blockDim.x + threadIdx.y + 1;
	float Fi = (prev[(i - 1) * SIZE + j] + prev[(i + 1) * SIZE + j]) / constants[1];
	float Fj = (prev[i * SIZE + j - 1] + prev[i * SIZE + j + 1]) / constants[1];
	current[i * SIZE + j] = (Fi + Fj - r(i * constants[0], j * constants[0])) / constants[2];
	if (fabs(current[i * SIZE + j] - prev[i * SIZE + j]) > 1e-5)
	{
		end[0] = 1;
	}
}

__host__
float solution(float x, float y)
{
	return x + y;
}

__host__
void Init()
{
	int i, j;
	for (i = 0; i < SIZE; i++)
	{
		for (j = 0; j < SIZE; j++)
		{
			if ((i != 0) && (j != 0) && (i != SIZE - 1) && (j != SIZE - 1))
			{
				F[i][j] = 0;
			}
			else
			{
				F[i][j] = solution(i * h, j * h);
			}
		}
	}
}


int main(int argc, char * argv[])
{
	float * prev = NULL;
	float * current = NULL;
	int* end;
	clock_t start;
	double duration;
	
	int* complete = new int[1];
	Init();
	float * ar = new float[3];
	ar[0] = h;
	ar[1] = h_sq;
	ar[2] = c;
	cudaMemcpyToSymbol(constants, ar, 3 * sizeof(float));
	cudaMalloc((void**)&prev, SIZE * SIZE * sizeof (float));
	cudaMalloc((void**)&current, SIZE * SIZE * sizeof (float));
	cudaMalloc((void**)&end, sizeof(int));

	cudaMemcpy(prev, F, SIZE * SIZE * sizeof (float), cudaMemcpyHostToDevice);
	cudaMemcpy(current, F, SIZE * SIZE * sizeof (float), cudaMemcpyHostToDevice);
	start = clock();
	
	int iteration = 1;
	do
	{
		cudaMemset(end, 0, 1);
		kernel << <dim3(N / BLOCK_SIZE, N / BLOCK_SIZE, 1), dim3(BLOCK_SIZE, BLOCK_SIZE, 1) >> > (prev, current, end);
		cudaMemcpy(complete, end, sizeof (int), cudaMemcpyDeviceToHost);
		swap(prev, current);
		iteration++;

	} while (iteration < MAXITER);

	cudaMemcpy(F, prev, SIZE * SIZE * sizeof (float), cudaMemcpyDeviceToHost);
	cudaFree(prev);
	float maxError = 0;
	for (int i = 0; i < SIZE; i++)
	{
		for (int j = 0; j < SIZE; j++)
		{
			if (fabs(F[i][j] - solution(i * h, j * h)) > maxError)
			{
				maxError = fabs(F[i][j] - solution(i * h, j * h));
			}
		}
	}

	duration = (clock() - start) / (double)CLOCKS_PER_SEC;

	std::cout << "time: " << duration << '\n';
	printf("Iterations = %d\nError = %f", iteration, maxError);
	return 0;
}