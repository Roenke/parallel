#include <time.h>
#include <stdio.h>
#include <conio.h>
#include <iostream>
using namespace std;

const int MAXITER = 1000000;
const int N = 32 * 32;						// Количество внутренних узлов сетки
const int SIZE = N + 2;					// Общее количество узлов в сетке
float LENGTH = 10;				// Длина интервала расчета
const float h = LENGTH / (SIZE - 1);	// Величина шага сетки
const float a = 1.0;					// Параметр дифф. уравнения
float F[SIZE][SIZE];					// Матрица значений функции на заданной сетке
float Fprev[SIZE][SIZE];

const float h_sq = h * h;
const float c = 4.0 / h_sq + a;


float r(float x, float y)
{
	return -(x + y);
}

float solution(float x, float y)
{
	return x + y;
}

void Init()
{
	int i, j, k;
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
				Fprev[i][j] = F[i][j];
			}
		}
	}
}

int main(int argc, char * argv[])
{
	clock_t start;
	double duration;
	bool complete = false;
	Init();

	start = clock();
	int iteration;
	for (iteration = 1; complete == false && iteration < MAXITER; iteration++)
	{
		complete = true;
		for (int i = 1; i <= N; i++)
		{
			for (int j = 1; j <= N; j++)
			{
				float Fi = (Fprev[i - 1][j] + Fprev[i + 1][j]) / h_sq;
				float Fj = (Fprev[i][j - 1] + Fprev[i][j + 1]) / h_sq;
				F[i][j] = (Fi + Fj - r(i * h, j * h)) / c;
				if (fabs(F[i][j] - Fprev[i][j]) > 1e-5)
				{
					complete = false;
				}
			}
		}
		swap(Fprev, F);
	}

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
	std::cout << "printf: " << duration << '\n';
	printf("\nError = %f, Iteration = %d", maxError, iteration);
	_getch();
	return 0;
}