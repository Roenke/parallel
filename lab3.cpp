#include <mpi.h>
#include <math.h>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
using namespace std;

const int ROOT = 0;
const int PARAMETERS_COUNT = 6;

void fillMatrix(double* &a, double* &b, double* &c, int n, int m, int k)
{
	a = new double[n * m];
	b = new double[m * k];
	c = new double[n * k];

	double div = static_cast<double>(n);

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			a[i * n + j] = (j + i) / div;
		}

		for (int j = 0; j < k; j++)
		{
			c[i * n + j] = 0.0;
		}
	}

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < k; j++)
		{
			b[i * m + j] = (2 * i + 3 * j) / div;
		}
	}
}

void showUsage()
{
	cerr << "Usage: ./lab3 n m k dimX dimY \r\n";
	cerr << "(int)n - first matrix rows count\r\n";
	cerr << "(int)m - first matrix rows (second matrix colunms) count\r\n";
	cerr << "(int)k - second matrix columns count\r\n";
	cerr << "(int)dimX - First dimension node count\r\n";
	cerr << "(int)dimY - Second dimension node count\r\n";
}

int main(int argc, char** argv)
{
	// Переменные, которые есть во всех узлах
	int n, m, k;
	int dimX, dimY;
	int rank, size;

	// Переменные только для первого узла
	double *a, *b, *c;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	if (rank == ROOT){
		if (argc != PARAMETERS_COUNT)
		{
			bool result = true;
			result &= sscanf(argv[1], "%d", &n) && sscanf(argv[2], "%d", &m) && sscanf(argv[3], "%d", &k);
			result &= sscanf(argv[4], "%d", &dimX) && sscanf(argv[5], "%d", &dimY);
			if (!result || dimX * dimY != size)
			{
				showUsage();
				return 1;
			}

			fillMatrix(a, b, c, m, n, k);
		}
	}

}