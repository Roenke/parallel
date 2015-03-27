#define _CRT_SECURE_NO_WARNINGS
#include <mpi.h>
#include <math.h>
#include <iostream>
#include <string>
#include <stdio.h>
using namespace std;

const int NUM_DIMS = 2;
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
			a[i * m + j] = 1 + i;
		}

		for (int j = 0; j < k; j++)
		{
			c[i * k + j] = 0.0;
		}
	}

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < k; j++)
		{
			b[i * k + j] = 1.0 + j;
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

void matrixProd(int n, int m, int k, double *A, double *B, double *C, int dimX, int dimY, MPI_Comm comm)
{
	int localsizeA, localsizeB;
	double *localA, *localB, *localC;
	int coords[2];
	int dims[NUM_DIMS];
	int periods[NUM_DIMS];
	int remains[NUM_DIMS];
	int size;
	int rank, globalRank;
	MPI_Aint sizeOfDouble;
	MPI_Comm comm_2D, comm_1D[2], pcomm;
	MPI_Datatype typeB, typeC;
	int *countb, *countc;
	int *dispb, *dispc;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &globalRank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_dup(comm, &pcomm);

	dims[0] = dimX;
	dims[1] = dimY;
	periods[0] = periods[1] = 0;
	MPI_Cart_create(pcomm, NUM_DIMS, dims, periods, 0, &comm_2D);

	MPI_Comm_rank(comm_2D, &rank);
	MPI_Cart_coords(comm_2D, rank, NUM_DIMS, coords);

	localsizeA = n / dimX;
	localsizeB = k / dimY;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 2; j++)
			remains[j] = (i == j);
		MPI_Cart_sub(comm_2D, remains, &comm_1D[i]);
	}

	localA = new double[localsizeA * m];
	localB = new double[localsizeB * m];
	localC = new double[localsizeA * localsizeB];
	
	MPI_Type_extent(MPI_DOUBLE, &sizeOfDouble);

	MPI_Type_vector(m, localsizeB, k, MPI_DOUBLE, &typeB);
	MPI_Type_create_resized(typeB, 0, sizeOfDouble * localsizeB, &typeB);
	MPI_Type_commit(&typeB);

	MPI_Type_vector(localsizeA, localsizeB, k, MPI_DOUBLE, &typeC);
	MPI_Type_create_resized(typeC, 0, sizeOfDouble * localsizeB, &typeC);
	MPI_Type_commit(&typeC);

	dispb = new int[dimY];
	countb = new int[dimY];
	for (int i = 0; i < dimY; i++)
	{
		dispb[i] = i;
		countb[i] = 1;
	}

	dispc = new int[dimX * dimY];
	countc = new int[dimX * dimY];

	for (int i = 0; i < dimX; i++)
	{
		for (int j = 0; j < dimY; j++)
		{
			dispc[i * dimY + j] = i* dimY * localsizeA + j;
			countc[i * dimY + j] = 1;
		}
	}

	if (coords[1] == ROOT)
		MPI_Scatter(A, localsizeA * m, MPI_DOUBLE, localA, localsizeA * m, MPI_DOUBLE, 0, comm_1D[0]);

	if (coords[0] == ROOT)
		MPI_Scatterv(B, countb, dispb, typeB, localB, localsizeB * m, MPI_DOUBLE, 0, comm_1D[1]);

	MPI_Barrier(comm_2D);

	MPI_Bcast(localA, localsizeA * m, MPI_DOUBLE, 0, comm_1D[1]);
	MPI_Bcast(localB, localsizeB * m, MPI_DOUBLE, 0, comm_1D[0]);

	for (int i = 0; i < localsizeA; i++)
	{
		for (int j = 0; j < localsizeB; j++)
		{
			localC[localsizeB * i + j] = 0.0;
			for (int l = 0; l < m; l++)
			{
				localC[localsizeB * i + j] += localA[m* i + l] * localB[localsizeB * l + j];
			}
		}
	}

	MPI_Gatherv(localC, localsizeA * localsizeB, MPI_DOUBLE, C, countc, dispc, typeC, 0, comm_2D);
}

int main(int argc, char** argv)
{
	// ѕеременные, которые есть во всех узлах
	int n, m, k;
	int dimX, dimY;
	int rank, size;
	int dims[NUM_DIMS], periods[NUM_DIMS];
	// ѕеременные только дл€ первого узла
	double *a, *b, *c;
	a = b = c = NULL;
	MPI_Comm comm;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	for (int i = 0; i < NUM_DIMS; i++) { dims[i] = 0; periods[i] = 0; }
	/* «аполн€ем массив dims, где указываютс€ размеры двумерной решетки */
	MPI_Dims_create(size, NUM_DIMS, dims);
	/* —оздаем топологию "двумерна€ решетка" с communicator(ом) comm */
	MPI_Cart_create(MPI_COMM_WORLD, NUM_DIMS, dims, periods, 0, &comm);

	if (rank == ROOT){
		if (argc == PARAMETERS_COUNT)
		{
			bool result = true;
			result &= sscanf(argv[1], "%d", &n) && sscanf(argv[2], "%d", &m) && sscanf(argv[3], "%d", &k);
			result &= sscanf(argv[4], "%d", &dimX) && sscanf(argv[5], "%d", &dimY);
			if (!result || dimX * dimY != size)
			{
				showUsage();
				return 1;
			}

			fillMatrix(a, b, c, n, m, k);
		}
		else
		{
			showUsage();
			return 1;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&dimX, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&dimY, 1, MPI_INT, 0, MPI_COMM_WORLD);

	double timeStart = MPI_Wtime();
	matrixProd(n, m, k, a, b, c, dimX, dimY, comm);
	if (rank == ROOT)
	{
		/*
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < k; j++)
			{
				cerr<< c[i * k + j] << " ";
			}
			cerr << "\r\n";
		}
		*/
		cout << "Total time = " << MPI_Wtime() - timeStart << "\r\n";
	}

	MPI_Finalize();
	return 0;
}