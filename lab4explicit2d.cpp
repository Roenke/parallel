#define _CRT_SECURE_NO_WARNINGS
#include <mpi.h>
#include <math.h>
#include <iostream>
#include <string>
#include <stdio.h>
using namespace std;

/* Пусть область будет фиксирована  [0, 10] x [0, 10]*/
const double x_begin = 0.0;
const double y_begin = 0.0;
const double x_end = 10.0;
const double y_end = 10.0;

const int NUM_DIMS = 2;
const int ROOT = 0;
const int PARAMETERS_COUNT = 5;
const int MAXITER = 200;
const double EPS = 1e-7;

const double a = 1.0;
double *F, *Fprev;
/* Функция определения точного решения */
double solution(double x, double y)
{
	double res;
	res = x + y;
	return res;
}

/* Функция задания правой части уравнения */
double right(double x, double y)
{
	double d;
	d = -a * (x + y);
	return d;
}

/* Подпрограмма инициализации границ 3D пространства */
void init(int n, int m)
{
	int i, j;
	double hx = (x_end - x_begin) / (n - 1);
	double hy = (y_end - y_begin) / (m - 1);
	double tmp;
	F = new double[n * m];
	Fprev = new double[n * m];

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < m; j++)
		{
			if ((i != 0) && (j != 0) && (i != n - 1) && (j != m - 1))
			{
				F[i * m + j] = 0;
			}
			else
			{
				tmp = solution(x_begin + i * hx, y_begin + j * hy);
				Fprev[i * m + j] = F[i * m + j] = tmp;
			}
		}
	}
}


void compute(int n, int m, int dimX, int dimY)
{
	int rank;
	int localsizeX, localsizeY;
	int localResultSizeX, localResultSizeY;
	int dims[NUM_DIMS];
	int periods[NUM_DIMS];
	int coords[NUM_DIMS];
	int remains[NUM_DIMS];
	int *disp, *count;
	int size;

	double hx, hy;
	double owx, owy;
	double* localF;
	double* localResult;
	MPI_Comm comm_2D;
	MPI_Comm comm_x, comm_y;
	MPI_Aint sizeOfDouble;
	MPI_Datatype typeBlock, typeResultBlock;
	MPI_Status status;
	MPI_Request request;

	hx = (x_end - x_begin) / (n - 1);
	hy = (y_end - y_begin) / (m - 1);

	owx = hx * hx;
	owy = hy * hy;
	double c = 2 / owx + 2 / owy + a;

	localsizeX = (n - 2) / dimX + 2;
	localsizeY = (m - 2) / dimY + 2;
	localResultSizeX = localsizeX - 2;
	localResultSizeY = localsizeY - 2;

	localF = new double[localsizeX * localsizeY];
	localResult = new double[localResultSizeX * localResultSizeY];

	MPI_Comm_size(MPI_COMM_WORLD, &size);

	dims[0] = dimX;
	dims[1] = dimY;
	periods[0] = periods[1] = 0;
	MPI_Cart_create(MPI_COMM_WORLD, NUM_DIMS, dims, periods, 0, &comm_2D);

	MPI_Comm_rank(comm_2D, &rank);
	MPI_Cart_coords(comm_2D, rank, NUM_DIMS, coords);

	MPI_Type_extent(MPI_DOUBLE, &sizeOfDouble);
	MPI_Type_vector(localsizeY, localsizeX, n, MPI_DOUBLE, &typeBlock);
	MPI_Type_create_resized(typeBlock, 0, sizeOfDouble, &typeBlock);
	MPI_Type_commit(&typeBlock);

	MPI_Type_vector(localResultSizeY, localResultSizeX, n, MPI_DOUBLE, &typeResultBlock);
	MPI_Type_create_resized(typeResultBlock, 0, sizeOfDouble, &typeResultBlock);
	MPI_Type_commit(&typeResultBlock);

	disp = new int[dimX * dimY];
	count = new int[dimX * dimY];

	double localBeginX = x_begin + (coords[0] * localResultSizeX + 1) * hx;
	double localBeginY = y_begin + (coords[1] * localResultSizeY + 1) * hy;

	for (int i = 0; i < dimX; i++)
	{
		for (int j = 0; j < dimY; j++)
		{
			disp[i * dimY + j] = i * n * (localResultSizeY) + j * (localResultSizeX);
			count[i * dimY + j] = 1;
		}
	}

	int needStop = 0;
	int needStopTmp = 0;
	double F1, Fi, Fj;
	int iteration = 1;
	double error = EPS + 1;
	while (!needStop && iteration <= MAXITER)
	{
		MPI_Scatterv(F, count, disp, typeBlock, localF, localsizeX * localsizeY, MPI_DOUBLE, 0, comm_2D);

		for (int i = 1; i < localsizeY - 1; i++)
		{
			for (int j = 1; j < localsizeX - 1; j++)
			{
				Fi = (localF[i * localsizeX + j - 1] + localF[i * localsizeX + j + 1]) / owx;
				Fj = (localF[(i - 1) * localsizeX + j] + localF[(i + 1) * localsizeX + j]) / owy;
				localResult[(i - 1) * (localResultSizeX) + j - 1] = (Fi + Fj - right(localBeginX + (i - 1) * hx, localBeginY + (j - 1) * hy)) / c;
			}
		}

		MPI_Allreduce(&needStopTmp, &needStop, 1, MPI_INT, MPI_PROD, MPI_COMM_WORLD);
		swap(needStop, needStopTmp);

		MPI_Gatherv(localResult, localResultSizeX * localResultSizeY, MPI_DOUBLE, &Fprev[n + 1], count, disp, typeResultBlock, 0, comm_2D);
		swap(F, Fprev);
		
		iteration++;
	}

	swap(F, Fprev);
	
}

void showUsage()
{
	cerr << "Usage: ./lab4 n m dimX dimY \r\n";
	cerr << "(int)n - x - section count\r\n";
	cerr << "(int)m - y - section count\r\n";
	cerr << "(int)dimX - First dimension node count\r\n";
	cerr << "(int)dimY - Second dimension node count\r\n";
}

int main(int argc, char** argv)
{
	double max, N, t1, t2;
	double owz, c, e;
	double Fi, Fj, Fk, F1;
	int rank;
	int size;

	int n, m;
	int dimX, dimY;
	int i, j, k, mi, mj, mk;
	int R, fl, fl1, fl2;
	int it, f;
	long int osdt;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (rank == ROOT){
		if (argc == PARAMETERS_COUNT)
		{
			bool result = true;
			result &= sscanf(argv[1], "%d", &n) && sscanf(argv[2], "%d", &m);
			result &= sscanf(argv[3], "%d", &dimX) && sscanf(argv[4], "%d", &dimY);
			result &= n % dimX == 0 || m % dimY == 0;
			if (!result || dimX * dimY != size)
			{
				showUsage();
				return 1;
			}

			init(n + 2, m + 2);
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
	MPI_Bcast(&dimX, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&dimY, 1, MPI_INT, 0, MPI_COMM_WORLD);

	double timeStart;
	if (rank == ROOT)
	{
		timeStart = MPI_Wtime();
	}
	compute(n + 2, m + 2, dimX, dimY);

	if (rank == ROOT)
	{
		std::cerr << MPI_Wtime() - timeStart << "\r\n";
	}

	MPI_Finalize();
}