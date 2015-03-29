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
const int ITERATIONS = 500;

const int ROOT_NODE = 0xAA;
const int UP_MID_NODE = 0xAB;
const int UP_RIGHT_NODE = 0xAC;
const int MID_RIGHT_NODE = 0xAD;
const int DOWN_RIGHT_NODE = 0xAE;
const int DOWN_MID_NODE = 0xAF;
const int DOWM_LEFT_NODE = 0xBA;
const int MID_LEFT_NODE = 0xBB;
const int MID_MID_NODE = 0xBC;


const double a = 1.0;
double *F;
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

	F = new double[n * m];

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
				F[i * m + j] = solution(x_begin + i * hx, y_begin + j * hy);
			}
		}
	}
}

int getNodeBahavior(int* coords, int dimX, int dimY)
{
	if (coords[0] == 0)
	{
		if (coords[1] == 0) return ROOT_NODE;
		if (coords[1] == dimY - 1) return DOWM_LEFT_NODE;
		return MID_LEFT_NODE;
	}

	if (coords[1] == 0)
	{
		if (coords[0] == dimX - 1) return UP_RIGHT_NODE;
		return UP_MID_NODE;
	}

	if (coords[0] == dimX - 1)
	{
		if (coords[1] == dimY - 1) return DOWN_RIGHT_NODE;
		return MID_RIGHT_NODE;
	}
	if (coords[1] == dimY - 1) return DOWN_MID_NODE;
	return MID_MID_NODE;
}

inline bool need_top(int* coords)
{
	return coords[1] != 0;
}

inline bool need_down(int* coords, int dimY)
{
	return coords[1] != dimY - 1;
}

inline bool need_left(int* coords)
{
	return coords[0] != 0;
}

inline need_right(int* coords, int dimX)
{
	return coords[0] != dimX - 1;
}

void compute(int n, int m, int dimX, int dimY)
{
	int rank;
	int localsizeX, localsizeY;
	int dims[NUM_DIMS];
	int periods[NUM_DIMS];
	int coords[NUM_DIMS];
	int remains[NUM_DIMS];
	int *disp, *count;

	double hx, hy;
	double owx, owy;
	double* localF;
	double *top, *down;
	double *left, *righ;
	MPI_Comm comm_2D;
	MPI_Comm comm_x, comm_y;
	MPI_Aint sizeOfDouble;
	MPI_Datatype typeBlock, typeColumn;
	MPI_Status status;
	MPI_Request request;

	hx = (x_end - x_begin) / n;
	hy = (y_end - y_begin) / m;

	owx = hx * hx;
	owy = hy * hy;
	double c = 2 / owx + 2 / owy + a;

	localsizeX = n / dimX;
	localsizeY = m / dimY;
	localF = new double[localsizeX * localsizeY];

	top = new double[localsizeX];
	down = new double[localsizeX];
	left = new double[localsizeY];
	righ = new double[localsizeY];
	for (int i = 0; i < localsizeX; i++)
	{
		down[i] = righ[i] = 0.0;
	}

	dims[0] = dimX;
	dims[1] = dimY;
	periods[0] = periods[1] = 0;
	MPI_Cart_create(MPI_COMM_WORLD, NUM_DIMS, dims, periods, 0, &comm_2D);

	MPI_Comm_rank(comm_2D, &rank);
	MPI_Cart_coords(comm_2D, rank, NUM_DIMS, coords);

	MPI_Type_extent(MPI_DOUBLE, &sizeOfDouble);
	MPI_Type_vector(localsizeY, localsizeX, n, MPI_DOUBLE, &typeBlock);
	MPI_Type_create_resized(typeBlock, 0, sizeOfDouble * localsizeX, &typeBlock);
	MPI_Type_commit(&typeBlock);

	MPI_Type_vector(localsizeY, 1, localsizeX, MPI_DOUBLE, &typeBlock);
	MPI_Type_create_resized(typeBlock, 0, sizeOfDouble * localsizeX, &typeColumn);
	MPI_Type_commit(&typeColumn);

	remains[0] = 1; remains[1] = 0;
	MPI_Cart_sub(comm_2D, remains, &comm_x);
	swap(remains[0], remains[1]);
	MPI_Cart_sub(comm_2D, remains, &comm_y);
	

	disp = new int[dimX * dimY];
	count = new int[dimX * dimY];

	double local_begin_x = x_begin + coords[0] * localsizeX * hx;
	double local_begin_y = y_begin + coords[1] * localsizeY * hy;
	
	for (int i = 0; i < dimX; i++)
	{
		for (int j = 0; j < dimY; j++)
		{
			disp[i * dimY + j] = i* dimY * localsizeX + j;
			count[i * dimY + j] = 1;
		}
	}

	MPI_Scatterv(F, count, disp, typeBlock, localF, localsizeX * localsizeY, MPI_DOUBLE, 0, comm_2D);
	
	int last_x, last_y;
	bool f;
	double F1, Fi, Fj;
	
	int it = 0;
	for (int iteration = 0; iteration < ITERATIONS; iteration ++)
	{
		if (rank != ROOT)
		{
			if (need_top(coords))
				MPI_Recv(top, localsizeX, MPI_DOUBLE, coords[1] - 1, 0, comm_y, &status);
			if (need_left(coords))
				MPI_Recv(left, localsizeY, MPI_DOUBLE, coords[0] - 1, 0, comm_x, &status);
		}

		last_x = localsizeX - 1;
		last_y = localsizeY - 1;

		if (coords[0] == 0 || coords[1] == 0)
		{
			// Верхний левый угол
			Fi = (localF[1] + left[0]) / owx;
			Fj = (localF[localsizeX] + top[0]) / owy;
			localF[0] = (Fi + Fj - right(local_begin_x, local_begin_y)) / c;
		}
		if (coords[0] == 0)
		{
			// Верхняя внутренняя линия
			for (int i = 1; i < last_x; i++)
			{
				Fi = (localF[i + 1] + localF[i - 1]) / owx;
				Fj = (localF[localsizeX + i] + top[i]) / owy;
				localF[i] = (Fi + Fj - right(local_begin_x + i * hx, local_begin_y)) / c;
			}
		}

		if (coords[1] == 0)
		{
			// Левая внутренняя линия
			for (int i = 1; i < last_y; i++)
			{
				Fi = (localF[i * localsizeX + 1] + left[i]) / owx;
				Fj = (localF[(i + 1) * localsizeX] + localF[(i - 1) * localsizeX]) / owy;
				localF[i * localsizeX] = (Fi + Fj - right(local_begin_x, local_begin_y + i * hy)) / c;
			}
		}

		if (coords[0] == 0 || coords[1] == dimX - 1)
		{
			// Верхний правый угол
			Fi = (righ[0] + localF[last_x - 1]) / owx;
			Fj = (top[last_x] + localF[localsizeX + last_x]) / owy;
			localF[last_x] = (Fi + Fj - right(local_begin_x + last_x * hx, local_begin_y)) / c;
		}

		if (need_top(coords))
		{
			MPI_Isend(localF, localsizeX, MPI_DOUBLE, coords[1] - 1, 0, comm_y, &request);
		}
		if (need_left(coords))
		{
			MPI_Isend(localF, 1, typeColumn, coords[0] - 1, 0, comm_x, &request);
		}

		// Середина
		for (int i = 1; i < last_y; i++)
		{
			for (int j = 1; j < last_x; j++)
			{
				Fi = (localF[i *localsizeX + j - 1] + localF[i * localsizeX + j + 1]) / owx;
				Fj = (localF[(i - 1) * localsizeX + j] + localF[(i + 1) * localsizeX + j]) / owy;
				localF[i * localsizeX + j] = (Fi + Fj - right(local_begin_y + i*hx, local_begin_y + j*hy)) / c;
			}
		}

		if (need_down(coords, dimY))
		{
			// Нижняя внутренняя линия
			for (int i = 1; i < last_x; i++)
			{
				Fi = (localF[localsizeX * last_y * i + 1] + localF[localsizeX * last_y * i - 1]) / owx;
				Fj = (localF[localsizeX * (last_y - 1) + i] + down[i]) / owy;
				localF[last_y * localsizeX + i] = (Fi + Fj - right(local_begin_x + hx * i, local_begin_y + last_y * hy)) / c;
			}
		}

		if (need_right(coords, dimX))
		{
			// Правая внутренняя линия
			for (int i = 1; i < last_y; i++)
			{
				Fi = (localF[i * localsizeX + last_x - 1] + righ[i]) / owx;
				Fj = (localF[(i + 1) * localsizeX + last_x] + localF[(i - 1) * localsizeX + last_x]) / owy;
				localF[i * localsizeX + last_x] = (Fi + Fj - right(local_begin_x + last_x * hx, local_begin_y + i * hy)) / c;
			}
		}

		if (need_right(coords, dimX) || need_down(coords, dimY))
		{
			// Нижний правый угол
			Fi = (righ[last_y] + localF[last_y * localsizeX + last_x - 1]) / owx;
			Fj = (down[last_x] + localF[(last_y - 1) * localsizeX + last_x]) / owy;
			localF[last_y * localsizeX + last_x] = (Fi + Fj - right(local_begin_x + last_x * hx, local_begin_y + last_y * hy)) / c;
		}	

		if (need_right(coords, dimX))
		{
			MPI_Send(localF + last_x, 1, typeColumn, coords[0] + 1, 0, comm_x);
			MPI_Recv(righ, 1, typeColumn, coords[0] + 1, 0, comm_x);
		}
		if (need_down(coords, dimY))
		{
			MPI_Send(localF + localsizeX * last_y, localsizeX, MPI_DOUBLE, coords[1] + 1, 0, comm_y);
		}

	}

	MPI_Gatherv(localF, localsizeX * localsizeY, MPI_DOUBLE, F, count, disp, typeBlock, 0, comm_2D);
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
			result &= (n + 1) % dimX == 0 || (m + 1) % dimY == 0;
			if (!result || dimX * dimY != size)
			{
				showUsage();
				return 1;
			}

			init(n + 1, m + 1);
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

	double timeStart = MPI_Wtime();
	compute(n + 1, m + 1, dimX, dimY);
	MPI_Barrier(MPI_COMM_WORLD);
}