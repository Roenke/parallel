#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <iostream>

using namespace std;

const int ROOT = 0;
const int MAXITER = 500;
const double EPS = 1e-5;

const int Im = 101;
const int Jm = 101;
const int Km = 101;

const int Xm = 100;
const int Ym = 100;
const int Zm = 100;

const int NUM_DIMS = 2;
const int P0 = 2;
const int P1 = 2;

const double a = 1.0;

const int SIZE = 4;

const int REORD = 1;

inline double min(double x, double y)
{
	return x > y ? y : x;
}

inline double max(double x, double y)
{
	return x > y ? x : y;
}
inline double fFi(float x, float y, float z) {
	return x + y + z;
}

inline double fRo(float x, float y, float z) {
	return -a*(x + y + z);
}


int main(int argc, char ** argv) {
	int rank, size;
	MPI_Status st;

	int dims_2D[NUM_DIMS], period_2D[NUM_DIMS];

	MPI_Comm grid_2D;

	for (int i = 0; i < NUM_DIMS; i++)
		period_2D[i] = 0;

	dims_2D[0] = P0;
	dims_2D[1] = P1;


	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	double Fi[(Im - 1) / P1 + 2][(Jm - 1) / P0 + 2][Km + 1];
	double Ro[(Im - 1) / P1 + 2][(Jm + 1) / P0 + 2][Km + 1];

	double M[(Im - 1) / P1 + 2][Km + 1];
	double hx = static_cast<double>(Xm) / Im;
	double hy = static_cast<double>(Ym) / Jm;
	double hz = static_cast<double>(Zm) / Km;
	int i, j, k;
	int i2, j2,
		Im2 = (Im - 1) / P1 + 1,
		Jm2 = (Jm - 1) / P0 + 1;

	/* Init data */
	for (i = 0; i <= Im2; i++)
	{
		for (j = 0; j <= Jm2; j++)
		{
			for (k = 0; k <= Km; k++)
			{
				i2 = (rank%P1)*(Im2 - 1) + i;
				j2 = (rank / P1)*(Jm2 - 1) + j;
				if (i2 == 0 || i2 == Im || j2 == 0 || j2 == Jm || k == 0 || k == Km)
				{
					Fi[i][j][k] = fFi(hx*i2, hy*j2, hz*k);
				}
				else
				{
					Fi[i][j][k] = 0.0;
				}
					
				Ro[i][j][k] = fRo(hx*i2, hy*j2, hz*k);
			}
		}
	}

	double C = 1 / (2 * (1 / (hx * hx) + 1 / (hy * hy) + 1 / (hz * hz)) + a);
	double maxdif = 100.0, Fiold;
	int niter = 0;
	int maxitt = 500;
	int rank_2D;
	int coords[NUM_DIMS];

	MPI_Cart_create(MPI_COMM_WORLD, NUM_DIMS, dims_2D, period_2D, REORD, &grid_2D);
	MPI_Comm_rank(grid_2D, &rank_2D);
	MPI_Cart_coords(grid_2D, rank_2D, NUM_DIMS, coords);

	int left, right, upper, down;

	MPI_Cart_shift(grid_2D, 0, 1, &left, &right);
	MPI_Cart_shift(grid_2D, 1, 1, &down, &upper);

	FILE * FO;
	char fname[20];
	sprintf(fname, "%d.dat", rank);
	FO = fopen(fname, "wt");
	fprintf(FO, "\nrank = %d left = %d right = %d\n ", rank, left, right);
	fprintf(FO, "\nrank = %d down = %d upper = %d\n ", rank, down, upper);

	double timeStart = MPI_Wtime();
	while (niter <= maxitt &&!(maxdif < EPS))
	{
		fprintf(FO, "\n\nstart iter %d\n", niter);

		// send left
		if (left != -1) {
			fprintf(FO, "send to %d\n", left);
			for (i = 0; i <= Im2; i++)
			{
				for (k = 0; k <= Km; k++)
				{
					M[i][k] = Fi[i][1][k];
				}
			}

			MPI_Send(M, (Im2 + 1)*(Km + 1), MPI_DOUBLE, left, 15, grid_2D);
		}


		if (down != -1) {
			fprintf(FO, "send to %d\n", down);
			MPI_Send(Fi[1], (Jm2 + 1)*(Km + 1), MPI_DOUBLE, down, 15, grid_2D);
		}
		// send up
		if (upper != -1)
		{
			fprintf(FO, "send to %d\n", upper);
			MPI_Send(Fi[Im2 - 1], (Jm2 + 1)*(Km + 1), MPI_DOUBLE, upper, 15, grid_2D);
		}
		// send right
		if (right != -1)
		{
			fprintf(FO, "send to %d\n", right);
			for (i = 0; i <= Im2; i++)
			{
				for (k = 0; k <= Km; k++)
				{
					M[i][k] = Fi[i][Jm2 - 1][k];
				}
			}

			MPI_Send(M, (Im2 + 1)*(Km + 1), MPI_DOUBLE, right, 15, grid_2D);
		}
		// recv from down
		if (down != -1) {
			fprintf(FO, "wait recv from %d\n", down);
			MPI_Recv(Fi[0], (Jm2 + 1)*(Km + 1), MPI_DOUBLE, down, 15, grid_2D, &st);
			fprintf(FO, "recvied from %d\n", down);
		}
		// recv from left
		if (left != -1) {
			fprintf(FO, "wait recv from %d\n", left);
			MPI_Recv(M, (Im2 + 1)*(Km + 1), MPI_DOUBLE, left, 15, grid_2D, &st);

			for (i = 0; i <= Im2; i++)
			{
				for (k = 0; k <= Km; k++)
				{
					Fi[i][0][k] = M[i][k];
				}
			}
			fprintf(FO, "recvied from %d\n", left);
		}

		maxdif = -1.0;


		if (right != -1)
		{
			// recv from right
			fprintf(FO, "wait recv from %d\n", right);
			MPI_Recv(M, (Im2 + 1)*(Km + 1), MPI_DOUBLE, right, 15, grid_2D, &st);

			for (i = 0; i <= Im2; i++)
			{
				for (k = 0; k <= Km; k++)
				{
					Fi[i][Jm2][k] = M[i][k];
				}
			}

			fprintf(FO, "recvied from %d\n", right);
		}

		if (upper != -1)
		{
			// recv from up
			fprintf(FO, "wait recv from %d\n", upper);
			MPI_Recv(Fi[Im2], (Jm2 + 1)*(Km + 1), MPI_DOUBLE, upper, 15, grid_2D, &st);
			fprintf(FO, "recvied from %d\n", upper);
		}

		// calc
		for (j = 1; j <= Jm2 - 1; j++)
		{
			for (k = 1; k <= Km - 1; k++) 
			{
				Fiold = Fi[i][j][k];
				Fi[i][j][k] = C*((Fi[i + 1][j][k] + Fi[i - 1][j][k]) / (hx*hx) +
					(Fi[i][j + 1][k] + Fi[i][j - 1][k]) / (hy*hy) +
					(Fi[i][j][k + 1] + Fi[i][j][k - 1]) / (hz*hz) - Ro[i][j][k]);
				maxdif = max(maxdif, fabs(Fi[i][j][k] - Fiold));
			}
		}

		fprintf(FO, "maxdif = %lf\n", maxdif);
		fflush(FO);
		niter++;
	}

	fprintf(FO, "\n\nfinished\n");
	fflush(FO);
	fclose(FO);

	if (rank == ROOT)
	{
		cerr << MPI_Wtime() - timeStart << "\r\n";
	}

	maxdif = -1.0;
	for (i = 1; i <= Im2 - 1; i++)
	{
		for (j = 1; j <= Jm2 - 1; j++)
		{
			for (k = 1; k < Km; k++)
			{
				i2 = (rank%P1)*(Im2 - 1) + i;
				j2 = (rank / P1)*(Jm2 - 1) + j;
				maxdif = max(maxdif, fFi(hx*i2, hy*j2, hz*k) - Fi[i][j][k]);
			}
		}
	}

	double alldif;
	int maxiter;
	MPI_Allreduce(&maxdif, &alldif, 1, MPI_DOUBLE, MPI_MAX, grid_2D);
	MPI_Allreduce(&niter, &maxiter, 1, MPI_INT, MPI_MAX, grid_2D);

	if (rank == 0) {
		FO = fopen("result.txt", "wt");
		fprintf(FO, "maxiter = %d\nmaxdif = %d\n", maxiter, niter);
		fclose(FO);
	}

	MPI_Finalize();
	return 0;
}
