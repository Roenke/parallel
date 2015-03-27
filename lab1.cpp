#include <mpi.h>
#include <math.h>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
using namespace std;

const double THETA = 0.1;
const int MAXITER = 10000;
const double EPS = 1e-7;
const int ROOT = 0;

// Чтение вектора правой части и матрицы из файлов
void input(string matrixFile, string vectorFile, double* &matrix, double* &vector, int &size)
{
	ifstream file(matrixFile.c_str());
	file >> size;
	matrix = new double[size * size];
	vector = new double[size];
	
	// Считаем матрицу как транспонированную
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			file >> matrix[j * size + i];
		}
	}

	file.close();
	file.open(vectorFile.c_str());

	int sizeVector;
	file >> sizeVector;

	if (size != sizeVector) throw;

	for (int i = 0; i < size; i++)
	{
		file >> vector[i];
	}

	file.close();
}

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

// Первый параметр выходной.
void partialMatrixVectorProd(double *localVector, double* matrixColumns, double* xElements, int localVectorSize, int taskSize)
{
	for (int j = 0; j < taskSize; j++)
	{
		localVector[j] = xElements[0] * matrixColumns[j];
	}

	for (int i = 1; i < localVectorSize; i++)
	{
		for (int j = 0; j < taskSize; j++)
		{
			localVector[j] += matrixColumns[i * taskSize + j] * xElements[i];
		}
	}
}

void partialVectorSub(double* localTemp, double* localVector, int localVectorSize)
{
	for (int i = 0; i < localVectorSize; i++)
	{
		localTemp[i] -= localVector[i];
	}
}

double getPartialError(double* localTemp, int localVectorSize)
{
	double result = 0;
	for (int i = 0; i < localVectorSize; i++)
		result += localTemp[i] * localTemp[i];
	return result;
}

void partialNextValue(double* xElements, double* localTemp, int localVectorSize)
{
	for (int i = 0; i < localVectorSize; i++)
		localTemp[i] = xElements[i] - THETA * localTemp[i];
}

int main(int argc, char** argv)
{
	// Используются во всех узлах
	int rank, size, taskSize;
	int *countsMatrix, *countsVector;
	int *displsMatrix, *displsVector;
	double *matrixColumns, *vectorElements;
	double *localVector, *xElements, *localTemp;
	
	// Используется только в корневом
	double* matrix;
	double* vector;
	double* result;
	double timeStart;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (rank == ROOT)
	{
		try
		{
			if (argc == 2 && sscanf(argv[1], "%d", &taskSize) == 1)
			{
				cerr << "root : generating matrix processed\r\n";
				generate(matrix, vector, taskSize);
				cerr << "root : generating matrix completed\r\n";
			}
			else
			{
				cerr << "root : input matrix processed\r\n";
				input("matrix.txt", "vector.txt", matrix, vector, taskSize);
				cerr << "root : input matrix completed\r\n";
			}
			
			if (taskSize < rank) throw;

			result = new double[taskSize];
			for (int i = 0; i < taskSize; i++)
			{
				result[i] = 0.0;
			}

			countsMatrix = new int[size];
			displsMatrix = new int[size];
			countsVector = new int[size];
			displsVector = new int[size];

			int rankSize;
			rankSize = taskSize / size + (taskSize % size != 0 ? 1 : 0);
			countsMatrix[0] = taskSize * rankSize;
			displsMatrix[0] = displsVector[0] = 0;
			countsVector[0] = rankSize;
			
			for (int i = 1; i < size; i++)
			{
				// Заполним массивы для MPI_scatterv
				rankSize = taskSize / size + (i < taskSize % size ? 1 : 0);
				countsMatrix[i] = taskSize * rankSize;
				countsVector[i] = rankSize;
				displsMatrix[i] = displsMatrix[i - 1] + countsMatrix[i];
				displsVector[i] = displsVector[i - 1] + rankSize;
			}
		}
		catch (exception)
		{
			cerr << "Input error. Check files matrix.txt and vector.txt";
			return 1;
		}
	}
	else
	{
		matrix = vector = result = NULL;
		countsMatrix = displsMatrix = countsVector = displsVector = NULL;
	}

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(&taskSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

	int localVectorSize = taskSize / size + (rank < taskSize % size ? 1 : 0);
	int localMatrixSize = taskSize * localVectorSize;

	vectorElements = new double[localVectorSize];
	matrixColumns = new double[localMatrixSize];
	localVector = new double[taskSize];
	xElements = new double[localVectorSize];
	localTemp = new double[localVectorSize];

	// передадим узлам их части матрицы и вектора правой части
	MPI_Scatterv(matrix, countsMatrix, displsMatrix, MPI_DOUBLE, matrixColumns, localMatrixSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatterv(vector, countsVector, displsVector, MPI_DOUBLE, vectorElements, localVectorSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	timeStart = MPI_Wtime();
	double error = 1.0;
	for (int iteration = 1; iteration <= MAXITER && error > EPS; iteration++)
	{
		if (rank == 0)
		{
			//cerr << "Start iteration " << iteration << ". Error = " << error << "\r\n";
		}
		// Найдем Ax(k)
		MPI_Scatterv(result, countsVector, displsVector, MPI_DOUBLE, xElements, localVectorSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		partialMatrixVectorProd(localVector, matrixColumns, xElements, localVectorSize, taskSize);
		// result <- Ax(k)
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Reduce(localVector, result, taskSize, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

		// ****************
		// Найдем Ax(k) - f
		MPI_Scatterv(result, countsVector, displsVector, MPI_DOUBLE, localTemp, localVectorSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		partialVectorSub(localTemp, vectorElements, localVectorSize);
		// result <- Ax(k) - f
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Gatherv(localTemp, localVectorSize, MPI_DOUBLE, result, countsVector, displsVector, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		// ****************

		// ****************
		// Посчитаем невязку
		error = 0;
		MPI_Scatterv(result, countsVector, displsVector, MPI_DOUBLE, localTemp, localVectorSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		double localResult = getPartialError(localTemp, localVectorSize);
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Allreduce(&localResult, &error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		// ****************
		error = sqrt(error);

		// ****************
		// найдем следующий X
		MPI_Scatterv(result, countsVector, displsVector, MPI_DOUBLE, localTemp, localVectorSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		partialNextValue(xElements, localTemp, localVectorSize);
		// result <- x(k+1)
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Gatherv(localTemp, localVectorSize, MPI_DOUBLE, result, countsVector, displsVector, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		// ****************
		
	}

	if (rank == 0)
	{
		error = 0;
		for (int i = 0; i < taskSize; i++)
		{
			error += (result[i] - 1.0) * (result[i] - 1.0);
		}
		error = sqrt(error / taskSize);
		std::cerr << "Total error = " << error << "\r\n";
		std::cerr << "Total time = " << MPI_Wtime() - timeStart << "\r\n";
	}

	MPI_Finalize();
	return 0;
}
