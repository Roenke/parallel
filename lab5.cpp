#include <mpi.h>
#include <pthread.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

// Будем хранить задачи в списке LIFO
class TaskRepository
{
private:
	int *_tasks;
	int _last_task_index;
	unsigned _size;
public:
	TaskRepository(unsigned size);
	void FillRandomTasks();
	bool IsEmpty();
	bool IsFull();
	bool AddTask(int task);
	bool AddRange(int *tasks, int count);
	int* GetHalfOfTasks();
	int Count();
	int GetAndRemoveTask();
};
using namespace std;

int node_rank;
int cluster_size;

const int EXIT_MESSAGE = -3;
const int TASKS_COUNT = 1000;
const int MIN_TASK_COST = 10000;
const int MAX_TASK_COST = 100000;

const int NO_TASK = -2;
const int LAST_TASK = -1;

fstream worker_file, sender_file, receiver_file;
pthread_t worker_thread, receiver_thread, sender_thread;
pthread_attr_t attrs;
pthread_mutex_t mutex;
pthread_cond_t condition;
TaskRepository task_repository(TASKS_COUNT);

TaskRepository::TaskRepository(unsigned size)
{
	_tasks = new int[size];
	_last_task_index = -1;
	_size = size;
}

void TaskRepository::FillRandomTasks()
{
	int random_max = MAX_TASK_COST - MIN_TASK_COST;
	for (int i = 0; i < _size; i++)
	{
		_tasks[i] = rand() % random_max + MIN_TASK_COST;
	}
	_last_task_index = _size - 1;
}

bool TaskRepository::IsEmpty()
{
	return _last_task_index == -1;
}

bool TaskRepository::IsFull()
{
	return _last_task_index == _size - 1;
}

bool TaskRepository::AddTask(int task)
{
	if (!IsFull())
	{
		_tasks[++_last_task_index] = task;
		return true;
	}
	return false;
}

bool TaskRepository::AddRange(int* tasks, int count)
{
	if (_size - _last_task_index - 1 > count)
	{
		for (int i = 0; i < count; i++)
		{
			cerr << "added task: " << tasks[i] << "\r\n";
			_tasks[++_last_task_index] = tasks[i];
		}
		return true;
	}
	return false;
}

int* TaskRepository::GetHalfOfTasks()
{
	int count = Count();
	if (count < 2)
	{
		return NULL;
	}

	int send_count = count / 2;
	int* result = new int[send_count + 1];

	result[0] = send_count;
	for (int i = 1; i <= send_count; i++)
	{
		result[i] = GetAndRemoveTask();
	}

	return result;
}

int TaskRepository::Count()
{
	return _last_task_index + 1;
}

int TaskRepository::GetAndRemoveTask()
{
	if (_last_task_index < 0) return 0;
	return _tasks[_last_task_index--];
}

void *worker(void*)
{
	pthread_mutex_lock(&mutex);
	int current_task = task_repository.GetAndRemoveTask();
	pthread_mutex_unlock(&mutex);
	while (current_task != LAST_TASK)
	{
		worker_file << "Worker[" << node_rank << "]: Start task " << current_task << "\r\n";
		double task_result = 0;
		for (int i = current_task; i > 0; i--)
		{
			task_result += 1 / sqrt(i);
		}

		worker_file << "Worker[" << node_rank << "]: Task complete. result =  " << task_result << "\r\n";
		pthread_mutex_lock(&mutex);
		if (task_repository.IsEmpty())
		{
			// Дождаться, пока появятся задачи
			worker_file << "Worker[" << node_rank << "]: Tasks is over. Wait for new tasks. \r\n";
			pthread_cond_signal(&condition);
			pthread_cond_wait(&condition, &mutex);
			worker_file << "Worker[" << node_rank << "]: Answer has received. \r\n";
		}
		
		current_task = task_repository.GetAndRemoveTask();
		pthread_mutex_unlock(&mutex);
	}

	worker_file << "Worker[" << node_rank << "]: Last task compeleted. Exit from thread. \r\n";
	worker_file << "Worker[" << node_rank << "]: Close sender thread. \r\n";

	return NULL;
}

void* task_receiver(void*)
{
	MPI_Status st;
	int buffer_size = TASKS_COUNT;
	int* buffer = new int[buffer_size];
	buffer[0] = NO_TASK;
	while (true)
	{
		if (!task_repository.IsEmpty())
		{
			receiver_file << "Receiver[" << node_rank << "]. Waiting for request.\r\n";
			pthread_cond_wait(&condition, &mutex);
			receiver_file << "Receiver[" << node_rank << "]. Receive request for new tasks!.\r\n";
		}

		if (task_repository.IsEmpty())
		{
			pthread_mutex_unlock(&mutex);
			for (int i = 0; i < cluster_size; i++)
			{
				if (i == node_rank) continue;
				receiver_file << "Receiver[" << node_rank << "]. Sent reqest to " << i << " node.\r\n";
				MPI_Send(&node_rank, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
				MPI_Recv(buffer, buffer_size, MPI_INT, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, &st);
				if (buffer[0] != NO_TASK)
				{
					receiver_file << "Receiver[" << node_rank << "]. Tasks successfully received. (" << buffer[0] << " tasks)\r\n";
					pthread_mutex_lock(&mutex);
					task_repository.AddRange(buffer + 1, buffer[0]);
					pthread_cond_signal(&condition);
					break;
				}
			}
			if (buffer[0] == NO_TASK)
			{
				receiver_file << "Receiver[" << node_rank << "]. No available tasks :( \r\n";
				pthread_mutex_lock(&mutex);
				task_repository.AddTask(LAST_TASK);
				receiver_file << "Receiver[" << node_rank << "]. Last task added.\r\n";
				pthread_cond_signal(&condition);
				break;
			}
		}
		pthread_mutex_unlock(&mutex);
	}

	pthread_mutex_unlock(&mutex);
	return NULL;
}

void* task_sender(void*)
{
	MPI_Status st;
	int node_number;
	int no_task = NO_TASK;
	while (true)
	{
		MPI_Recv(&node_number, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &st);
		if (node_number == EXIT_MESSAGE) break;
		sender_file << "Sender[" << node_rank << "]. Receive request for tasks from " << node_number << " node.\r\n";
		pthread_mutex_lock(&mutex);
		if (task_repository.Count() >= 2)
		{
			// Перешлем половину задач на другой узел
			sender_file << "Sender[" << node_rank << "]. I have " << task_repository.Count() << " tasks. Sent half of task on this node\r\n";
			int* tasks_to_send = task_repository.GetHalfOfTasks();
			pthread_mutex_unlock(&mutex);
			sender_file << "Sender[" << node_rank << "]. Send " << tasks_to_send[0] << " tasks.\r\n";
			MPI_Send(tasks_to_send, tasks_to_send[0] + 1, MPI_INT, node_number, 2, MPI_COMM_WORLD);
		}
		else
		{
			sender_file << "Sender[" << node_rank << "]. I cannot sent any task. Number of tasks: " << task_repository.Count() << ".\r\n";
			MPI_Send(&no_task, 1, MPI_INT, node_number, 2, MPI_COMM_WORLD);
		} 
		pthread_mutex_unlock(&mutex);
	}
	sender_file << "Sender[" << node_rank << "]. Exit from sender.\r\n";
	return NULL;
}

int main(int argc, char **argv)
{
	char log_file_name[50];
	int ret_code;
	int exit_code = EXIT_MESSAGE;
	int provide;													// Ранг процесса и общее количество процессов соответственно
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provide);	// Инициализация параллельной части приложения
	MPI_Comm_rank(MPI_COMM_WORLD, &node_rank);						// Определение номера процесса в группе
	MPI_Comm_size(MPI_COMM_WORLD, &cluster_size);					// Определение общего числа параллельных процессов

	
	task_repository.FillRandomTasks();

	sprintf(log_file_name, "node-%d-worker.log", node_rank);
	worker_file.open(log_file_name, ios::out);
	worker_file.setf(ios::unitbuf);

	sprintf(log_file_name, "node-%d-sender.log", node_rank);
	sender_file.open(log_file_name, ios::out);
	sender_file.setf(ios::unitbuf);

	sprintf(log_file_name, "node-%d-receiver.log", node_rank);
	receiver_file.open(log_file_name, ios::out);
	receiver_file.setf(ios::unitbuf);

	pthread_cond_init(&condition, NULL);

	if (0 != pthread_mutex_init(&mutex, NULL))
	{
		cerr << "Init mutex failed";
		abort();
	}

	ret_code = pthread_create(&worker_thread, NULL, worker, NULL);
	ret_code += pthread_create(&receiver_thread, NULL, task_receiver, NULL);
	ret_code += pthread_create(&sender_thread, NULL, task_sender, NULL);

	if (ret_code != 0)
	{
		cerr << "Threads creating failed";
		abort();
	}

	ret_code = pthread_join(worker_thread, NULL);
	ret_code += pthread_join(receiver_thread, NULL);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Send(&exit_code, 1, MPI_INT, node_rank, 1, MPI_COMM_WORLD);
	ret_code += pthread_join(sender_thread, NULL);

	if (ret_code != 0)
	{
		cerr << "Threads joining failed";
		abort();
	}
	MPI_Barrier(MPI_COMM_WORLD);

	pthread_attr_destroy(&attrs);
	pthread_mutex_destroy(&mutex);
	MPI_Finalize();
	
	return 0;
}