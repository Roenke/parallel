mpiexec -n $1 ./lab1 50 2>> out.txt;
mpiexec -n $1 ./lab1 100 2>> out.txt;
mpiexec -n $1 ./lab1 500 2>> out.txt;
mpiexec -n $1 ./lab1 1000 2>> out.txt;
mpiexec -n $1 ./lab1 5000 2>> out.txt;
mpiexec -n $1 ./lab1 10000 2>> out.txt