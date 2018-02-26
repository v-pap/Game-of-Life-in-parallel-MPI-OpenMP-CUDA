# Type 'make' or 'make game' to compile the serial version.
# Type 'make mpi' to compile the MPI version.
# Type 'make collective' to compile the MPI (with Collective I/O) version.
# Type 'make async' to compile the MPI (with Async I/O) version.
# Type 'make openmp' to compile the MPI + OpenMP version.
# Type 'make cuda' to compile the Cuda version.
CC = gcc
MPICC = mpicc
NVCC = nvcc
FLAGS = -std=c99 -Wall -O3

game:
	$(CC) $(FLAGS) src/game.c

mpi:
	$(MPICC) $(FLAGS) src/game_mpi.c -lm

collective:
	$(MPICC) $(FLAGS) src/game_mpi_collective.c -lm

async:
	$(MPICC) $(FLAGS) src/game_mpi_async.c -lm

openmp:
	$(MPICC) $(FLAGS) src/game_openmp.c -lm -fopenmp

cuda:
	$(NVCC) src/game_cuda.cu

clean:
	rm -f *.out
