
# Game of Life in parallel

This project is focused on the implementation of [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) with parallel processing. The following techonologies were used: MPI, OpenMp and CUDA. In total 6 programs were made in order to create meaningful benchmarks:
* simple serial C program
* MPI with serial I/O
* MPI with async I/O
* MPI with collective I/O
* MPI with collective I/O + OpenMP
* CUDA

## Getting Started

The following steps are required in order to set up your enviroment and run the above programs.

### Prerequisites


At first you will need to ensure that you have a compiler compatible with OpenMP. Use the following command to check the version of `gcc` that you are using (version 4.4 or newer is required):
```
gcc --version
```
Then you will need to install MPI and CUDA. 
```
sudo apt-get install mpich
sudo apt-get install nvidia-cuda-toolkit
```

### Installing

1) Get a copy of the project
```
git clone https://github.com/v-pap/Game-of-Life-in-parallel.git
```
2) Go into the directory
```
cd Game of Life
```
3) Choose which version you would like to build
```
make <version_name>
```

Where `<version_name>` can be `game`, `cuda`, `mpi`, `async`, `collective` or `openmp`.

## Using the program(s)

In order to run the program(s) you will have to use one one of the following commands:

* For the simple C program and the CUDA version
```
./a.out <width> <height> <input_file>
```
* For all the MPI or MPI + OpenMP versions
```
mpiexec -n <x> ./a.out <width> <height> <input_file>
```

Notes about the arguments:

The `<input_file>` must be a text file containing `<height>` rows and `<width>` columns and the acceptable values are `0` or `1` (in the current version of the project width should be equal to height). The script `generate.sh` generates valid inputs with randomized values.

Once the executable of any version of the Game of Life has finished running, an output file will be generated which contains the array of the last generation that was calculated.

Each source file contains two constants, `GEN_LIMIT` and `SIMILARITY_FREQUENCY` (which are defined on the preprocessor section). The default values are 1000 and 3 respectively and their values can be changed (this requires recompilation of the source file). The `GEN_LIMIT` constant describes the maximum amount of generations to be calculated. The `SIMILARITY_FREQUENCY` constant describes how often two consecutive generations will be compared for similarity. If the two generations are found similar then the program will exit. The removal of the similarity check can be achieved by removing the `#define CHECK_SIMILARITY` macro and then recompiling the source file.

## Authors

* [v-pap](https://github.com/v-pap)

* [vsakkas](https://github.com/vsakkas)

* [mahcloudservers](https://github.com/mahcloudservers)

## License

This project is licensed under the MIT License

