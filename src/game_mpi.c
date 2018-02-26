// To compile: make mpi
// To run: mpiexec -n [x] -f machines ./a.out [width] [height] [input_file]

#define _DEFAULT_SOURCE

#define GEN_LIMIT 1000

#define CHECK_SIMILARITY
#define SIMILARITY_FREQUENCY 3

#define true 1
#define false 0

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

void perror_exit(const char *message)
{
    perror(message);
    exit(EXIT_FAILURE);
}

void print_to_file(char **univ, int width, int height)
{
    FILE *fout = fopen("./mpi_output.out", "w"); // printing the result to a file with
                                                 // 1 or 0 (1 being an alive cell and 0 a dead cell)
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            fprintf(fout, "%c", univ[i][j]);
        }
        fprintf(fout, "\n");
    }

    fflush(fout);
    fclose(fout);
}

void show(char **univ, int width, int height)
{
    // Prints the result in stdout, using various VT100 escape codes
    printf("\033[H");

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            printf(univ[y][x] ? "\033[07m  \033[m" : "  ");
        }
        printf("\033[E");
    }

    fflush(stdout);
}

void evolve(char **local, char **new, int width_local, int height_local)
{
    // Access the cells of the actual grid, leaving out the auxiliary cells
    // around the grid, yet, taking them into account for the calculations
    for (int y = 1; y <= height_local; y++)
    {
        for (int x = 1; x <= width_local; x++)
        {
            int neighbors = 0;

            // Add the value of each cell to neighbor's variable
            // Adds the ASCII value of each cell, '1' = 49, '0' = 48
            neighbors = local[y - 1][x - 1] + local[y - 1][x] +
                        local[y - 1][x + 1] + local[y][x - 1] +
                        local[y][x + 1] + local[y + 1][x - 1] +
                        local[y + 1][x] + local[y + 1][x + 1];

            // Determine if the current cell is going to be alive or not
            // 387 means that it has 3 neighbors ( (3 * 49) + (5 * 48))
            // 386 means that it has 2 neighbors ( (2 * 49) + (6 * 48))
            if (neighbors == 387 || (neighbors == 386 && (local[y][x] == '1')))
                new[y][x] = '1';
            else
                new[y][x] = '0';
        }
    }
}

int empty(char **local, int width_local, int height_local)
{
    // Checks if local is empty or not (a.k.a. all the cells are dead)
    for (int y = 1; y <= height_local; y++)
    {
        for (int x = 1; x <= width_local; x++)
        {
            if (local[y][x])
                return false;
        }
    }

    return true;
}

int empty_all(char **local, int width_local, int height_local, MPI_Comm *new_comm, int comm_sz)
{
    // Calculates if all subgrids are empty
    int local_flag = empty(local, width_local, height_local),
        global_sum;

    MPI_Allreduce(&local_flag, &global_sum, 1, MPI_INT, MPI_SUM, *new_comm);

    // Compare the number of instances that have an empty grid
    // with the total amount of instances
    return (global_sum == comm_sz);
}

int similarity(char **local, char **local_old, int width_local, int height_local)
{
    // Check if the internal grid is the same with the previous generation
    for (int y = 1; y <= height_local; y++)
    {
        for (int x = 1; x <= width_local; x++)
        {
            if (local_old[y][x] != local[y][x])
                return false;
        }
    }

    return true;
}

int similarity_all(char **local, char **local_old, int width_local, int height_local, MPI_Comm *new_comm, int comm_sz)
{
    // Calculates if every subgrid is the same as in the previous generation
    int local_flag = similarity(local, local_old, width_local, height_local),
        global_sum;

    MPI_Allreduce(&local_flag, &global_sum, 1, MPI_INT, MPI_SUM, *new_comm);

    // Compare the number of instances that have the same grid
    // between generations, with the total number of instances
    return (global_sum == comm_sz);
}

void game(int width, int height, char *fileArg)
{
    // Allocate space for the universal array, the main grid
    char **univ = malloc(width * sizeof(char *));
    char *a = malloc(width * height * sizeof(char));
    if (univ == NULL || a == NULL)
        perror_exit("malloc: ");
    for (int i = 0; i < width; i++)
        univ[i] = &a[i * height];

    int my_rank, comm_sz;

    // Initialize the MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    MPI_Comm old_comm, new_comm;
    int ndims, reorder, periods[2], dim_size[2];

    old_comm = MPI_COMM_WORLD;
    ndims = 2; // 2D matrix/grid
    int rows_columns = (int)sqrt(comm_sz);

    int width_local, height_local;

    // Calculate the local dimensions for the local subarrays
    width_local = height_local = width / rows_columns;

    dim_size[0] = rows_columns; // number of rows
    dim_size[1] = rows_columns; // number of columns
    periods[0] = 1;             // rows periodic (each column forms a ring)
    periods[1] = 1;             // columns periodic (each row forms a ring)
    reorder = 1;                // allows processes reordered for efficiency

    // Create a fully periodic, 2D Cartesian topology
    MPI_Cart_create(old_comm, ndims, dim_size, periods, reorder, &new_comm);
    int me, coords[2];

    MPI_Comm_rank(new_comm, &me);
    MPI_Cart_coords(new_comm, me, ndims, coords);

    double t_start = MPI_Wtime();
    double msecs;
    FILE *filePtr = NULL;

    // Allocate space in each instance for the local array(s)
    char **local = malloc((width_local + 2) * sizeof(char *));
    char *b = malloc((width_local + 2) * (height_local + 2) * sizeof(char));
    if (local == NULL || b == NULL)
        perror_exit("malloc: ");
    for (int i = 0; i < (width_local + 2); i++)
        local[i] = &b[i * (height_local + 2)];

    MPI_Status status;

    if (me == 0) // If I am the master instance
    {
        filePtr = fopen(fileArg, "r");
        if (filePtr == NULL)
            perror_exit("fopen: ");

        // Populate univ with its contents
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width;)
            {
                char c = fgetc(filePtr);
                if ((c != EOF) && (c != '\n'))
                {
                    univ[y][x] = c;
                    x++;
                }
            }
        }

        // For every other instance, receive the local array
        if (comm_sz > 1)
        {
            for (int i = 1; i < comm_sz; i++)
            {
                MPI_Cart_coords(new_comm, i, ndims, coords);

                // And copy the data to univ
                for (int i = 0; i < width_local; i++)
                {
                    for (int j = 0; j < height_local; j++)
                    {
                        local[i + 1][j + 1] = univ[i + coords[0] * height_local][j + coords[1] * width_local];
                    }
                }

                MPI_Send(&local[0][0], (width_local + 2) * (height_local + 2), MPI_CHAR, i, 0, new_comm);
            }
        }

        // Copy the data from univ to my array
        for (int i = 0; i < width_local; i++)
        {
            for (int j = 0; j < height_local; j++)
            {
                local[i + 1][j + 1] = univ[i][j];
            }
        }
    }
    else // If I am not the master instance
    {
        // Receive the local array from the master instance
        MPI_Recv(&local[0][0], (width_local + 2) * (height_local + 2), MPI_CHAR, 0, MPI_ANY_TAG, new_comm, &status);
    }

    if (me == 0)
    {
        fclose(filePtr);
        filePtr = NULL;
    }

    msecs = (MPI_Wtime() - t_start) * 1000;

    if (me == 0)
        printf("Reading file:\t%.2lf msecs\n", msecs);

    // Allocate space for the new array which holds the next generation of the local grid
    char **new = malloc((width_local + 2) * sizeof(char *));
    char *c = malloc((width_local + 2) * (height_local + 2) * sizeof(char));
    if (new == NULL || c == NULL)
        perror_exit("malloc: ");
    for (int i = 0; i < (width_local + 2); i++)
        new[i] = &c[i * (height_local + 2)];

    int generation = 1;
#ifdef CHECK_SIMILARITY
    int counter = 0;
#endif

    MPI_Cart_coords(new_comm, me, ndims, coords);

    // Calculating the coordinates of the neighbours, relative to the current process ones
    int north;
    int south;
    int east;
    int west;

    int north_coords[2];
    int south_coords[2];
    int west_coords[2];
    int east_coords[2];

    north_coords[0] = coords[0] + 1;
    south_coords[0] = coords[0] - 1;
    west_coords[0] = coords[0];
    east_coords[0] = coords[0];

    north_coords[1] = coords[1];
    south_coords[1] = coords[1];
    west_coords[1] = coords[1] - 1;
    east_coords[1] = coords[1] + 1;

    int north_west;
    int north_east;
    int south_west;
    int south_east;

    int north_west_coords[2];
    int north_east_coords[2];
    int south_west_coords[2];
    int south_east_coords[2];

    north_west_coords[0] = coords[0] - 1;
    north_east_coords[0] = coords[0] - 1;
    south_west_coords[0] = coords[0] + 1;
    south_east_coords[0] = coords[0] + 1;

    north_west_coords[1] = coords[1] - 1;
    north_east_coords[1] = coords[1] + 1;
    south_west_coords[1] = coords[1] - 1;
    south_east_coords[1] = coords[1] + 1;

    // Get the rank of each direction
    MPI_Cart_rank(new_comm, north_coords, &north);
    MPI_Cart_rank(new_comm, south_coords, &south);
    MPI_Cart_rank(new_comm, west_coords, &west);
    MPI_Cart_rank(new_comm, east_coords, &east);

    MPI_Cart_rank(new_comm, north_west_coords, &north_west);
    MPI_Cart_rank(new_comm, north_east_coords, &north_east);
    MPI_Cart_rank(new_comm, south_west_coords, &south_west);
    MPI_Cart_rank(new_comm, south_east_coords, &south_east);

    // Vector datatype representing columns in an 2D array
    MPI_Datatype vertical_type;

    MPI_Type_vector(height_local, 1, width_local + 2, MPI_CHAR, &vertical_type);
    MPI_Type_commit(&vertical_type);

    MPI_Request requests_odd[16];
    MPI_Request requests_even[16];

    // Communication requests to exchange data from local array
    MPI_Recv_init(&local[0][1], width_local, MPI_CHAR, north, 1, new_comm, &requests_odd[0]);
    MPI_Send_init(&local[1][1], width_local, MPI_CHAR, north, 2, new_comm, &requests_odd[1]);
    MPI_Recv_init(&local[height_local + 1][1], width_local, MPI_CHAR, south, 2, new_comm, &requests_odd[2]);
    MPI_Send_init(&local[height_local][1], width_local, MPI_CHAR, south, 1, new_comm, &requests_odd[3]);

    MPI_Recv_init(&local[1][width_local + 1], 1, vertical_type, east, 3, new_comm, &requests_odd[4]);
    MPI_Send_init(&local[1][width_local], 1, vertical_type, east, 4, new_comm, &requests_odd[5]);
    MPI_Recv_init(&local[1][0], 1, vertical_type, west, 4, new_comm, &requests_odd[6]);
    MPI_Send_init(&local[1][1], 1, vertical_type, west, 3, new_comm, &requests_odd[7]);

    MPI_Recv_init(&local[0][0], 1, MPI_CHAR, north_west, 5, new_comm, &requests_odd[8]);
    MPI_Send_init(&local[1][1], 1, MPI_CHAR, north_west, 6, new_comm, &requests_odd[9]);
    MPI_Recv_init(&local[0][width_local + 1], 1, MPI_CHAR, north_east, 7, new_comm, &requests_odd[10]);
    MPI_Send_init(&local[1][width_local], 1, MPI_CHAR, north_east, 8, new_comm, &requests_odd[11]);

    MPI_Recv_init(&local[height_local + 1][0], 1, MPI_CHAR, south_west, 8, new_comm, &requests_odd[12]);
    MPI_Send_init(&local[height_local][1], 1, MPI_CHAR, south_west, 7, new_comm, &requests_odd[13]);
    MPI_Recv_init(&local[height_local + 1][width_local + 1], 1, MPI_CHAR, south_east, 6, new_comm, &requests_odd[14]);
    MPI_Send_init(&local[height_local][width_local], 1, MPI_CHAR, south_east, 5, new_comm, &requests_odd[15]);

    // Communication requests to exchange data from new array
    MPI_Recv_init(&new[0][1], width_local, MPI_CHAR, north, 1, new_comm, &requests_even[0]);
    MPI_Send_init(&new[1][1], width_local, MPI_CHAR, north, 2, new_comm, &requests_even[1]);
    MPI_Recv_init(&new[height_local + 1][1], width_local, MPI_CHAR, south, 2, new_comm, &requests_even[2]);
    MPI_Send_init(&new[height_local][1], width_local, MPI_CHAR, south, 1, new_comm, &requests_even[3]);

    MPI_Recv_init(&new[1][width_local + 1], 1, vertical_type, east, 3, new_comm, &requests_even[4]);
    MPI_Send_init(&new[1][width_local], 1, vertical_type, east, 4, new_comm, &requests_even[5]);
    MPI_Recv_init(&new[1][0], 1, vertical_type, west, 4, new_comm, &requests_even[6]);
    MPI_Send_init(&new[1][1], 1, vertical_type, west, 3, new_comm, &requests_even[7]);

    MPI_Recv_init(&new[0][0], 1, MPI_CHAR, north_west, 5, new_comm, &requests_even[8]);
    MPI_Send_init(&new[1][1], 1, MPI_CHAR, north_west, 6, new_comm, &requests_even[9]);
    MPI_Recv_init(&new[0][width_local + 1], 1, MPI_CHAR, north_east, 7, new_comm, &requests_even[10]);
    MPI_Send_init(&new[1][width_local], 1, MPI_CHAR, north_east, 8, new_comm, &requests_even[11]);

    MPI_Recv_init(&new[height_local + 1][0], 1, MPI_CHAR, south_west, 8, new_comm, &requests_even[12]);
    MPI_Send_init(&new[height_local][1], 1, MPI_CHAR, south_west, 7, new_comm, &requests_even[13]);
    MPI_Recv_init(&new[height_local + 1][width_local + 1], 1, MPI_CHAR, south_east, 6, new_comm, &requests_even[14]);
    MPI_Send_init(&new[height_local][width_local], 1, MPI_CHAR, south_east, 5, new_comm, &requests_even[15]);

    t_start = MPI_Wtime();

    // The actual loop of Game of Life
    while ((!empty_all(local, width_local, height_local, &new_comm, comm_sz)) && (generation <= GEN_LIMIT))
    {

        // Different requests for odd and even generations in order to compensate the pointer swap of local and new arrays
        if ((generation % 2) == 1)
        {
            MPI_Startall(16, requests_odd);
            MPI_Waitall(16, requests_odd, MPI_STATUSES_IGNORE);
        }
        else
        {
            MPI_Startall(16, requests_even);
            MPI_Waitall(16, requests_even, MPI_STATUSES_IGNORE);
        }

        evolve(local, new, width_local, height_local);

        // The pointer swap
        char **temp_array = local;
        local = new;
        new = temp_array;

#ifdef CHECK_SIMILARITY
        counter++;
        if (counter == SIMILARITY_FREQUENCY)
        {
            if (similarity_all(local, new, width_local, height_local, &new_comm, comm_sz))
                break;
            counter = 0;
        }
#endif

        generation++;

    } // End of while loop

    msecs = (MPI_Wtime() - t_start) * 1000;

    if (me == 0) // If I am not the master instance
        printf("Generations:\t%d\nExecution time:\t%.2lf msecs\n", generation - 1, msecs);

    if (me == 0) // If I am the master instance
    {
        t_start = MPI_Wtime();
        // Copy the dat from my array to univ
        for (int i = 0; i < width_local; i++)
        {
            for (int j = 0; j < height_local; j++)
            {
                univ[i][j] = local[i + 1][j + 1];
            }
        }

        // For every other instance, receive the local array
        if (comm_sz > 1)
        {
            for (int i = 1; i < comm_sz; i++)
            {
                MPI_Cart_coords(new_comm, i, ndims, coords);
                MPI_Recv(&local[0][0], (width_local + 2) * (height_local + 2), MPI_CHAR, i, 0, new_comm, &status);

                // And copy the data to univ
                for (int i = 0; i < width_local; i++)
                {
                    for (int j = 0; j < height_local; j++)
                    {
                        univ[i + coords[0] * height_local][j + coords[1] * width_local] = local[i + 1][j + 1];
                    }
                }
            }
        }

        // show(univ, width, height);

        print_to_file(univ, width, height);

        msecs = (MPI_Wtime() - t_start) * 1000;

        printf("Writing file:\t%.2lf msecs\n", msecs);
    }
    else // If I am not the master instance
    {
        // Send the local array to the master instance
        MPI_Send(&local[0][0], (width_local + 2) * (height_local + 2), MPI_CHAR, 0, 0, new_comm);
    }

    // Deallocate space no longoer needed
    free(b);
    free(local);
    b = NULL;
    local = NULL;

    free(c);
    free(new);
    c = NULL;
    new = NULL;

    free(a);
    free(univ);
    a = NULL;
    univ = NULL;

    MPI_Type_free(&vertical_type);

    MPI_Finalize();
}

int main(int argc, char *argv[])
{
    int width = 0, height = 0;

    if (argc > 1)
        width = atoi(argv[1]);
    if (argc > 2)
        height = atoi(argv[2]);

    height = width;

    if (width <= 0)
        width = 30;
    if (height <= 0)
        height = 30;

    if (argc > 3)
        game(width, height, argv[3]);

    printf("Finished\n");
    fflush(stdout);

    return 0;
}
