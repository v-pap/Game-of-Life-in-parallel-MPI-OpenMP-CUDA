// To compile: make game
// To run: ./a.out [width] [height] [input_file]

#define _DEFAULT_SOURCE

#define GEN_LIMIT 1000

#define CHECK_SIMILARITY
#define SIMILARITY_FREQUENCY 3

#define true 1
#define false 0

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

void perror_exit(const char *message)
{
    perror(message);
    exit(EXIT_FAILURE);
}

void print_to_file(unsigned char **univ, int width, int height)
{
    FILE *fout = fopen("./game_output.out", "w"); // printing the result to a file with
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

void show(unsigned char **univ, int width, int height)
{
    // Prints the result in stdout, using various VT100 escape codes
    printf("\033[H");

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            printf((univ[y][x] == '1') ? "\033[07m  \033[m" : "  ");
        }

        printf("\033[E");
    }

    fflush(stdout);
}

void evolve(unsigned char **univ, unsigned char **new_univ, int width, int height)
{
    // Generate new generation: keep it in new_univ
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int neighbors = 0;

            for (int y1 = y - 1; y1 <= y + 1; y1++)
            {
                for (int x1 = x - 1; x1 <= x + 1; x1++)
                {
                    int x2 = x1, y2 = y1;
                    if (x1 == -1)
                        x2 = width - 1;
                    if (y1 == -1)
                        y2 = height - 1;
                    if (x1 == width)
                        x2 = 0;
                    if (y1 == height)
                        y2 = 0;

                    if (univ[y2][x2] == '1')
                        neighbors++;
                }
            }

            if (univ[y][x] == '1')
                neighbors--;

            if (neighbors == 3 || (neighbors == 2 && (univ[y][x] == '1')))
            {
                new_univ[y][x] = '1';
            }
            else
            {
                new_univ[y][x] = '0';
            }
        }
    }
}

int empty(unsigned char **univ, int width, int height)
{
    // Checks if local is empty or not (a.k a. all the cells are dead)
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            if (univ[y][x] == '1')
                return false;
        }
    }

    return true;
}

int similarity(unsigned char **univ, unsigned char **new_univ, int width, int height)
{
    // Check if the new generation is the same with the previous generation
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            if (univ[y][x] != new_univ[y][x])
                return false;
        }
    }

    return true;
}

void game(int width, int height, char *fileArg)
{
    // Allocate space for the two game arrays (one for current generation, the other for the new one)
    unsigned char **univ = malloc(width * sizeof(unsigned char *)),
                  **new_univ = malloc(width * sizeof(unsigned char *));
    if (univ == NULL || new_univ == NULL)
        perror_exit("malloc: ");

    for (int i = 0; i < width; i++)
    {
        univ[i] = malloc(height * sizeof(unsigned char));
        new_univ[i] = malloc(height * sizeof(unsigned char));
        if (univ[i] == NULL || new_univ[i] == NULL)
            perror_exit("malloc: ");
    }

    FILE *filePtr = fopen(fileArg, "r");
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
    fclose(filePtr);
    filePtr = NULL;

    int generation = 1;
#ifdef CHECK_SIMILARITY
    int counter = 0;
#endif

    // Get currect timestamp: calculations are about to start
    clock_t t_start = clock();

    while ((!empty(univ, width, height)) && (generation <= GEN_LIMIT))
    {
        evolve(univ, new_univ, width, height);

#ifdef CHECK_SIMILARITY
        counter++;
        if (counter == SIMILARITY_FREQUENCY)
        {
            if (similarity(univ, new_univ, width, height))
                break;
            counter = 0;
        }
#endif

        unsigned char **temp_univ = univ;
        univ = new_univ;
        new_univ = temp_univ;

        generation++;
    }

    // Get the total duration of the loop above in milliseconds
    double msecs = ((float)clock() - t_start) / CLOCKS_PER_SEC * 1000.0f;

    printf("Finished.\n\n");
    printf("Generations:\t%d\n", generation - 1);
    printf("Execution time:\t%.2f msecs\n", msecs);

    // show (univ, width, height);
    print_to_file(univ, width, height);

    // Free allocated memory
    for (int i = 0; i < width; i++)
    {
        free(univ[i]);
        free(new_univ[i]);

        univ[i] = NULL;
        new_univ[i] = NULL;
    }
    free(univ);
    free(new_univ);

    univ = NULL;
    new_univ = NULL;
}

int main(int argc, char *argv[])
{
    int width = 0, height = 0;

    if (argc > 1)
        width = atoi(argv[1]);
    if (argc > 2)
        height = atoi(argv[2]);

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
