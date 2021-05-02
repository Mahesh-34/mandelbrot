#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

const int MASTER = 0, TAG = 1;
int STOP = -1;

unsigned int bailout(float complex z, unsigned int max_iters) {
 
  unsigned int i = 0;
  float complex zi = 0.0 + 0.0 * I;
  while (creal(zi)*creal(zi)+cimag(zi)*cimag(zi) < 4 && i++ < max_iters) {
    zi = zi*zi + z;
  }
  if (i != max_iters) {
    return i;
  } else {
    return max_iters;
  }
}

int main(int argc, char **argv) {
  
  int err, num_ranks, rank;
  MPI_Status status;
  err = MPI_Init(&argc, &argv);
  if (err != MPI_SUCCESS) {
    fprintf(stderr, "Couldn't start MPI.\n");
    MPI_Abort(MPI_COMM_WORLD, err);
  }
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
  if (num_ranks < 1) {
    fprintf(stderr, "Need at least one rank besides master.\n");
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  float x0 = -2.0, y0 = -1.0, xn = 1.0, yn = 1.0;
  unsigned int w = 800, h = 600, max_iters = 20;
  if (argc >= 3) w = (unsigned)atoi(argv[2]);
  if (argc >= 4) h = (unsigned)atoi(argv[3]);
  if (argc >= 5) max_iters = (unsigned)atoi(argv[4]);
  if (argc >= 6) x0 = atof(argv[5]);
  if (argc >= 7) y0 = atof(argv[6]);
  if (argc >= 8) xn = atof(argv[7]);
  if (argc >= 9) yn = atof(argv[8]);
  float scale_x = (xn - x0) / (float)w;
  float scale_y = (yn - y0) / (float)h;
  unsigned int **img = NULL; 
  if (rank == MASTER) {
    
    img = (unsigned int**)malloc(h*sizeof(unsigned int*));
    for (size_t i = 0; i < h; ++i) {
      img[i] = (unsigned int*)malloc(w*sizeof(unsigned int));
    }
    
    int i = 0; 
    while (i < h) {
      int k = 1;
      do {
        
        err = MPI_Send(&i, 1, MPI_INT, k, TAG, MPI_COMM_WORLD);
        if (err != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, err);
        
        err = MPI_Recv(img[i], w, MPI_UNSIGNED, k, TAG, MPI_COMM_WORLD,
            &status);
        if (err != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, err);
      } while (++i < h && ++k < num_ranks);
    }
    
    for (int k = 1; k < num_ranks; ++k) {
      MPI_Send(&STOP, 1, MPI_INT, k, TAG, MPI_COMM_WORLD);
    }
  } else {
    
    int i = !STOP;
    while (i != STOP) {
      
      unsigned int *row = (unsigned int*)malloc(w*sizeof(unsigned int));
     
      float im = y0 + i * scale_y;
      for (unsigned int j = 0; j < w; ++j) {
        float re = x0 + j * scale_x;
        float complex z = re + im * I;
        row[j] = bailout(z, max_iters);
      }
      
      free(row);
      if (err != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, err);
    }
  }
  
  if (rank == MASTER) {
    char const *fname = NULL;
    if (argc >= 2) fname = argv[1];
    FILE *f = fopen(fname, "w");
    for (size_t i = 0; i < h; ++i) {
      if (f) {
        for (size_t j = 0; j < w; ++j) {
          fprintf(f, "%zu,", img[i][j]);
        }
        fprintf(f, "\n");
      }
      free(img[i]);
    }
    free(img);
    if (f) {
      fclose(f);
    }
  }
  MPI_Finalize();
  return EXIT_SUCCESS;
}
