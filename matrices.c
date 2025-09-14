#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Función para reservar memoria dinámica para una matriz cuadrada n x n
double **alloc_matrix(int n) {
  double **mat = (double **)malloc(n * sizeof(double *));
  for (int i = 0; i < n; i++) {
    mat[i] = (double *)malloc(n * sizeof(double));
  }
  return mat;
}

// Inicializar matriz con valores aleatorios
void init_matrix(double **mat, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      mat[i][j] = (double)(rand() % 10);
    }
  }
}

// Multiplicación clásica de matrices C = A * B
void multiply(double **A, double **B, double **C, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      C[i][j] = 0.0;
      for (int k = 0; k < n; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

// Liberar memoria
void free_matrix(double **mat, int n) {
  for (int i = 0; i < n; i++) {
    free(mat[i]);
  }
  free(mat);
}

int main() {
  srand(time(NULL));

  int sizes[] = {100, 200, 500, 1000};
  int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

  for (int s = 0; s < num_sizes; s++) {
    int n = sizes[s];
    printf("\nMultiplicando matrices de tamaño %d x %d...\n", n, n);

    double **A = alloc_matrix(n);
    double **B = alloc_matrix(n);
    double **C = alloc_matrix(n);

    init_matrix(A, n);
    init_matrix(B, n);

    clock_t start = clock();
    multiply(A, B, C, n);
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Tiempo: %.3f segundos\n", elapsed);

    free_matrix(A, n);
    free_matrix(B, n);
    free_matrix(C, n);
  }

  return 0;
}
