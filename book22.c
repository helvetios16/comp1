#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX 1000

double A[MAX][MAX], x[MAX], y[MAX];

int main() {
  int i, j;

  // Inicializamos el vector x y la matriz A
  for (i = 0; i < MAX; i++) {
    x[i] = 1.0; // valores simples para probar
    y[i] = 0.0;
    for (j = 0; j < MAX; j++) {
      A[i][j] = (double)(i + j) / MAX;
    }
  }

  // Primera versión (mejor localidad)
  clock_t start1 = clock();
  for (i = 0; i < MAX; i++) {
    for (j = 0; j < MAX; j++) {
      y[i] += A[i][j] * x[j];
    }
  }
  clock_t end1 = clock();

  printf("Tiempo primera versión: %f segundos\n",
         (double)(end1 - start1) / CLOCKS_PER_SEC);

  // Reiniciamos y
  for (i = 0; i < MAX; i++) {
    y[i] = 0.0;
  }

  // Segunda versión (peor localidad)
  clock_t start2 = clock();
  for (j = 0; j < MAX; j++) {
    for (i = 0; i < MAX; i++) {
      y[i] += A[i][j] * x[j];
    }
  }
  clock_t end2 = clock();

  printf("Tiempo segunda versión: %f segundos\n",
         (double)(end2 - start2) / CLOCKS_PER_SEC);

  return EXIT_SUCCESS;
}
