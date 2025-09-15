/* matmul_blocked_vs_classic.c
 *
 * Compilar:
 *   gcc -O3 -march=native -funroll-loops matmul.c -o matmul
 *
 * Uso:
 *   ./matmul
 *
 * El programa imprimirá resultados en formato CSV:
 * size,method,block,seconds,checksum
 */

#define _POSIX_C_SOURCE 200112L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_seconds() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Reservar matriz n x n en un solo bloque contiguo (row-major) */
double *alloc_mat(int n) {
  // usar aligned allocation para mejor performance en SIMD
  void *ptr = NULL;
  size_t bytes = (size_t)n * n * sizeof(double);
#if defined(_ISOC11_SOURCE) || (__STDC_VERSION__ >= 201112L)
  ptr = aligned_alloc(64, ((bytes + 63) / 64) * 64); // múltiplo de 64
  if (!ptr) {
    ptr = malloc(bytes);
  }
#else
  if (posix_memalign(&ptr, 64, bytes) != 0)
    ptr = malloc(bytes);
#endif
  if (!ptr) {
    fprintf(stderr, "Error: no memory\n");
    exit(EXIT_FAILURE);
  }
  return (double *)ptr;
}

void free_mat(double *m) { free(m); }

/* Inicializa la matriz con valores pseudo-aleatorios (reproducible) */
void init_mat(double *A, int n, unsigned int seed) {
  srand(seed);
  for (int i = 0; i < n * n; ++i) {
    // valores pequeños para evitar overflow y reducir errores numericos
    A[i] = (double)(rand() % 10);
  }
}

void zero_mat(double *A, int n) {
  size_t nn = (size_t)n * n;
  for (size_t i = 0; i < nn; ++i)
    A[i] = 0.0;
}

/* checksum: suma de todos los elementos (para verificar igualdad) */
double checksum(double *A, int n) {
  double s = 0.0;
  for (int i = 0; i < n * n; ++i)
    s += A[i];
  return s;
}

/* Acceso auxiliar: elemento (i,j) en array row-major */
static inline double a_at(double *A, int n, int i, int j) {
  return A[(size_t)i * n + j];
}

/* Multiplicación clásica C = A * B (i, j, k) */
void matmul_classic(double *A, double *B, double *C, int n) {
  // C debe estar inicializada en cero antes de llamar
  for (int i = 0; i < n; ++i) {
    size_t rowi = (size_t)i * n;
    for (int j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int k = 0; k < n; ++k) {
        sum += A[rowi + k] * B[(size_t)k * n + j];
      }
      C[rowi + j] = sum;
    }
  }
}

/* Multiplicación por bloques (block size = bs)
   Implementación con 6 bucles: ii, jj, kk, i, j, k
   Maneja tamaños no múltiplos de bs usando min(...) */
void matmul_blocked(double *A, double *B, double *C, int n, int bs) {
  for (int ii = 0; ii < n; ii += bs) {
    int i_max = (ii + bs < n) ? (ii + bs) : n;
    for (int jj = 0; jj < n; jj += bs) {
      int j_max = (jj + bs < n) ? (jj + bs) : n;
      for (int kk = 0; kk < n; kk += bs) {
        int k_max = (kk + bs < n) ? (kk + bs) : n;

        // multiplicar bloque A[ii:i_max, kk:k_max] por B[kk:k_max, jj:j_max]
        for (int i = ii; i < i_max; ++i) {
          size_t rowi = (size_t)i * n;
          for (int k = kk; k < k_max; ++k) {
            double aik = A[rowi + k];
            size_t rowk = (size_t)k * n;
            // j interno
            for (int j = jj; j < j_max; ++j) {
              C[rowi + j] += aik * B[rowk + j];
            }
          }
        }
      }
    }
  }
}

int main(void) {
  // Tamaños a probar (puedes editarlos)
  int sizes[] = {128, 256, 512}; // tamaños típicos; ajusta según RAM/CPU
  int nsizes = sizeof(sizes) / sizeof(sizes[0]);

  // Bloqueos a probar (probar varios bs)
  int blocks[] = {8, 16, 32, 64};
  int nblocks = sizeof(blocks) / sizeof(blocks[0]);

  // Imprimir cabecera CSV
  printf("size,method,block,seconds,checksum\n");

  for (int si = 0; si < nsizes; ++si) {
    int n = sizes[si];

    // Reservar matrices
    double *A = alloc_mat(n);
    double *B = alloc_mat(n);
    double *C = alloc_mat(n);

    // Inicializar
    init_mat(A, n, 1234 + n);
    init_mat(B, n, 4321 + n);

    // CLÁSICA
    zero_mat(C, n);
    double t0 = now_seconds();
    matmul_classic(A, B, C, n);
    double t1 = now_seconds();
    double cs_classic = checksum(C, n);
    printf("%d,classical,NA,%.6f,%.12e\n", n, t1 - t0, cs_classic);

    // BLOQUEADA: probar varios tamaños de bloque
    for (int bi = 0; bi < nblocks; ++bi) {
      int bs = blocks[bi];
      zero_mat(C, n);
      double tb0 = now_seconds();
      matmul_blocked(A, B, C, n, bs);
      double tb1 = now_seconds();
      double cs_block = checksum(C, n);
      // verificar que checksum sea cercano al clásico (pequeña diferencia por
      // orden de sumas)
      double diff = fabs(cs_block - cs_classic);
      printf("%d,blocked,%d,%.6f,%.12e\n", n, bs, tb1 - tb0, cs_block);
      if (diff > 1e-6 * fabs(cs_classic + 1.0)) {
        fprintf(stderr,
                "WARNING: checksum difference for n=%d bs=%d -> diff=%.6e\n", n,
                bs, diff);
      }
    }

    free_mat(A);
    free_mat(B);
    free_mat(C);
  }

  return 0;
}
