/*
 * matmul_cache_sim.c
 *
 * Compilar:
 *   gcc -std=c11 -O2 matmul_cache_sim.c -o matmul_cache_sim -lm
 *
 * Uso (valores por defecto si no pasas argumentos):
 *   ./matmul_cache_sim [n] [block] [cache_bytes] [line_bytes] [assoc] [verbose]
 *
 * Ejemplo pequeño (para trazar paso-a-paso):
 *   ./matmul_cache_sim 4 2 256 16 1 1
 *
 * Ejemplo realista (medir):
 *   ./matmul_cache_sim 256 32 32768 64 8 0
 *
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef unsigned long long u64;

/* ----------------------- Simulador de caché simple ----------------------- */

typedef struct {
  int valid;
  u64 tag;
  u64 last_used;
} CacheLine;

typedef struct {
  CacheLine *lines; // array of 'assoc' CacheLine
} CacheSet;

typedef struct {
  int line_size; // bytes por línea
  int assoc;     // vías
  int num_sets;
  CacheSet *sets;
  u64 timer;
  u64 hits;
  u64 misses;
  int verbose;
} Cache;

void cache_init(Cache *c, size_t cache_bytes, int line_size, int assoc,
                int verbose) {
  c->line_size = line_size;
  c->assoc = assoc;
  int num_lines = (int)(cache_bytes / line_size);
  if (num_lines < 1)
    num_lines = 1;
  c->num_sets = num_lines / assoc;
  if (c->num_sets < 1)
    c->num_sets = 1;
  c->sets = (CacheSet *)malloc(sizeof(CacheSet) * c->num_sets);
  for (int s = 0; s < c->num_sets; ++s) {
    c->sets[s].lines = (CacheLine *)malloc(sizeof(CacheLine) * assoc);
    for (int a = 0; a < assoc; ++a) {
      c->sets[s].lines[a].valid = 0;
      c->sets[s].lines[a].tag = 0;
      c->sets[s].lines[a].last_used = 0;
    }
  }
  c->timer = 1;
  c->hits = 0;
  c->misses = 0;
  c->verbose = verbose;
}

void cache_free(Cache *c) {
  for (int s = 0; s < c->num_sets; ++s)
    free(c->sets[s].lines);
  free(c->sets);
}

/* Simula acceso a la dirección 'addr' (bytes) */
void cache_access(Cache *c, uintptr_t addr, int is_write) {
  uintptr_t line_no = addr / (uintptr_t)c->line_size;
  int set_idx = (int)(line_no % (uintptr_t)c->num_sets);
  u64 tag = (u64)(line_no / (uintptr_t)c->num_sets);

  CacheSet *set = &c->sets[set_idx];
  int hit = 0;
  for (int a = 0; a < c->assoc; ++a) {
    if (set->lines[a].valid && set->lines[a].tag == tag) {
      hit = 1;
      set->lines[a].last_used = c->timer++;
      break;
    }
  }
  if (hit) {
    c->hits++;
    if (c->verbose)
      printf("  CACHE HIT set=%d tag=%llu\n", set_idx, (unsigned long long)tag);
    return;
  }
  /* miss -> replace LRU */
  c->misses++;
  if (c->verbose)
    printf("  CACHE MISS set=%d tag=%llu\n", set_idx, (unsigned long long)tag);
  int victim = -1;
  u64 oldest = (u64)(-1);
  for (int a = 0; a < c->assoc; ++a) {
    if (!set->lines[a].valid) {
      victim = a;
      break;
    }
    if (set->lines[a].last_used < oldest) {
      oldest = set->lines[a].last_used;
      victim = a;
    }
  }
  set->lines[victim].valid = 1;
  set->lines[victim].tag = tag;
  set->lines[victim].last_used = c->timer++;
}

/* ----------------------- Auxiliares de memoria y tiempo ------------------ */

double *alloc_mat(int n) {
  double *m = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
  if (!m) {
    fprintf(stderr, "No memory\n");
    exit(1);
  }
  return m;
}

void init_mat(double *M, int n, unsigned seed) {
  srand(seed);
  for (size_t i = 0; i < (size_t)n * (size_t)n; ++i)
    M[i] = (double)(rand() % 10);
}

void zero_mat(double *M, int n) {
  memset(M, 0, sizeof(double) * (size_t)n * (size_t)n);
}

static inline double now_seconds() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Lectura/escritura que simula acceso a caché (offset: elemento en double) */
static inline double cache_read(double *base, int offset, Cache *c) {
  uintptr_t addr = (uintptr_t)offset * sizeof(double);
  cache_access(c, addr, 0);
  return base[offset];
}

static inline void cache_write(double *base, int offset, double value,
                               Cache *c) {
  uintptr_t addr = (uintptr_t)offset * sizeof(double);
  cache_access(c, addr, 1);
  base[offset] = value;
}

/* ----------------------- Multiplicaciones (simuladas) ------------------- */

/* clásica i,j,k: C[i,j] = sum_k A[i,k]*B[k,j] */
void matmul_classic_sim(double *A, double *B, double *C, int n, Cache *c,
                        int verbose) {
  for (int i = 0; i < n; ++i) {
    int rowi = i * n;
    for (int j = 0; j < n; ++j) {
      int idxC = rowi + j;
      double sum = 0.0;
      for (int k = 0; k < n; ++k) {
        int idxA = rowi + k;
        int idxB = k * n + j;
        if (verbose)
          printf("Access A[%d,%d] idx=%d B[%d,%d] idx=%d Cidx=%d\n", i, k, idxA,
                 k, j, idxB, idxC);
        double a = cache_read(A, idxA, c);
        double b = cache_read(B, idxB, c);
        sum += a * b;
      }
      cache_write(C, idxC, sum, c);
    }
  }
}

/* bloqueada: ii,jj,kk con bs blocksize */
void matmul_blocked_sim(double *A, double *B, double *C, int n, int bs,
                        Cache *c, int verbose) {
  for (int ii = 0; ii < n; ii += bs) {
    int i_max = (ii + bs < n) ? ii + bs : n;
    for (int jj = 0; jj < n; jj += bs) {
      int j_max = (jj + bs < n) ? jj + bs : n;
      for (int kk = 0; kk < n; kk += bs) {
        int k_max = (kk + bs < n) ? kk + bs : n;
        for (int i = ii; i < i_max; ++i) {
          int rowi = i * n;
          for (int k = kk; k < k_max; ++k) {
            int idxA = rowi + k;
            double a = cache_read(A, idxA, c);
            for (int j = jj; j < j_max; ++j) {
              int idxB = k * n + j;
              int idxC = rowi + j;
              if (verbose)
                printf("Block access Aidx=%d Bidx=%d Cidx=%d\n", idxA, idxB,
                       idxC);
              double b = cache_read(B, idxB, c);
              double cval = cache_read(C, idxC, c);
              cval += a * b;
              cache_write(C, idxC, cval, c);
            }
          }
        }
      }
    }
  }
}

/* ----------------------- Programa principal ----------------------------- */

int main(int argc, char **argv) {
  int n = 128;
  int bs = 32;
  size_t cache_bytes = 32 * 1024;
  int line_bytes = 64;
  int assoc = 8;
  int verbose = 0;

  if (argc > 1)
    n = atoi(argv[1]);
  if (argc > 2)
    bs = atoi(argv[2]);
  if (argc > 3)
    cache_bytes = (size_t)atoi(argv[3]);
  if (argc > 4)
    line_bytes = atoi(argv[4]);
  if (argc > 5)
    assoc = atoi(argv[5]);
  if (argc > 6)
    verbose = atoi(argv[6]);

  printf("n=%d bs=%d cache=%zuB line=%dB assoc=%d verbose=%d\n", n, bs,
         cache_bytes, line_bytes, assoc, verbose);

  double *A = alloc_mat(n), *B = alloc_mat(n), *C = alloc_mat(n);
  init_mat(A, n, 123);
  init_mat(B, n, 321);

  /* CLÁSICO */
  Cache cache1;
  cache_init(&cache1, cache_bytes, line_bytes, assoc, verbose);
  zero_mat(C, n);
  double t0 = now_seconds();
  matmul_classic_sim(A, B, C, n, &cache1, verbose);
  double t1 = now_seconds();
  double time_classic = t1 - t0;
  u64 accesses_classic = cache1.hits + cache1.misses;
  printf("\nCLASSIC: time=%.6f s accesses=%llu hits=%llu misses=%llu "
         "hit_rate=%.4f\n",
         time_classic, (unsigned long long)accesses_classic,
         (unsigned long long)cache1.hits, (unsigned long long)cache1.misses,
         (double)cache1.hits / (double)accesses_classic);
  cache_free(&cache1);

  /* BLOQUEADO */
  Cache cache2;
  cache_init(&cache2, cache_bytes, line_bytes, assoc, verbose);
  zero_mat(C, n);
  double t2 = now_seconds();
  matmul_blocked_sim(A, B, C, n, bs, &cache2, verbose);
  double t3 = now_seconds();
  double time_blocked = t3 - t2;
  u64 accesses_blocked = cache2.hits + cache2.misses;
  printf("\nBLOCKED: time=%.6f s accesses=%llu hits=%llu misses=%llu "
         "hit_rate=%.4f\n",
         time_blocked, (unsigned long long)accesses_blocked,
         (unsigned long long)cache2.hits, (unsigned long long)cache2.misses,
         (double)cache2.hits / (double)accesses_blocked);
  cache_free(&cache2);

  free(A);
  free(B);
  free(C);
  return 0;
}
