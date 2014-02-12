#ifndef __CUDA_UTILS_H__
#define __CUDA_UTILS_H__

#include "cutil.h"
#include <gmp.h>

#define WORDS_PER_INT 1024 / 8 / sizeof(unsigned int)

#define TO_DEV cudaMemcpyHostToDevice
#define TO_HOST cudaMemcpyDeviceToHost

#define THREADS_PER_BLOCK WORDS_PER_INT
#define NUM_BLOCKS 65504
#define NUM_STREAMS 4

typedef struct {
   unsigned int digits[WORDS_PER_INT];
} cmpz_t;

void cmpz_init_set(cmpz_t *target, mpz_t value);
void cmpz_to_mpz(cmpz_t *target, mpz_t value);
__global__ void factor_keys(cmpz_t *keys, unsigned char *result_matrix, unsigned int num_keys, unsigned int offset);
__device__ int cuda_gcd(cmpz_t *result, cmpz_t *a, cmpz_t *b);
__device__ void cmpz_rshift(cmpz_t *result, cmpz_t *value);
__device__ void cmpz_sub(cmpz_t *diff, cmpz_t *a, cmpz_t *b);
__device__ int cmpz_tz(cmpz_t *value);
__device__ int cmpz_gt(cmpz_t *a, cmpz_t *b);

#endif
