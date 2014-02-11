#include "cuda_utils.h"

void cmpz_init_set(cmpz_t *target, mpz_t value) {
   mpz_export(target, NULL, -1, sizeof(cmpz_t), 0, 0, value);
}

void cmpz_to_mpz(cmpz_t *target, mpz_t value) {
   mpz_import(value, WORDS_PER_INT, -1, sizeof(unsigned int), 0, 0, target);
}

__global__ void cuda_gcd(cmpz_t *keys, unsigned int *result_matrix, unsigned int num_keys) {
   cmpz_t *u, *v, *t;

   unsigned long long position = (blockDim.x * blockIdx.x) + (blockDim.y * blockIdx.y) + blockIdx.z;
   unsigned long long x = position / num_keys;
   unsigned long long y = position % num_keys;

   if (y > x) {
      return;
   }

   u = keys + x;
   v = keys + y;

   while (__any(v->digits[threadIdx.x])) {
      /* remove all factors of 2 in v -- they are not common */
      /*   note: v is not zero, so while will terminate */
      while (!cmpz_tz(v)) {  /* Loop X */
         cmpz_rshift(v, v);
      }

      /* Now u and v are both odd. Swap if necessary so u <= v,
         then set v = v - u (which is even). For bignums, the
         swapping is just pointer movement, and the subtraction
         can be done in-place. */
      if (cmpz_gt(u, v)) {
         t = v;
         v = u;
         u = t;
      }
      cmpz_sub(v, v, u);
   }

   result_matrix[blockIdx.x] = __any(v->digits[threadIdx.x]);
}

// Shift value right by 1 bit and store result in result
__device__ void cmpz_rshift(cmpz_t *result, cmpz_t *value) {
   int i = threadIdx.x;
   unsigned int savedBit = 0;
   if (i < 31) {
      savedBit = value->digits[i + 1];
   }
   result->digits[i] = (unsigned int)((value->digits[i] >> 1) | (savedBit << 31));
}

// Sets diff to a - b
__device__ void cmpz_sub(cmpz_t *diff, cmpz_t *a, cmpz_t *b) {
   int i = threadIdx.x;
   double borrow = 0;
   double difference = (double)(a->digits[i]) - b->digits[i];
   int wtf = 0;

   if (difference < 0) {
      a->digits[i + 1]--;
      borrow += UINT_MAX;
      wtf = 1;
   }
   __syncthreads();

   difference =(double)(a->digits[i]) - b->digits[i] + borrow + wtf;
   //printf("difference: %lf\n", difference);
   diff->digits[i] = (unsigned int) difference;
}

__device__ int cmpz_tz(cmpz_t *value) {
   if (threadIdx.x != WORDS_PER_INT - 1) {
      return 0;
   }

   return value->digits[threadIdx.x] & 0x1;
}

__device__ int cmpz_gt(cmpz_t *a, cmpz_t *b) {
   unsigned char result[WORDS_PER_INT];
   result[threadIdx.x] = a->digits[threadIdx.x] >= b->digits[threadIdx.x];

   int i = 0;
   while (result[i] == 1) {
      i++;
   }

   return i == WORDS_PER_INT;
}
