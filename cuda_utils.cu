#include "cuda_utils.h"

#define CUDA_DEBUG(str) if (threadIdx.x == 0) {printf(str);}

void cmpz_init_set(cmpz_t *target, mpz_t value) {
   mpz_export(target, NULL, -1, sizeof(cmpz_t), 0, 0, value);
}

void cmpz_to_mpz(cmpz_t *target, mpz_t value) {
   mpz_import(value, WORDS_PER_INT, -1, sizeof(uint32_t), 0, 0, target);
}


__global__ void factor_keys(cmpz_t *keys, unsigned char *result_matrix, uint32_t num_keys, uint32_t offset) {
   unsigned long long position = offset + blockIdx.x;

   if (position > (num_keys * num_keys) / 2) {
      return;
   }

   uint32_t x = position / num_keys;
   uint32_t y = position % num_keys;

   if (x < y) {
      x = num_keys - x - 1;
      y = num_keys - y - 1;
   }

   cmpz_t *u, *v;
   cmpz_t result;

   int i;
   for (i = 0; i < WORDS_PER_INT; i++) {
      result.digits[i] = 0;
   }

   u = &keys[x];
   v = &keys[y];

   int has_gcd = cuda_gcd(&result, u, v);

   result_matrix[x] |= has_gcd;
   result_matrix[y] |= has_gcd;
}

__device__ int cuda_gcd(cmpz_t *result, cmpz_t *in_u, cmpz_t *in_v) {
   __shared__ cmpz_t su, sv;
   cmpz_t *t, *u = &su, *v = &sv;

   u->digits[threadIdx.x] = in_u->digits[threadIdx.x];
   v->digits[threadIdx.x] = in_v->digits[threadIdx.x];
   __syncthreads();

   while (__any(v->digits[threadIdx.x])) {
      while (cmpz_tz(v)) {
         cmpz_rshift(v, v);
      }

      if (cmpz_gt(u, v)) {
         t = u;
         u = v;
         v = t;
      }
      cmpz_sub(v, v, u);
   }

   cmpz_rshift(u, u);
   return __any(u->digits[threadIdx.x]);
   //result->digits[threadIdx.x] = u->digits[threadIdx.x];
}

// Shift value right by 1 bit and store result in result
__device__ void cmpz_rshift(cmpz_t *result, cmpz_t *value) {
   int i = threadIdx.x;

   uint32_t savedBit = 0;
   if (i < 31) {
      savedBit = value->digits[i + 1];
   }
   __syncthreads();
   result->digits[i] = (uint32_t)((value->digits[i] >> 1) | (savedBit << 31));
   __syncthreads();
}

// Sets diff to a - b
__device__ void cmpz_sub(cmpz_t *diff, cmpz_t *a, cmpz_t *b) {
   __shared__ uint32_t borrows[32];
   uint32_t t;

   if (!threadIdx.x) {
      borrows[0] = 0;
   }

   t = a->digits[threadIdx.x] - b->digits[threadIdx.x];

   if (threadIdx.x != WORDS_PER_INT - 1) {
      borrows[threadIdx.x + 1] = (t > a->digits[threadIdx.x]);
   }
   __syncthreads();

   while (__any(borrows[threadIdx.x])) {
      if (borrows[threadIdx.x]) {
         t--;
      }

      if (threadIdx.x != WORDS_PER_INT - 1) {
         borrows[threadIdx.x + 1] = (t == 0xffffffffU && 
               borrows[threadIdx.x]);
      }
   }
   __syncthreads();
   diff->digits[threadIdx.x] = t;
   __syncthreads();
}
/*
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
diff->digits[i] = (uint32_t) difference;
}
 */

__device__ int cmpz_tz(cmpz_t *value) {
   return !(value->digits[0] & 0x1);
}

__device__ int cmpz_gt(cmpz_t *a, cmpz_t *b) {
   __shared__ double result[WORDS_PER_INT];
   result[threadIdx.x] = (double)(a->digits[threadIdx.x]) - b->digits[threadIdx.x];
   __syncthreads();

   int i = WORDS_PER_INT - 1;
   while (result[i] == 0 && i) {
      i--;
   }

   return result[i] > 0;
}
