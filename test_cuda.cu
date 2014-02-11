#include "utils.h"
#include "cuda_utils.h"
#include "cutil.h"

#define TEST(val) if (val) {printf("PASSED\n");} else {printf("FAILED\n");}

__global__ void test_sanity(int *wut);
__global__ void test_rshift(cmpz_t *result, cmpz_t *a);
__global__ void test_sub(cmpz_t *result, cmpz_t *a, cmpz_t *b);
__global__ void test_tz(cmpz_t *result, cmpz_t *a);
__global__ void test_gt(cmpz_t *result, cmpz_t *a, cmpz_t *b);
__global__ void test_gcd(cmpz_t *result, cmpz_t *a, cmpz_t *b);

int main(int argc, char **argv) {
   FILE *fa, *fb;
   mpz_t a, b, result, control;

   if (argc != 3) {
      printf("Usage:\n\ttest_cuda <file1> <file2>\n");
      return 0;
   }

   fa = fopen(argv[1], "r");
   fb = fopen(argv[2], "r");

   if (fa == NULL || fb == NULL) {
      return 0;
   }

   mpz_init(a);
   mpz_init(b);
   mpz_init(result);
   mpz_init(control);

   read_int(fa, a);
   read_int(fb, b);

   cmpz_t ca, cb, cres, *dev_a, *dev_b, *dev_res;
   cmpz_init_set(&ca, a);
   cmpz_init_set(&cb, b);

   int devCount;
   cudaGetDeviceCount(&devCount);
   printf("CUDA Device Query...\n");
   printf("There are %d CUDA devices.\n", devCount);

   int wut, *dev_wut;
   CUDA_SAFE_CALL(cudaMalloc((void **)&dev_wut, sizeof(int)));
   CUDA_SAFE_CALL(cudaMemset(dev_wut, 0, sizeof(int)));
   printf("Sanity Check\n");
   test_sanity<<<1, 1>>>(dev_wut);
   CUDA_SAFE_CALL(cudaMemcpy(&wut, dev_wut, sizeof(int), TO_HOST));
   cudaDeviceSynchronize();
   printf("%d\n", wut);

   CUDA_SAFE_CALL(cudaMalloc((void **)&dev_a, sizeof(cmpz_t)));
   CUDA_SAFE_CALL(cudaMemcpy(dev_a, &ca, sizeof(cmpz_t), TO_DEV));

   CUDA_SAFE_CALL(cudaMalloc((void **)&dev_b, sizeof(cmpz_t)));
   CUDA_SAFE_CALL(cudaMemcpy(dev_b, &cb, sizeof(cmpz_t), TO_DEV));

   CUDA_SAFE_CALL(cudaMalloc((void **)&dev_res, sizeof(cmpz_t)));

   printf("Test rshift\n");
   CUDA_SAFE_CALL(cudaMemset(dev_res, 0, sizeof(cmpz_t)));
   test_rshift<<<1, WORDS_PER_INT>>>(dev_res, dev_a);
   CUDA_SAFE_CALL(cudaMemcpy(&cres, dev_res, sizeof(cmpz_t), TO_HOST));
   cmpz_to_mpz(&cres, result);
   mpz_tdiv_q_2exp(control, a, 1);
   TEST(!mpz_cmp(result, control));

   /*
   print_int(control);
   printf("\n");
   print_int(result);
   printf("\n");
   */

   printf("Test sub\n");
   CUDA_SAFE_CALL(cudaMemset(dev_res, 0, sizeof(cmpz_t)));
   test_sub<<<1, WORDS_PER_INT>>>(dev_res, dev_a, dev_b);
   CUDA_SAFE_CALL(cudaMemcpy(&cres, dev_res, sizeof(cmpz_t), TO_HOST));
   cmpz_to_mpz(&cres, result);
   sub(a, b, control);
   TEST(!mpz_cmp(result, control));

   /*
   print_int(control);
   printf("\n");
   print_int(result);
   printf("\n");
   */

   printf("Test tz\n");
   CUDA_SAFE_CALL(cudaMemset(dev_res, 0, sizeof(cmpz_t)));
   test_tz<<<1, WORDS_PER_INT>>>(dev_res, dev_a);
   CUDA_SAFE_CALL(cudaMemcpy(&cres, dev_res, sizeof(cmpz_t), TO_HOST));
   cmpz_to_mpz(&cres, result);
   print_int(result);
   printf("\n");

   printf("Test gt\n");
   CUDA_SAFE_CALL(cudaMemset(dev_res, 0, sizeof(cmpz_t)));
   test_gt<<<1, WORDS_PER_INT>>>(dev_res, dev_a, dev_b);
   CUDA_SAFE_CALL(cudaMemcpy(&cres, dev_res, sizeof(cmpz_t), TO_HOST));
   cmpz_to_mpz(&cres, result);
   print_int(result);
   printf("\n");

   CUDA_SAFE_CALL(cudaMemset(dev_res, 0, sizeof(cmpz_t)));
   test_gt<<<1, WORDS_PER_INT>>>(dev_res, dev_b, dev_a);
   CUDA_SAFE_CALL(cudaMemcpy(&cres, dev_res, sizeof(cmpz_t), TO_HOST));
   cmpz_to_mpz(&cres, result);
   print_int(result);
   printf("\n");

   printf("Test gcd\n");
   CUDA_SAFE_CALL(cudaMemset(dev_res, 0, sizeof(cmpz_t)));
   test_gcd<<<1, WORDS_PER_INT>>>(dev_res, dev_a, dev_b);
   CUDA_SAFE_CALL(cudaMemcpy(&cres, dev_res, sizeof(cmpz_t), TO_HOST));
   cmpz_to_mpz(&cres, result);
   print_int(result);
   printf("\n");
}

__global__ void test_sanity(int *wut) {
   *wut = 1;
}

__global__ void test_rshift(cmpz_t *result, cmpz_t *a) {
   cmpz_rshift(result, a);
}

__global__ void test_sub(cmpz_t *result, cmpz_t *a, cmpz_t *b) {
   cmpz_sub(result, a, b);
}

__global__ void test_tz(cmpz_t *result, cmpz_t *a) {
   result->digits[0] = cmpz_tz(a);
}

__global__ void test_gt(cmpz_t *result, cmpz_t *a, cmpz_t *b) {
   result->digits[0] = cmpz_gt(a, b);
}

__global__ void test_gcd(cmpz_t *result, cmpz_t *a, cmpz_t *b) {
   cuda_gcd(result, a, b);
}
