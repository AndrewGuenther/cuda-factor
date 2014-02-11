#include "utils.h"
#include "cuda_utils.h"

int main(int argc, char **argv) {
   FILE *f;
   unsigned int num_keys = 0;
   unsigned int key_arr_size = KEY_CHUNK;
   mpz_t input;
   cmpz_t *keys = (cmpz_t *)malloc(sizeof(cmpz_t) * key_arr_size);

   if (argc != 2) {
      printf("Usage:\n\tgpu-crack <key-file>\n");
      return 0;
   }

   f = fopen(argv[1], "r");

   if (f == NULL) {
      return 0;
   }

   int bytes_read = 1;
   mpz_init(input);
   while (bytes_read > 0) {
      if (num_keys >= key_arr_size) {
         key_arr_size *= 2;
         keys = (cmpz_t *)realloc(keys, sizeof(cmpz_t) * key_arr_size);
      }

      bytes_read = read_int(f, input);
      cmpz_init_set(&(keys[num_keys++]), input);
   }
   fclose(f);
   num_keys--;

   printf("%d KEYS READ SUCCESSFULLY\n", num_keys);

   cmpz_t *dev_keys;
   unsigned int *dev_result_matrix, *result_matrix;
   result_matrix = (unsigned int *)calloc(1, RESULT_BITMAP_SIZE);

   // Allocate space on the card and copy the keys
   CUDA_SAFE_CALL(cudaMalloc((void **)&dev_keys, sizeof(cmpz_t) * num_keys));
   CUDA_SAFE_CALL(cudaMemcpy(dev_keys, keys, sizeof(cmpz_t) * num_keys, TO_DEV));

   // Allocate space for the result bitmap
   CUDA_SAFE_CALL(cudaMalloc((void **)&dev_result_matrix, RESULT_BITMAP_SIZE));
   CUDA_SAFE_CALL(cudaMemset(dev_result_matrix, 0, RESULT_BITMAP_SIZE));

   // Run the kernel
   cuda_gcd<<<WORDS_PER_INT, FACTORS_PER_KERNEL>>>(dev_keys, dev_result_matrix, num_keys);

   // Copy the result matrix back
   CUDA_SAFE_CALL(cudaMemcpy(result_matrix, dev_result_matrix, RESULT_BITMAP_SIZE, TO_HOST));

   // Free device mallocs
   CUDA_SAFE_CALL(cudaFree(dev_result_matrix));
   CUDA_SAFE_CALL(cudaFree(dev_keys));

   // Print the first ten elements of the result matrix
   int idx;
   for (idx = 0; idx < 10; idx++) {
      printf("%u\n", result_matrix[idx]);
   }
}
