#include "utils.h"
#include "cuda_utils.h"

int main(int argc, char **argv) {
   FILE *f;
   uint32_t num_keys = 0;
   uint32_t key_arr_size = KEY_CHUNK;
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

   DEBUG("KEYS READ SUCCESSFULLY\n");

   cmpz_t *dev_keys;
   unsigned char *dev_result_matrix, *result_matrix;
   result_matrix = (unsigned char *)calloc(sizeof(char), num_keys);

   // Allocate space on the card and copy the keys
   CUDA_SAFE_CALL(cudaMalloc((void **)&dev_keys, sizeof(cmpz_t) * num_keys));
   CUDA_SAFE_CALL(cudaMemcpy(dev_keys, keys, sizeof(cmpz_t) * num_keys, TO_DEV));

   // Allocate space for the result bitmap
   CUDA_SAFE_CALL(cudaMalloc((void **)&dev_result_matrix, num_keys * sizeof(char)));
   CUDA_SAFE_CALL(cudaMemset(dev_result_matrix, 0, num_keys * sizeof(char)));

   // Run the kernel
   uint32_t offset = 0;
   while (offset < (num_keys * num_keys) / 2) {
      fprintf(stderr, "Kernel call: %d\n", offset);
      factor_keys<<<NUM_BLOCKS, WORDS_PER_INT>>>(dev_keys, dev_result_matrix, num_keys, offset);
      offset += NUM_BLOCKS;
   }

   // Copy the result matrix back
   CUDA_SAFE_CALL(cudaMemcpy(result_matrix, dev_result_matrix, num_keys * sizeof(char), TO_HOST));

   // Free device mallocs
   CUDA_SAFE_CALL(cudaFree(dev_result_matrix));
   CUDA_SAFE_CALL(cudaFree(dev_keys));

   // Print the first ten elements of the result matrix
   mpz_t gcd_res, a, b;
   mpz_init(gcd_res);
   mpz_init(a);
   mpz_init(b);
   int idx, cmp;
   for (idx = 0; idx < num_keys; idx++) {
      if (result_matrix[idx]) {
         for (cmp = idx + 1; cmp < num_keys; cmp++) {
            if (result_matrix[cmp]) {
               cmpz_to_mpz(&keys[idx], a);
               cmpz_to_mpz(&keys[cmp], b);
               gcd(a, b, gcd_res);
               if (mpz_cmp_ui(gcd_res, 1)) {
                  output_factor(a, gcd_res);
                  output_factor(b, gcd_res);
               }
            }
         }
      }
   }
}
