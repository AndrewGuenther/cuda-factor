#include "utils.h"

#include <stdlib.h>

#define KEY_CHUNK 100

int main(int argc, char **argv) {
   FILE *f;
   unsigned int num_keys = 0;
   unsigned int key_arr_size = KEY_CHUNK;
   mpz_t *keys = (mpz_t *)malloc(sizeof(mpz_t) * key_arr_size);

   if (argc != 2) {
      printf("Usage:\n\tcpu-crack <key-file>\n");
      return 0;
   }

   f = fopen(argv[1], "r");

   if (f == NULL) {
      return 0;
   }

   int bytes_read = 1;
   while (bytes_read > 0) {
      if (num_keys >= key_arr_size) {
         key_arr_size *= 2;
         keys = (mpz_t *)realloc(keys, sizeof(mpz_t) * key_arr_size);
      }

      mpz_init(keys[num_keys]);
      bytes_read = read_int(f, keys[num_keys++]);
   }
   fclose(f);
   num_keys--;

   DEBUG("KEYS READ SUCCESSFULLY\n");

   int row, col;
   mpz_t result;

   mpz_init(result);
   for (row = 0; row < num_keys; row++) {
      fprintf(stderr, "%d\n", row);
      for (col = row + 1; col < num_keys; col++) {
         gcd(keys[row], keys[col], result);
         if (mpz_cmp_ui(result, 1)) {
            output_factor(keys[row], result);
            output_factor(keys[col], result);
         }
      }
   }
   DEBUG("DONE\n");
}
