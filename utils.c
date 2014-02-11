#include "utils.h"

gmp_randstate_t state;

void generate_keys(mpz_t n, mpz_t e, mpz_t d) {
   mpz_t p, q;

   mpz_init(p);
   mpz_init(q);

   while (!test_prime(p)) {
      get_random(p);
   }

   while (!test_prime(q)) {
      get_random(q);
   }

   mul(p, q, n);

   mpz_set_ui(e, E);
   calculate_d(p, q, e, d);
}

void output_factor(mpz_t n, mpz_t p) {
   mpz_t q, e, d;

   mpz_init(q);
   mpz_init_set_ui(e, E);
   mpz_init(d);

   my_div(n, p, q);
   calculate_d(p, q, e, d);
   print_int(n);
   printf(":");
   print_int(d);
   printf("\n");
}

void calculate_d(mpz_t p, mpz_t q, mpz_t e, mpz_t d) {
   mpz_t sub1, sub2, phi, one;
   mpz_init(sub1);
   mpz_init(sub2);
   mpz_init(phi);
   mpz_init_set_ui(one, 1);

   sub(p, one, sub1);
   sub(q, one, sub2);

   mul(sub1, sub2, phi);
   mod_inv(e, phi, d);
}

void encrypt(mpz_t data, mpz_t n, mpz_t e) {
   mod_exp(data, e, n, data);
}

void decrypt(mpz_t data, mpz_t n, mpz_t d) {
   mod_exp(data, d, n, data);
}

void init_rand() {
   gmp_randinit_default(state);
}

void get_random(mpz_t res) {
   mpz_urandomb(res, state, RANDOM_BITS);
}

int test_prime(mpz_t test) {
   return mpz_probab_prime_p(test, PRIME_REPS);
}

void print_int(mpz_t out) {
   write_int(stdout, out);
}

void write_int(FILE *fp, mpz_t out) {
   mpz_out_str(fp, BASE_10, out);
}

int read_int(FILE *fp, mpz_t ret) {
   return mpz_inp_str(ret, fp, BASE_10);
}

void add(mpz_t a, mpz_t b, mpz_t ret) {
   mpz_add(ret, a, b);
}

void sub(mpz_t a, mpz_t b, mpz_t ret) {
   mpz_sub(ret, a, b);
}

void mul(mpz_t a, mpz_t b, mpz_t ret) {
   mpz_mul(ret, a, b);
}

void my_div(mpz_t n, mpz_t d, mpz_t ret) {
   mpz_cdiv_q(ret, n, d);
}

void mod_inv(mpz_t inv, mpz_t mod, mpz_t ret) {
   mpz_invert(ret, inv, mod);
}

void mod_exp(mpz_t base, mpz_t exp, mpz_t mod, mpz_t ret) {
   mpz_powm_sec(ret, base, exp, mod);
}

void gcd(mpz_t a, mpz_t b, mpz_t ret) {
   mpz_gcd(ret, a, b);
}

int my_gcd(int a, int b) {
   if (a == b || b == 0) {
      return a;
   } else if (a > b) {
      return my_gcd(a - b, b);
   } else if (a < b) {
      return my_gcd(a, b - a);
   }

   return -1;
}
