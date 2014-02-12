#ifndef __UTILS_H__
#define __UTILS_H__

#include <gmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define RANDOM_BITS 512
#define PRIME_REPS 25
#define BASE_10 10
#define E 65537
#define KEY_CHUNK 100
#define DEBUG(str) fprintf(stderr, str);

void init_rand();
void generate_keys(mpz_t n, mpz_t e, mpz_t d);
void output_factor(mpz_t n, mpz_t p);
void calculate_d(mpz_t p, mpz_t q, mpz_t e, mpz_t d);
void encrypt(mpz_t data, mpz_t n, mpz_t e);
void decrypt(mpz_t data, mpz_t n, mpz_t d);
void get_random(mpz_t ret);
int test_prime(mpz_t test);
void print_int(mpz_t out);
void write_int(FILE *fp, mpz_t out);
int read_int(FILE *fp, mpz_t ret);
void add(mpz_t a, mpz_t b, mpz_t ret);
void sub(mpz_t a, mpz_t b, mpz_t ret);
void mul(mpz_t a, mpz_t b, mpz_t ret);
void my_div(mpz_t n, mpz_t d, mpz_t ret);
void mod_inv(mpz_t inv, mpz_t mod, mpz_t ret);
void mod_exp(mpz_t base, mpz_t exp, mpz_t mod, mpz_t ret);
void gcd(mpz_t a, mpz_t b, mpz_t ret);
int my_gcd(int a, int b);

#endif
