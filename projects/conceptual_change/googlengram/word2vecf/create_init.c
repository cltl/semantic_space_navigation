#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vocab.h"
#include "io.h"

#define MAX_STRING 500

typedef float real;                    // Precision of float numbers

char wvocab_file[MAX_STRING], cvocab_file[MAX_STRING];
char output_file[MAX_STRING];
real *syn0, *syn1neg;
long long layer1_size = 100;

struct vocabulary *wv;
struct vocabulary *cv;

void InitNet(struct vocabulary *wv, struct vocabulary *cv) {
   long long a, b;
   a = posix_memalign((void **)&syn0, 128, (long long)wv->vocab_size * layer1_size * sizeof(real));
   if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
   for (b = 0; b < layer1_size; b++)
      for (a = 0; a < wv->vocab_size; a++)
         syn0[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;

   a = posix_memalign((void **)&syn1neg, 128, (long long)cv->vocab_size * layer1_size * sizeof(real));
   if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
   for (b = 0; b < layer1_size; b++)
      for (a = 0; a < cv->vocab_size; a++)
        syn1neg[a * layer1_size + b] = 0;
}

void write_initialization(char *output_file) {
  long a, b;
  FILE *fo;
    fo = fopen(output_file, "wb");
    for (a = 0; a < wv->vocab_size; a++) {
        for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
    }
    for (a = 0; a < cv->vocab_size; a++) {
        for (b = 0; b < layer1_size; b++) fwrite(&syn1neg[a * layer1_size + b], sizeof(real), 1, fo);
    }
  fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;

  if (argc == 1) {
    printf("Creating initialization matrices for word2vec\n\n");
    printf("Options:\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the parameter matrices (binary format)\n");
    printf("\t-wvocab filename\n");
    printf("\t\twords vocabulary file\n");
    printf("\t-cvocab filename\n");
    printf("\t\tcontexts vocabulary file\n");
  }

  if ((i = ArgPos((char *)"-wvocab", argc, argv)) > 0) strcpy(wvocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-cvocab", argc, argv)) > 0) strcpy(cvocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);

  wv = ReadVocab(wvocab_file);
  cv = ReadVocab(cvocab_file);
  InitNet(wv, cv);
  write_initialization(output_file);
    return 0;
}
