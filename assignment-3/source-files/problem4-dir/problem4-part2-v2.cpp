#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define NSEC_SEC_MUL (1.0e9)

// OpenMP VERSION 2: Nested parallelism with collapse directive
// Strategy: Collapse multiple outer loops for finer-grained parallelism
void gridloopsearch(
    double dd1, double dd2, double dd3, double dd4, double dd5, double dd6, double dd7, double dd8,
    double dd9, double dd10, double dd11, double dd12, double dd13, double dd14, double dd15,
    double dd16, double dd17, double dd18, double dd19, double dd20, double dd21, double dd22,
    double dd23, double dd24, double dd25, double dd26, double dd27, double dd28, double dd29,
    double dd30, double c11, double c12, double c13, double c14, double c15, double c16, double c17,
    double c18, double c19, double c110, double d1, double ey1, double c21, double c22, double c23,
    double c24, double c25, double c26, double c27, double c28, double c29, double c210, double d2,
    double ey2, double c31, double c32, double c33, double c34, double c35, double c36, double c37,
    double c38, double c39, double c310, double d3, double ey3, double c41, double c42, double c43,
    double c44, double c45, double c46, double c47, double c48, double c49, double c410, double d4,
    double ey4, double c51, double c52, double c53, double c54, double c55, double c56, double c57,
    double c58, double c59, double c510, double d5, double ey5, double c61, double c62, double c63,
    double c64, double c65, double c66, double c67, double c68, double c69, double c610, double d6,
    double ey6, double c71, double c72, double c73, double c74, double c75, double c76, double c77,
    double c78, double c79, double c710, double d7, double ey7, double c81, double c82, double c83,
    double c84, double c85, double c86, double c87, double c88, double c89, double c810, double d8,
    double ey8, double c91, double c92, double c93, double c94, double c95, double c96, double c97,
    double c98, double c99, double c910, double d9, double ey9, double c101, double c102,
    double c103, double c104, double c105, double c106, double c107, double c108, double c109,
    double c1010, double d10, double ey10, double kk) {
  
  long pnts = 0;
  
  const double e1 = kk * ey1;
  const double e2 = kk * ey2;
  const double e3 = kk * ey3;
  const double e4 = kk * ey4;
  const double e5 = kk * ey5;
  const double e6 = kk * ey6;
  const double e7 = kk * ey7;
  const double e8 = kk * ey8;
  const double e9 = kk * ey9;
  const double e10 = kk * ey10;
  
  const int s1 = (int)floor((dd2 - dd1) / dd3);
  const int s2 = (int)floor((dd5 - dd4) / dd6);
  const int s3 = (int)floor((dd8 - dd7) / dd9);
  const int s4 = (int)floor((dd11 - dd10) / dd12);
  const int s5 = (int)floor((dd14 - dd13) / dd15);
  const int s6 = (int)floor((dd17 - dd16) / dd18);
  const int s7 = (int)floor((dd20 - dd19) / dd21);
  const int s8 = (int)floor((dd23 - dd22) / dd24);
  const int s9 = (int)floor((dd26 - dd25) / dd27);
  const int s10 = (int)floor((dd29 - dd28) / dd30);
  
  FILE* fptr = fopen("./results-part2-v2.txt", "w");
  if (fptr == NULL) {
    printf("Error in creating file!\n");
    exit(1);
  }
  
  const int total_r9_r10 = s9 * s10;
  
  double* x9_arr = (double*)malloc(total_r9_r10 * sizeof(double));
  double* x10_arr = (double*)malloc(total_r9_r10 * sizeof(double));
  double* base1_arr = (double*)malloc(total_r9_r10 * sizeof(double));
  double* base2_arr = (double*)malloc(total_r9_r10 * sizeof(double));
  double* base3_arr = (double*)malloc(total_r9_r10 * sizeof(double));
  double* base4_arr = (double*)malloc(total_r9_r10 * sizeof(double));
  double* base5_arr = (double*)malloc(total_r9_r10 * sizeof(double));
  double* base6_arr = (double*)malloc(total_r9_r10 * sizeof(double));
  double* base7_arr = (double*)malloc(total_r9_r10 * sizeof(double));
  double* base8_arr = (double*)malloc(total_r9_r10 * sizeof(double));
  double* base9_arr = (double*)malloc(total_r9_r10 * sizeof(double));
  double* base10_arr = (double*)malloc(total_r9_r10 * sizeof(double));
  
  // Pre-compute arrays
  int idx = 0;
  for (int r9 = 0; r9 < s9; ++r9) {
    const double x9 = dd25 + r9 * dd27;
    const double c19_x9 = c19 * x9, c29_x9 = c29 * x9, c39_x9 = c39 * x9, c49_x9 = c49 * x9, c59_x9 = c59 * x9;
    const double c69_x9 = c69 * x9, c79_x9 = c79 * x9, c89_x9 = c89 * x9, c99_x9 = c99 * x9, c109_x9 = c109 * x9;
    
    for (int r10 = 0; r10 < s10; ++r10, ++idx) {
      const double x10 = dd28 + r10 * dd30;
      x9_arr[idx] = x9;
      x10_arr[idx] = x10;
      base1_arr[idx] = c19_x9 + c110 * x10 - d1;
      base2_arr[idx] = c29_x9 + c210 * x10 - d2;
      base3_arr[idx] = c39_x9 + c310 * x10 - d3;
      base4_arr[idx] = c49_x9 + c410 * x10 - d4;
      base5_arr[idx] = c59_x9 + c510 * x10 - d5;
      base6_arr[idx] = c69_x9 + c610 * x10 - d6;
      base7_arr[idx] = c79_x9 + c710 * x10 - d7;
      base8_arr[idx] = c89_x9 + c810 * x10 - d8;
      base9_arr[idx] = c99_x9 + c910 * x10 - d9;
      base10_arr[idx] = c109_x9 + c1010 * x10 - d10;
    }
  }
  
  // PARALLEL REGION: preserve output order identical to sequential version
  // We buffer results per (r1,r2,r3) chunk and flush in lexicographic order later.

  const int T1 = s1, T2 = s2, T3 = s3;
  const size_t total_chunks = (size_t)T1 * (size_t)T2 * (size_t)T3;

  // One buffer per chunk: chunk_bufs[i] with length chunk_lens[i]
  char **chunk_bufs = (char**)calloc(total_chunks, sizeof(char*));
  size_t *chunk_lens = (size_t*)calloc(total_chunks, sizeof(size_t));
  long   *chunk_hits = (long*)calloc(total_chunks, sizeof(long));
  if (!chunk_bufs || !chunk_lens || !chunk_hits) {
    fprintf(stderr, "Out of memory for chunk buffers\n");
    exit(EXIT_FAILURE);
  }

  // We will accumulate pnts after the loop from chunk_hits
  long pnts_local_sum = 0;

  #pragma omp parallel for collapse(3) schedule(dynamic,2) reduction(+:pnts_local_sum)
  for (int r1 = 0; r1 < s1; ++r1) {
    for (int r2 = 0; r2 < s2; ++r2) {
      for (int r3 = 0; r3 < s3; ++r3) {

        // Compute linear chunk index in lexicographic order
        size_t chunk_idx = ((size_t)r1 * (size_t)s2 + (size_t)r2) * (size_t)s3 + (size_t)r3;

        // Each chunk writes into its own memory stream
        char *buf = NULL;
        size_t len = 0;
        FILE *m = open_memstream(&buf, &len);
        if (!m) {
          // Fallback (very unlikely): skip buffering for this chunk
          continue;
        }

        long chunk_count = 0;

        double x1 = dd1 + r1 * dd3;
        double x2_base = dd4 + r2 * dd6; // will be updated per r2 but kept here for clarity
        double x3_base = dd7 + r3 * dd9;

        for (int r4 = 0; r4 < s4; ++r4) {
          double x4 = dd10 + r4 * dd12;

          for (int r5 = 0; r5 < s5; ++r5) {
            double x5 = dd13 + r5 * dd15;

            for (int r6 = 0; r6 < s6; ++r6) {
              double x6 = dd16 + r6 * dd18;

              for (int r7 = 0; r7 < s7; ++r7) {
                double x7 = dd19 + r7 * dd21;

                for (int r8 = 0; r8 < s8; ++r8) {
                  double x8 = dd22 + r8 * dd24;

                  // Recompute x2 and x3 once per (r2,r3)
                  double x2 = x2_base;
                  double x3 = x3_base;

                  // Partials that do not depend on r9/r10
                  const double partial1  = c11  * x1 + c12  * x2 + c13  * x3 + c14  * x4 + c15  * x5 + c16  * x6 + c17  * x7 + c18  * x8;
                  const double partial2  = c21  * x1 + c22  * x2 + c23  * x3 + c24  * x4 + c25  * x5 + c26  * x6 + c27  * x7 + c28  * x8;
                  const double partial3  = c31  * x1 + c32  * x2 + c33  * x3 + c34  * x4 + c35  * x5 + c36  * x6 + c37  * x7 + c38  * x8;
                  const double partial4  = c41  * x1 + c42  * x2 + c43  * x3 + c44  * x4 + c45  * x5 + c46  * x6 + c47  * x7 + c48  * x8;
                  const double partial5  = c51  * x1 + c52  * x2 + c53  * x3 + c54  * x4 + c55  * x5 + c56  * x6 + c57  * x7 + c58  * x8;
                  const double partial6  = c61  * x1 + c62  * x2 + c63  * x3 + c64  * x4 + c65  * x5 + c66  * x6 + c67  * x7 + c68  * x8;
                  const double partial7  = c71  * x1 + c72  * x2 + c73  * x3 + c74  * x4 + c75  * x5 + c76  * x6 + c77  * x7 + c78  * x8;
                  const double partial8  = c81  * x1 + c82  * x2 + c83  * x3 + c84  * x4 + c85  * x5 + c86  * x6 + c87  * x7 + c88  * x8;
                  const double partial9  = c91  * x1 + c92  * x2 + c93  * x3 + c94  * x4 + c95  * x5 + c96  * x6 + c97  * x7 + c98  * x8;
                  const double partial10 = c101 * x1 + c102 * x2 + c103 * x3 + c104 * x4 + c105 * x5 + c106 * x6 + c107 * x7 + c108 * x8;

                  // Iterate r9,r10 in the SAME traversal order as the sequential code:
                  // r9 outer, r10 inner. Our precomputed arrays already follow this order,
                  // and 'i = 0..total_r9_r10-1' enumerates exactly that lexicographic sequence.
                  for (int i = 0; i < total_r9_r10; ++i) {
                    const double abs_q1  = fabs(partial1  + base1_arr[i]);
                    const double abs_q2  = fabs(partial2  + base2_arr[i]);
                    const double abs_q3  = fabs(partial3  + base3_arr[i]);
                    const double abs_q4  = fabs(partial4  + base4_arr[i]);
                    const double abs_q5  = fabs(partial5  + base5_arr[i]);
                    const double abs_q6  = fabs(partial6  + base6_arr[i]);
                    const double abs_q7  = fabs(partial7  + base7_arr[i]);
                    const double abs_q8  = fabs(partial8  + base8_arr[i]);
                    const double abs_q9  = fabs(partial9  + base9_arr[i]);
                    const double abs_q10 = fabs(partial10 + base10_arr[i]);

                    if (abs_q1 <= e1 && abs_q2 <= e2 && abs_q3 <= e3 && abs_q4 <= e4 &&
                        abs_q5 <= e5 && abs_q6 <= e6 && abs_q7 <= e7 && abs_q8 <= e8 &&
                        abs_q9 <= e9 && abs_q10 <= e10) {

                      // Emit to the chunk buffer; order within a chunk matches the sequential nest
                      fprintf(m, "%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",
                              x1, x2, x3, x4, x5, x6, x7, x8, x9_arr[i], x10_arr[i]);
                      ++chunk_count;
                    }
                  }
                }
              }
            }
          }
        }

        // Finalize the memory stream and store results for this chunk
        fclose(m); // this sets buf and len
        chunk_bufs[chunk_idx] = buf;
        chunk_lens[chunk_idx] = len;
        chunk_hits[chunk_idx] = chunk_count;
        pnts_local_sum += chunk_count;
      }
    }
  }

  // Now write all chunk buffers to the output file in the exact lexicographic order
  // of (r1, r2, r3) to match the baseline's line order.
  for (int r1 = 0; r1 < s1; ++r1) {
    for (int r2 = 0; r2 < s2; ++r2) {
      for (int r3 = 0; r3 < s3; ++r3) {
        size_t chunk_idx = ((size_t)r1 * (size_t)s2 + (size_t)r2) * (size_t)s3 + (size_t)r3;
        if (chunk_lens[chunk_idx] > 0 && chunk_bufs[chunk_idx]) {
          fwrite(chunk_bufs[chunk_idx], 1, chunk_lens[chunk_idx], fptr);
          free(chunk_bufs[chunk_idx]);
        }
      }
    }
  }

  // Update pnts and free bookkeeping
  pnts = pnts_local_sum;
  free(chunk_bufs);
  free(chunk_lens);
  free(chunk_hits);
}

struct timespec begin_grid, end_main;
double a[120], b[30];

int main() {
  FILE* fp = fopen("./disp.txt", "r");
  if (fp == NULL) { printf("Error: could not open file\n"); return 1; }
  for (int i = 0; i < 120 && !feof(fp); i++) {
    if (fscanf(fp, "%lf", &a[i]) != 1) {
      printf("Error: fscanf failed\n"); fclose(fp); return EXIT_FAILURE;
    }
  }
  fclose(fp);
  
  FILE* fpq = fopen("./grid.txt", "r");
  if (fpq == NULL) { printf("Error: could not open file\n"); return 1; }
  for (int j = 0; j < 30 && !feof(fpq); j++) {
    if (fscanf(fpq, "%lf", &b[j]) != 1) {
      printf("Error: fscanf failed\n"); fclose(fpq); return EXIT_FAILURE;
    }
  }
  fclose(fpq);
  
  printf("Number of OpenMP threads: %d\n", omp_get_max_threads());
  
  clock_gettime(CLOCK_MONOTONIC_RAW, &begin_grid);
  gridloopsearch(b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12],
                 b[13], b[14], b[15], b[16], b[17], b[18], b[19], b[20], b[21], b[22], b[23], b[24],
                 b[25], b[26], b[27], b[28], b[29], a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
                 a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19],
                 a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31],
                 a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43],
                 a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52], a[53], a[54], a[55],
                 a[56], a[57], a[58], a[59], a[60], a[61], a[62], a[63], a[64], a[65], a[66], a[67],
                 a[68], a[69], a[70], a[71], a[72], a[73], a[74], a[75], a[76], a[77], a[78], a[79],
                 a[80], a[81], a[82], a[83], a[84], a[85], a[86], a[87], a[88], a[89], a[90], a[91],
                 a[92], a[93], a[94], a[95], a[96], a[97], a[98], a[99], a[100], a[101], a[102],
                 a[103], a[104], a[105], a[106], a[107], a[108], a[109], a[110], a[111], a[112],
                 a[113], a[114], a[115], a[116], a[117], a[118], a[119], 0.3);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end_main);
  
  printf("Total time = %f seconds\n", (end_main.tv_nsec - begin_grid.tv_nsec) / NSEC_SEC_MUL +
                                          (end_main.tv_sec - begin_grid.tv_sec));
  
  return EXIT_SUCCESS;
}