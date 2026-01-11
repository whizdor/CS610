#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>

using std::cerr;
using std::cout;
using std::endl;
using std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t;

#define INP_H (1 << 6)
#define INP_W (1 << 6)
#define INP_D (1 << 6)
#define FIL_H (3)
#define FIL_W (3)
#define FIL_D (3)

/** Cross-correlation without padding */
void cc_3d_no_padding(const uint64_t* input,
                      const uint64_t (*kernel)[FIL_W][FIL_D], uint64_t* result,
                      const uint64_t outputHeight, const uint64_t outputWidth,
                      const uint64_t outputDepth) {
  for (uint64_t i = 0; i < outputHeight; i++) {
    for (uint64_t j = 0; j < outputWidth; j++) {
      for (uint64_t k = 0; k < outputDepth; k++) {
        uint64_t sum = 0;
        for (uint64_t ki = 0; ki < FIL_H; ki++) {
          for (uint64_t kj = 0; kj < FIL_W; kj++) {
            for (uint64_t kk = 0; kk < FIL_D; kk++) {
              sum += input[(i + ki) * INP_W * INP_D + (j + kj) * INP_D +
                           (k + kk)] *
                     kernel[ki][kj][kk];
            }
          }
        }
        result[i * outputWidth * outputDepth + j * outputDepth + k] += sum;
      }
    }
  }
}

int main() {
  uint64_t* input = new uint64_t[INP_H * INP_W * INP_D];
  std::fill_n(input, INP_H * INP_W * INP_D, 1);

  uint64_t filter[FIL_H][FIL_W][FIL_D] = {{{2, 1, 3}, {2, 1, 3}, {2, 1, 3}},
                                          {{2, 1, 3}, {2, 1, 3}, {2, 1, 3}},
                                          {{2, 1, 3}, {2, 1, 3}, {2, 1, 3}}};

  uint64_t outputHeight = INP_H - FIL_H + 1;
  uint64_t outputWidth = INP_W - FIL_W + 1;
  uint64_t outputDepth = INP_D - FIL_D + 1;

  auto* result = new uint64_t[outputHeight * outputWidth * outputDepth]{0};
  cc_3d_no_padding(input, filter, result, outputHeight, outputWidth,
                   outputDepth);

  cout << "3D convolution without padding:\n";
  for (uint64_t i = 0; i < outputHeight; i++) {
    for (uint64_t j = 0; j < outputWidth; j++) {
      for (uint64_t k = 0; k < outputDepth; k++) {
        cout << result[i * outputWidth * outputDepth + j * outputDepth + k]
             << " ";
      }
      cout << "\n";
    }
    cout << "\n";
  }

  delete[] result;

  return EXIT_SUCCESS;
}
