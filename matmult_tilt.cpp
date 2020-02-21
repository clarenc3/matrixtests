// https://en.wikipedia.org/wiki/Loop_nest_optimization
#include "omp.h"
#include <iostream>
#include <algorithm>
int main() {

  double dtime = omp_get_wtime();
  int i, j, a[1000][100], b[100], c[100];
  int n = 1000;
  for (i = 0; i < n; i++) {
    c[i] = 0;
    for (j = 0; j < n; j++) {
      c[i] = c[i] + a[i][j] * b[j];
    }
  }
  dtime = omp_get_wtime()-dtime;
  std::cout << dtime << "s" << std::endl;

  dtime = omp_get_wtime();
  //int i, j, x, y, a[1000][100], b[100], c[100];
  int  x, y;
  //int n = 1000;
  for (i = 0; i < n; i += 2) {
    c[i] = 0;
    c[i + 1] = 0;
    for (j = 0; j < n; j += 2) {
      for (x = i; x < std::min(i + 2, n); x++) {
        for (y = j; y < std::min(j + 2, n); y++) {
          c[x] = c[x] + a[x][y] * b[y];
        }
      }
    }
  }
  dtime = omp_get_wtime()-dtime;
  std::cout << dtime << "s" << std::endl;
}
