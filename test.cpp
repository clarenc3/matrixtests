#include <iostream>
#include <stdio.h>
#include <omp.h>

using namespace std;

#define SIZE 10000
#define N 1000
#define S 25

int n = N;
int s = S;

double a[SIZE],b[SIZE],c[SIZE];

// Initializing the matrices 

void mat_init(double *a, double *b, int n) {
  for(int i=0; i<n; i++) {
    for(int j=0; j<n; j++) {
      a[i*n + j] = 1;
    }
  }

  for(int i=0; i<n; i++) {
    for(int j=0; j<n; j++) {
      b[i*n + j] = 2;
    }
  }

}


void mat_multi(double *a, double *b, double *c, int n) {

  double start=omp_get_wtime();
#pragma omp parallel for
  for(int i=0; i<n; i++) {
    for(int j=0; j<n; j++) {
      for(int k=0; k<n; k++) {
        c[i*n+j]+=a[i*n+k]*b[k*n+j];
      }
    }
  }

  start = omp_get_wtime() - start;
  std::cout << "naive multiplication took " << start << "s" << std::endl;
}

void mat_empty(double *a, int n) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      c[i*n+j]=0;
    }
  }
}

void tiled_mat_multiply(double *a, double *b, double *c, int n) {

  double start=omp_get_wtime();

  omp_set_nested(1);

#pragma omp parallel for shared(a,b,c)
  for (int i1 = 0; i1 < n; i1+=s) {
    for (int j1 = 0; j1 < n; j1+=s) {
      for (int k1 = 0; k1 < n; k1+=s) {
        for(int i=i1; i <i1+s && i<n; i++) {
          for (int j=j1; j< j1+s && j<n; ++j) {
            for(int k=k1; k< k1+s && k<n; ++k) {
              c[i*n+j]+=a[i*n+k]*b[k*n+j];
            }
          }
        }
      }
    }
  }

  start = omp_get_wtime() - start;
  std::cout << "tiled multiplication took " << start << "s" << std::endl;

}

int main() {
  mat_init(a,b,n);
  mat_multi(a,b,c,n);
  //mat_print(c,n);
  mat_empty(c,n);
  tiled_mat_multiply(a,b,c,n);
  //mat_print(c,n);
  return 0;
}
