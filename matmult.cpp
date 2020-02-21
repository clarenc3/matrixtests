#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <iomanip>
#include "TMatrixD.h"
#include <iostream>

void transpose(double *A, double *B, int n) {
  int i,j;
  for(i=0; i<n; i++) {
    for(j=0; j<n; j++) {
      B[j*n+i] = A[i*n+j];
    }
  }
}

void gemm(double *A, double *B, double *C, int n) {   
  int i, j, k;
  for (i = 0; i < n; i++) { 
    for (j = 0; j < n; j++) {
      double dot  = 0;
      for (k = 0; k < n; k++) {
        dot += A[i*n+k]*B[k*n+j];
      } 
      C[i*n+j ] = dot;
    }
  }
}

void gemm_omp(double *A, double *B, double *C, int n) {   
#pragma omp parallel for
  for (int i = 0; i < n; i++) { 
    for (int j = 0; j < n; j++) {
      double dot  = 0;
      for (int k = 0; k < n; k++) {
        dot += A[i*n+k]*B[k*n+j];
      } 
      C[i*n+j] = dot;
    }
  }
}

void gemmT(double *A, double *B, double *C, int n) {   
  double *B2 = (double*)malloc(sizeof(double)*n*n);
  transpose(B,B2, n);
  for (int i = 0; i < n; i++) { 
    for (int j = 0; j < n; j++) {
      double dot  = 0;
      for (int k = 0; k < n; k++) {
        dot += A[i*n+k]*B2[j*n+k];
      } 
      C[i*n+j ] = dot;
    }
  }
  free(B2);
}

void gemmT_omp(double *A, double *B, double *C, int n) {   
  double *B2 = (double*)malloc(sizeof(double)*n*n);
  transpose(B,B2, n);
#pragma omp parallel for
  for (int i = 0; i < n; i++) { 
    for (int j = 0; j < n; j++) {
      double dot  = 0;
      for (int k = 0; k < n; k++) {
        dot += A[i*n+k]*B2[j*n+k];
      } 
      C[i*n+j ] = dot;
    }
  }
  free(B2);
}

void tiled_mat_multiply(double *a, double *b, double *c, int n) {

  const int s = 1000;
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
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Need an argument, dimension" << std::endl;
    return -1;
  }

  int n = std::atoi(argv[1]);
  double *A = (double*)malloc(sizeof(double)*n*n);
  double *B = (double*)malloc(sizeof(double)*n*n);
  double *C = (double*)malloc(sizeof(double)*n*n);
  for (int i = 0; i < n*n; i++) {
    A[i] = 1.0*rand()/RAND_MAX; 
    B[i] = 1.0*rand()/RAND_MAX;
  }

  /*
  double dtime = omp_get_wtime();
  gemm(A,B,C, n);
  dtime = omp_get_wtime() - dtime;
  std::cout << std::setw(20) << "multipl: " << dtime << std::endl;

  dtime = omp_get_wtime();
  gemm_omp(A,B,C, n);
  dtime = omp_get_wtime() - dtime;
  std::cout << std::setw(20) << "multipl mp: " << dtime << std::endl;

  dtime = omp_get_wtime();
  gemmT(A,B,C, n);
  dtime = omp_get_wtime() - dtime;
  std::cout << std::setw(20) << "transpose: " << dtime << std::endl;

  dtime = omp_get_wtime();
  gemmT_omp(A,B,C, n);
  dtime = omp_get_wtime() - dtime;
  std::cout << std::setw(20) << "transpose mp: " << dtime << std::endl;

  dtime = omp_get_wtime();
  tiled_mat_multiply(A,B,C,n);
  dtime = omp_get_wtime() - dtime;
  std::cout << std::setw(20) << "tiled mp: " << dtime << std::endl;

  // Now compare to our friendly root implementation
  TMatrixD A_mat(n,n);
  TMatrixD B_mat(n,n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      A_mat(i,j) = A[i*n+j];
      B_mat(i,j) = B[i*n+j];
    }
  }
  dtime = omp_get_wtime();
  TMatrixD C_mat = A_mat*B_mat;
  dtime = omp_get_wtime()-dtime;
  std::cout << std::setw(20) << "ROOT: " << dtime << std::endl;
  */

  // Now compare to our friendly root implementation
  TMatrixD A_mat(n,n);
  TMatrixD B_mat(n,n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      A_mat(i,j) = A[i*n+j];
      B_mat(i,j) = B[i*n+j];
    }
  }
  //A_mat.Print();
  //B_mat.Print();

  (A_mat*B_mat).Print();
  gemmT_omp(A,B,C,n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      std::cout << std::setw(8) << C[i*n+j] << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}
