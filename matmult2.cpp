#include <iostream>
#include <iomanip>
#include "omp.h"
#include "TMatrixD.h"

double** MatrixMult(double **A, double **B, int n);
double* MatrixMult(double *A, double *B, int n);
TMatrixD MatrixMult(TMatrixD, TMatrixD);

double** MatrixMult(double **A, double **B, int n) {
  // First make into monolithic array
  double *A_mon = new double[n*n];
  double *B_mon = new double[n*n];

#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      A_mon[i*n+j] = A[i][j];
      B_mon[i*n+j] = B[i][j];
    }   
  }
  // Now call the monolithic calculator
  double *C_mon = MatrixMult(A_mon, B_mon, n);
  delete A_mon;
  delete B_mon;

  // Return the double pointer
  double **C = new double*[n];
  for (int i = 0; i < n; ++i) C[i] = new double[n];
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      C[i][j] = C_mon[i*n+j];
    }
  }
  delete C_mon;

  return C;
}

double* MatrixMult(double *A, double *B, int n) {

  // First transpose to increse cache hits
  double *BT = new double[n*n];
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
            BT[j*n+i] = B[i*n+j];
    }
  }

  // Now multiply
  double *C = new double[n*n];
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0;
      for (int k = 0; k < n; k++) {
        sum += A[i*n+k]*BT[j*n+k];
      }
      C[i*n+j] = sum;
    }
  }
  delete BT;

  return C;
}

// Multi-threaded matrix multiplication
TMatrixD MatrixMult(TMatrixD A, TMatrixD B) {
  double *C_mon = MatrixMult(A.GetMatrixArray(), B.GetMatrixArray(), A.GetNcols());
  TMatrixD C;
  C.Use(A.GetNcols(), A.GetNrows(), C_mon);
  return C;
}


int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Need an argument, dimension" << std::endl;
    return -1;
  }

  int n = std::atoi(argv[1]);
  double *A = (double*)malloc(sizeof(double)*n*n);
  double *B = (double*)malloc(sizeof(double)*n*n);
#pragma omp parallel for
  for (int i = 0; i < n*n; i++) {
    A[i] = 1.0*rand()/RAND_MAX; 
    B[i] = 1.0*rand()/RAND_MAX;
  }

  double wtime = omp_get_wtime();
  double *C = MatrixMult(A, B, n);
  wtime = omp_get_wtime()-wtime;
  int nthreads = 0;
#pragma omp parallel
  {
  nthreads = omp_get_num_threads();
  }
  std::cout << "Custom " << nthreads << " threaded took " << wtime << std::endl;

  // Now compare to our friendly root implementation
  TMatrixD A_mat(n,n);
  TMatrixD B_mat(n,n);
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      A_mat(i,j) = A[i*n+j];
      B_mat(i,j) = B[i*n+j];
    }
  }
  wtime = omp_get_wtime();
  TMatrixD C_mat = (A_mat*B_mat);
  wtime = omp_get_wtime()-wtime;
  std::cout << "ROOT single threaded took " << wtime << std::endl;

  wtime = omp_get_wtime();
  TMatrixD C_mat2 = MatrixMult(A_mat,B_mat);
  wtime = omp_get_wtime()-wtime;
  std::cout << "ROOT custom multi threaded took " << wtime << std::endl;

  // Try feeding the matrices in
  int nerrors = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (C_mat(i,j) != C[i*n+j] || C_mat(i,j) != C_mat2(i,j)) {
        std::cerr << "Entry " << i << ", " << j << " did not match" << std::endl;
        nerrors++;
      }
    }
  }
  std::cout << "Found " << nerrors << " errors in calculations" << std::endl;
}
