double** covarianceBase::MatrixMult(double **A, double **B, int n) {
  // First make into monolithic array
  double *A_mon = new double[n*n];
  double *B_mon = new double[n*n];

#if MULTITHREAD
#pragma omp parallel for
#endif
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      A_mon[i*n+j] = A[i][j];
      B_mon[i*n+j] = B[i][j];
    }   
  }
  // Now call the monolithic calculator
  double C_mon = MatrixMult(A_mon, B_mon);
  delete A_mon;
  delete B_mon;

  // Return the double pointer
  double **C = new double*[n];
  for (int i = 0; i < n; ++i) C[i] = new double[n];
  for (int i = 0; i < n; ++i) {
    for (int i = 0; i < n; ++i) {
      C[i][j] = C_mon[i*n+j];
    }
  }
  delete C_mon;

  return C;
}

double* covarianceBase::MatrixMult(double *A, double *B, int n) {

  // First transpose to increse cache hits
  double *BT = new double[n*n];
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

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Need an argument, dimension" << std::endl;
    return -1;
  }

  int n = std::atoi(argv[1]);
  double *A = (double*)malloc(sizeof(double)*n*n);
  double *B = (double*)malloc(sizeof(double)*n*n);
  for (int i = 0; i < n*n; i++) {
    A[i] = 1.0*rand()/RAND_MAX; 
    B[i] = 1.0*rand()/RAND_MAX;
  }

  C = MatrixMult(A, B, n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      std::cout << std::setw(8) << C[i*n+j] << " ";
    }
    std::cout << std::endl;
  }

  // Now compare to our friendly root implementation
  TMatrixD A_mat(n,n);
  TMatrixD B_mat(n,n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      A_mat(i,j) = A[i*n+j];
      B_mat(i,j) = B[i*n+j];
    }
  }
  (A_mat*B_mat).Print();

}
