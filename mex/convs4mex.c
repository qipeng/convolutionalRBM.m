#include <math.h>
#include <matrix.h>
#include <mex.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const mxArray *a, *b, *t;
    mxArray *c;
    const mwSize *dimsa, *dimsb;
    mwSize *dimsc;
    double *aa, *bb, *cc;
    int n, m, i, j, ii, jj, ni, nz, ndima, ndimb, k, k0, K, K0, N;

    a = prhs[0];
    b = prhs[1];
    
    dimsa = mxGetDimensions(a);
    dimsb = mxGetDimensions(b);
    
    N = dimsa[0]; K0 = dimsa[1]; K = dimsb[1];
    
    n = dimsa[2];
    m = dimsb[2];
    nz = n - m + 1;
    dimsc = (mwSize*)mxMalloc(sizeof(mwSize) * 4);
    dimsc[0] = K0; dimsc[1] = K; dimsc[2] = nz; dimsc[3] = nz;
    c = plhs[0] = mxCreateNumericArray(4, dimsc, mxDOUBLE_CLASS, mxREAL);
    mxFree(dimsc);
    
    aa = mxGetPr(a);
    bb = mxGetPr(b);
    cc = mxGetPr(c);
    
    for (i = 0; i < nz; i++)
        for (j = 0; j < nz; j++)
            for (ii = 0; ii < m; ii++)
                for (jj = 0; jj < m; jj++)
                    for (k = 0; k < K; k++) 
                        for (k0 = 0; k0 < K0; k0++)
                            for (ni = 0; ni < N; ni++)
                                cc[(k0 + K0 * k + K0 * K * (i * nz + j))] += aa[ni + N * k0 + N * K0 * ((i + ii) * n + j + jj )] * bb[ni + N * k + N * K * (ii * m + jj)];
}
