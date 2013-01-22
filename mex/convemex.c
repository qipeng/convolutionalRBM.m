#include <math.h>
#include <matrix.h>
#include <mex.h>

#define min(a,b) ((a)<(b))?(a):(b)

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const mxArray *a, *b;
    mxArray *c;
    const mwSize *dimsa, *dimsb;
    mwSize *dimsc;
    double *aa, *bb, *cc;
    int n, m, i, j, ii, jj, nz, ni, ndima, ndimb, k, k0, K, K0, N;

    a = prhs[0];
    b = prhs[1];
    
    dimsa = mxGetDimensions(a);
    dimsb = mxGetDimensions(b);
    
    N = dimsa[0]; K0 = dimsb[0]; K = dimsb[1];
    
    n = dimsa[2];
    m = dimsb[2];
    nz = n + m - 1;
    dimsc = (mwSize*)mxMalloc(sizeof(mwSize)*5);
    dimsc[0] = N; dimsc[1] = K0; dimsc[2] = K; dimsc[3] = nz; dimsc[4] = nz;
    c = plhs[0] = mxCreateNumericArray(5, dimsc, mxDOUBLE_CLASS, mxREAL);
    mxFree(dimsc);
    
    aa = mxGetPr(a);
    bb = mxGetPr(b);
    cc = mxGetPr(c);
    
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            for (ii = 0; ii < m; ii++)
                for (jj = 0; jj < m; jj++) 
                    for (k = 0; k < K; k++)
                        for (k0 = 0; k0 < K0; k0++)
                            for (ni = 0; ni < N; ni++)
                                cc[ni + N * k0 +  N * K0 * k + N * K0 * K * ((i+ii) * nz + j+jj)] += aa[ni + N * k + N * K * (i * n + j)] * bb[k0 + K0 * k + K0 * K * (ii * m + jj)];
}
