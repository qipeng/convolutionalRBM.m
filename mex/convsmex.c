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
    
    ndima = mxGetNumberOfDimensions(a);
    ndimb = mxGetNumberOfDimensions(b);
    
    N = dimsa[0]; K0 = dimsa[1]; K = dimsb[1];
    
    if (ndima <= 2) 
        n = 1;
    else
        n = dimsa[2];
    if (ndimb <= 2)
        m = 1;
    else
        m = dimsb[2];
    nz = n - m + 1;
    dimsc = (mwSize*)mxMalloc(sizeof(mwSize)*5);
    dimsc[0] = N; dimsc[1] = K0; dimsc[2] = K; dimsc[3] = nz; dimsc[4] = nz;
    c = plhs[0] = mxCreateNumericArray(5, dimsc, mxDOUBLE_CLASS, mxREAL);
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
                                cc[ni + N * k0 + N * K0 * k + N * K0 * K * (i * nz + j)] += aa[ni + N * k0 + N * K0 * ((i + ii) * n + j + jj )] * bb[k0 + K0 * k + K0 * K * (ii * m + jj)];
}
