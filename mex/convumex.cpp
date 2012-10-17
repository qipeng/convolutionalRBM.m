#include <math.h>
#include <matrix.h>
#include <mex.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const mxArray *a, *b;
    mxArray *c;
    const mwSize *dimsa, *dimsb;
    double *aa, *bb, *cc;
    int n, m, i, j, ii, jj, nz;

    a = prhs[0];
    b = prhs[1];
    
    dimsa = mxGetDimensions(a);
    dimsb = mxGetDimensions(b);
        
    n = dimsa[0];
    m = dimsb[0];
    nz = n - m + 1;
    c = plhs[0] = mxCreateDoubleMatrix(nz, nz, mxREAL);
    
    aa = mxGetPr(a);
    bb = mxGetPr(b);
    cc = mxGetPr(c);
    
    for (i = 0; i < nz; i++)
        for (j = 0; j < nz; j++)
            for (ii = 0; ii < m; ii++)
                for (jj = 0; jj < m; jj++)
                    cc[i * nz + j] += aa[(i + ii) * n + j + jj] * bb[ii * m + jj];
}
