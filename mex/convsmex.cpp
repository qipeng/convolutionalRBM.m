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
    int n, m, i, j, ii, jj, nz, ndima, ndimb, k, k0, K, K0, dc, da, db, 
        a3d, b3d, a4d, b4d, c4d = 0;

    a = prhs[0];
    b = prhs[1];
    if (nrhs > 2) {
        t = prhs[2];
        aa = mxGetPr(t);
        c4d = (int)*aa;
    }
    
    dimsa = mxGetDimensions(a);
    dimsb = mxGetDimensions(b);
    
    ndima = mxGetNumberOfDimensions(a);
    ndimb = mxGetNumberOfDimensions(b);
    K0 = 1; K = 1;
    
    switch (ndima) {
        case 4: a4d=1; a3d=0;break;
        case 3: a4d=0; a3d=1;K0=dimsa[2];break;
        default: a4d=0; a3d=0;break;
    }
    
    switch (ndimb) {
        case 4: b4d=1; b3d=0;K0=dimsb[3];K=dimsb[2];break;
        case 3: b4d=0; b3d=1;K=dimsb[2];break;
        default: b4d=0; b3d=0;break;
    }
        
    n = dimsa[0];
    m = dimsb[0];
    nz = n - m + 1;
    if (c4d) {
        dimsc = new mwSize[4];
        dimsc[0] = nz; dimsc[1] = nz; dimsc[2] = K; dimsc[3] = K0;
        c = plhs[0] = mxCreateNumericArray(4, dimsc, mxDOUBLE_CLASS, mxREAL);
    } else
        c = plhs[0] = mxCreateDoubleMatrix(nz, nz, mxREAL);
    
    dc = nz * nz;
    db = m * m;
    da = n * n;
    
    aa = mxGetPr(a);
    bb = mxGetPr(b);
    cc = mxGetPr(c);
    
    for (k0 = 0; k0 < K0; k0++)
        for (k = 0; k < K; k++) {
      //      double sumB = 0;
            for (i = 0; i < nz; i++)
                for (j = 0; j < nz; j++)
                    for (ii = 0; ii < m; ii++)
                        for (jj = 0; jj < m; jj++)
                            cc[i * nz + j + c4d * (k0*K+k) * dc] += aa[(i + ii) * n + j + jj + da*(a3d* k0 + a4d*(K*k0+k))] * bb[ii * m + jj + db*(b3d* k + b4d*(K*k0+k))];
        }
}
