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
    int n, m, i, j, ii, jj, nz, ndima, ndimb, k, k0, K, K0, dc, da, db, 
        a3d, b3d, a4d, b4d;

    a = prhs[0];
    b = prhs[1];
    
    dimsa = mxGetDimensions(a);
    dimsb = mxGetDimensions(b);
    
    ndima = mxGetNumberOfDimensions(a);
    ndimb = mxGetNumberOfDimensions(b);
    K0 = 1; K = 1;
    
    switch (ndima) {
        case 4: a4d=1; a3d=0;break;
        case 3: a4d=0; a3d=1;K=dimsa[2];break;
        default: a4d=0; a3d=0;break;
    }
    
    switch (ndimb) {
        case 4: b4d=1; b3d=0;K0=dimsb[3];K=dimsb[2];break;
        case 3: b4d=0; b3d=1;break;
        default: b4d=0; b3d=0;break;
    }
        
    n = dimsa[0];
    m = dimsb[0];
    nz = n + m - 1;
    dimsc = new mwSize[3];
    dimsc[0] = nz; dimsc[1] = nz; dimsc[2] = K0;// dimsc[3] = K0;
    c = plhs[0] = mxCreateNumericArray(3, dimsc, mxDOUBLE_CLASS, mxREAL);
    
    dc = nz * nz;
    db = m * m;
    da = n * n;
    
    aa = mxGetPr(a);
    bb = mxGetPr(b);
    cc = mxGetPr(c);
    
    for (k = 0; k < K; k++)
        for (k0 = 0; k0 < K0; k0++)
            for (i = 0; i < n; i++)
                for (j = 0; j < n; j++)
                    for (ii = 0; ii < m; ii++)
                        for (jj = 0; jj < m; jj++) 
                            cc[(i+ii) * nz + j+jj + dc * (k0)] += aa[i * n + j + da*(a3d* k + a4d*(k0*K+k))] * bb[ii * m + jj + db*(b3d* k + b4d*(k0 * K+k))];
}
