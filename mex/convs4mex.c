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
    int n, m, i, j, ii, jj, ni, ndima, ndimb, N, Nfilters, colors, Wfilter, Hfilter, W, H, W1, H1, nf, color;
    int strideH = 1, strideW = 1;

    a = prhs[0];
    b = prhs[1];
    
    dimsa = mxGetDimensions(a);
    dimsb = mxGetDimensions(b);
    
    ndima = mxGetNumberOfDimensions(a);
    ndimb = mxGetNumberOfDimensions(b);
    
    H = dimsa[0]; W = dimsa[1]; 
    H1 = dimsb[0]; W1 = dimsb[1];
    if (ndima <= 2) colors = 1;
    else colors = dimsa[2];
    if (ndima <= 3) N = 1;
    else N = dimsa[3];
    if (ndimb <= 2) Nfilters = 1;
    else Nfilters = dimsb[2];
    
    if (nrhs > 2) {
        strideW = (int)*((double*)mxGetPr(prhs[2]));

        if (nrhs > 3)
            strideH = (int)*((double*)mxGetPr(prhs[3]));
        else
            strideH = strideW;
    }
    
    Wfilter = W - (W1 - 1) * strideW;
    Hfilter = H - (H1 - 1) * strideH;
    
    dimsc = (mwSize*)mxMalloc(sizeof(mwSize) * 4);
    dimsc[0] = Hfilter; dimsc[1] = Wfilter; dimsc[2] = colors; dimsc[3] = Nfilters;
    c = plhs[0] = mxCreateNumericArray(4, dimsc, mxDOUBLE_CLASS, mxREAL);
    mxFree(dimsc);
    
    aa = mxGetPr(a);
    bb = mxGetPr(b);
    cc = mxGetPr(c);
    
    for (nf = 0; nf < Nfilters; nf++)
        for (color = 0; color < colors; color++)
            for (i = 0; i < Wfilter; i++)
                for (j = 0; j < Hfilter; j++) {
                    int idxRes = j + Hfilter * i + Hfilter * Wfilter * color + colors * Hfilter * Wfilter * nf;
                    cc[idxRes] = 0;
                    
                    for (ni = 0; ni < N; ni++) 
                        for (ii = 0; ii < W1; ii++)
                            for (jj = 0; jj < H1; jj++)
                                cc[idxRes] += bb[jj + H1 * ii + W1 * H1 * nf + Nfilters * W1 * H1 * ni]
                                    * aa[(jj * strideH + j) + H * (ii * strideW + i) + W * H * color + colors * W * H * ni];
                }
}
