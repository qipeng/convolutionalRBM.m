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
    int H, W, i, j, ii, jj, ni, ndima, ndimb, colors, color, Nfilters, nf, N, Wfilter, Hres, Wres;

    a = prhs[0];
    b = prhs[1];
    
    dimsa = mxGetDimensions(a);
    dimsb = mxGetDimensions(b);
    
    ndima = mxGetNumberOfDimensions(a);
    ndimb = mxGetNumberOfDimensions(b);
    
    H = dimsa[0]; W = dimsa[1];
    
    if (ndima <= 2) Nfilters = 1;
    else Nfilters = dimsa[2];
    if (ndima <= 3) N = 1;
    else N = dimsa[3];
    
    Wfilter = dimsb[0];
    if (ndimb <= 2) colors = 1;
    else colors = dimsb[2];
    
    Wres = W + Wfilter - 1; Hres = H + Wfilter - 1;
    
    dimsc = (mwSize*)mxMalloc(sizeof(mwSize)*4);
    dimsc[0] = Hres; dimsc[1] = Wres; dimsc[2] = colors; dimsc[3] = N;
    c = plhs[0] = mxCreateNumericArray(4, dimsc, mxDOUBLE_CLASS, mxREAL);
    mxFree(dimsc);
    
    aa = mxGetPr(a);
    bb = mxGetPr(b);
    cc = mxGetPr(c);
    
    for (ni = 0; ni < N; ni++)
        for (color = 0; color < colors; color++)
            for (j = 0; j < W; j++)
                for (i = 0; i < H; i++) {
                    
                    for (nf = 0; nf < Nfilters; nf++)
                        for (jj = 0; jj < Wfilter; jj++)
                            for (ii = 0; ii < Wfilter; ii++)
                                cc[(i+ii) + Hres * (j+jj) + Wres * Hres * color + colors * Wres * Hres * ni] +=
                                    aa[i + H * j + W * H * nf + Nfilters * W * H * ni] *
                                    bb[ii + Wfilter * jj + Wfilter * Wfilter * color + colors * Wfilter * Wfilter * nf];
                }
}
