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
    int H, W, i, j, ii, jj, ni, ndima, ndimb, colors, color, Nfilters, nf, N, Wfilter, Hfilter, Wres, Hres;
    int strideH = 1, strideW = 1;

    a = prhs[0];
    b = prhs[1];
    
    dimsa = mxGetDimensions(a);
    dimsb = mxGetDimensions(b);
    
    ndima = mxGetNumberOfDimensions(a);
    ndimb = mxGetNumberOfDimensions(b);
    
    H = dimsa[0]; W = dimsa[1];
    if (ndima <= 2) colors = 1;
    else colors = dimsa[2];
    if (ndima <= 3) N = 1;
    else N = dimsa[3];
    
    Hfilter = dimsb[0];
    Wfilter = dimsb[1];
    if (ndimb <= 3) Nfilters = 1;
    else Nfilters = dimsb[3];

    if (nrhs > 2) {
    	strideW = (int)*((double*)mxGetPr(prhs[2]));
    	if (nrhs > 3)
    		strideH = (int)*((double*)mxGetPr(prhs[3]));
    	else 
    		strideH = strideW;
    }

    Wres = (W - Wfilter)/strideW + 1;
    Hres = (H - Hfilter)/strideH + 1;
   
    dimsc = (mwSize*)mxMalloc(sizeof(mwSize)*4);
    dimsc[0] = Hres; dimsc[1] = Wres; dimsc[2] = Nfilters; dimsc[3] = N;
    c = plhs[0] = mxCreateNumericArray(4, dimsc, mxDOUBLE_CLASS, mxREAL);
    mxFree(dimsc);
    
    aa = mxGetPr(a);
    bb = mxGetPr(b);
    cc = mxGetPr(c);
    
    for (ni = 0; ni < N; ni++)
        for (nf = 0; nf < Nfilters; nf++)
            for (j = 0; j < Wres; j++)
                for (i = 0; i < Hres; i++) {
                    int idxRes = i + Hres * j + Wres * Hres * nf + Nfilters * Wres * Hres * ni;
                    cc[idxRes] = 0;
                
                    for (color = 0; color < colors; color++)
                        for (jj = 0; jj < Wfilter; jj++)
                            for (ii = 0; ii < Hfilter; ii++)
                                cc[idxRes] += aa[(i*strideH+ii) + H * (j*strideW+jj) + W * H * color + colors * W * H * ni]
                                    * bb[ii + Hfilter * jj + Hfilter * Wfilter * color + colors * Hfilter * Wfilter * nf];
                }
}
