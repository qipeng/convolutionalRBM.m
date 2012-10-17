#include <math.h>
#include <matrix.h>
#include <mex.h>
#include <string.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const mxArray *a, *b, *c;
    mxArray *res, *poolres;
    const mwSize *dimsa;
    double *aa, *bb, *cc, *resp, *pvalue, *poolresp;
    int p, np, nh, iter, i, j, jj, k, kk, idx, idx2;


    a = prhs[0];
    b = prhs[1];
    c = prhs[2];
    
    dimsa = mxGetDimensions(a);
    
        
    nh = dimsa[0];
    aa = mxGetPr(b);
    p = (int)(*aa);
    aa = mxGetPr(c);
    iter = (int)(*aa);
    np = nh / p;
    res = plhs[0] = mxCreateDoubleMatrix(nh, nh, mxREAL);
    if (nlhs > 1)
        poolres = plhs[1] = mxCreateDoubleMatrix(np, np, mxREAL);
    bb = new double[nh * nh];
    pvalue = new double[np * np];
    
    aa = mxGetPr(a);
    resp = mxGetPr(res);
    if (nlhs > 1) poolresp = mxGetPr(poolres);
    
    memcpy(bb, aa, sizeof(double) * nh * nh);
    
    for (i = 0; i < iter; i++) {
        for (j = 0; j < nh * nh; j++)
            bb[j] = exp(bb[j]);
            
        for (j = 0; j < np; j++)
            for (k = 0; k < np; k++) {
                idx = j * np + k;
                pvalue[idx] = 1;
                
                for (jj = 0; jj < p; jj++)
                    for (kk = 0; kk < p; kk++)
                        pvalue[idx] += bb[(j * p + jj) * nh + k * p + kk];
                        
                for (jj = 0; jj < p; jj++)
                    for (kk = 0; kk < p; kk++) {
                        idx2 = (j * p + jj) * nh + k * p + kk;
                        bb[idx2] /= pvalue[idx];
                        if (i < iter - 1) {
                            bb[idx2] += aa[idx2];
                            bb[idx2] /= 2;
                        }
                    }
            }
    }
    
    memcpy(resp, bb, sizeof(double) * nh * nh);
    if (nlhs > 1) {
        for (i = 0; i < np * np; i++)
            pvalue[i] = 1 / pvalue[i];
        memcpy(poolresp, pvalue, sizeof(double) * np * np);
    }
    
    delete bb;
    delete pvalue;
}