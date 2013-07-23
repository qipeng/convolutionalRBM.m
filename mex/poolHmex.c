#include <math.h>
#include <matrix.h>
#include <mex.h>
#include <string.h>
#include <time.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const mxArray *a, *b, *c;
    mxArray *res, *poolres, *poolsample;
    const mwSize *dimsa;
    mwSize *dimso;
    double *aa, *bb, *cc, *resp, *pvalue, *poolresp, *bias, *sample;
    int p, i, k, i1, j1, i2, j2, idx, idx2, N, K, W, H, Wres, Hres, Nfilters, ndima;
    double *max;

    a = prhs[0];
    b = prhs[1];
    c = prhs[2];
    
    dimsa = mxGetDimensions(a);
    ndima = mxGetNumberOfDimensions(a);
        
    H = dimsa[0];
    W = dimsa[1];
    if (ndima <= 2) Nfilters = 1;
    else Nfilters = dimsa[2];
    if (ndima <= 3) N = 1;
    else N = dimsa[3];
    
    bias = mxGetPr(b);
    aa = mxGetPr(c);
    p = (int)(*aa);
    Hres = H / p;
    Wres = W / p;
    
    dimso = (mwSize*)mxMalloc(sizeof(mwSize)*4);
    dimso[0] = H; dimso[1] = W; dimso[2] = Nfilters; dimso[3] = N;
    res = plhs[0] = mxCreateNumericArray(4, dimso, mxDOUBLE_CLASS, mxREAL);
    
    if (nlhs > 1) {
        dimso[0] = Hres; dimso[1] = Wres; dimso[2] = Nfilters; dimso[3] = N;
        poolres = plhs[1] = mxCreateNumericArray(4, dimso, mxDOUBLE_CLASS, mxREAL);
        pvalue = mxGetPr(poolres);
    } else 
        pvalue = (double*)mxMalloc(sizeof(double) * Hres * Wres * Nfilters * N);
        
    if (nlhs > 2) {
        dimso[0] = H; dimso[1] = W; dimso[2] = Nfilters; dimso[3] = N;
        poolsample = plhs[2] = mxCreateNumericArray(4, dimso, mxDOUBLE_CLASS, mxREAL);
        sample = mxGetPr(poolsample);
    }
    
    mxFree(dimso);
    
    aa = mxGetPr(a);
    resp = mxGetPr(res);
    bb = resp;
    
    memcpy(bb, aa, sizeof(double) * H * W * Nfilters * N);
    max = (double*)mxMalloc(sizeof(double) * Nfilters * N);
    
    for (i1 = 0; i1 < Wres; i1++)
        for (j1 = 0; j1 < Hres; j1++) {
            idx = i1 * Hres + j1;
            
            for (k = 0; k < Nfilters*N; k++)
                max[k] = -1e10;
            
            for (i = 0; i < N; i++)
                for (k = 0; k < Nfilters; k++)
                    for (i2 = 0; i2 < p; i2++)
                        for (j2 = 0; j2 < p; j2++) {
                            double t = bb[(i1 * p + i2) * H + j1 * p + j2 + W * H * k + Nfilters * W * H * i] + bias[k];
                            if (t > max[k + Nfilters * i])
                                max[k + Nfilters * i] = t;
                        }
            
            for (i = 0; i < N; i++)
                for (k = 0; k < Nfilters; k++)
                    pvalue[idx + Wres * Hres * k + Nfilters * Wres * Hres * i] = exp(-max[k + Nfilters * i]);
            
            for (i = 0; i < N; i++)
                for (k = 0; k < Nfilters; k++)
                    for (i2 = 0; i2 < p; i2++)
                        for (j2 = 0; j2 < p; j2++) {
                            idx2 = (i1 * p + i2) * H + j1 * p + j2 + W * H * k + Nfilters * W * H * i;
                            bb[idx2] = exp(bb[idx2] + bias[k] - max[k + Nfilters * i]);
                            pvalue[idx + Wres * Hres * k + Nfilters * Wres * Hres * i] += bb[idx2];
                        }
                    
            for (i = 0; i < N; i++)
                for (k = 0; k < Nfilters; k++)
                    for (i2 = 0; i2 < p; i2++)
                        for (j2 = 0; j2 < p; j2++) {
                            idx2 = (i1 * p + i2) * H + j1 * p + j2 + W * H * k + Nfilters * W * H * i;
                            bb[idx2] /= pvalue[idx + Wres * Hres * k + Nfilters * Wres * Hres * i];
                        }
            
            if (nlhs > 1)
                for (i = 0; i < N; i++)
                    for (k = 0; k < Nfilters; k++)
                        pvalue[idx + Wres * Hres * k + Nfilters * Wres * Hres * i] *= exp(max[k + Nfilters * i]);
        }

    mxFree(max);
    
    if (nlhs > 1) {
        for (i = 0; i < Hres * Wres * Nfilters * N; i++)
            pvalue[i] = (pvalue[i] - 1) / pvalue[i];
    } else
        mxFree(pvalue);
        
    if (nlhs > 2) {
        for (i = 0; i < N; i++)
            for (k = 0; k < Nfilters; k++)
                for (i1 = 0; i1 < Wres; i1++)
                    for (j1 = 0; j1 < Hres; j1++) {
                        double rnd = rand() % 10000 / 10000.0, acc = 0;
                        bool done = false;
                        for (i2 = 0; i2 < p; i2++) {
                            for (j2 = 0; j2 < p; j2++) {
                                idx = j2 + p * j1 + H * (i2 + p * i1) + W * H * k + Nfilters * W * H * i;
                                acc += resp[idx];

                                if (acc >= rnd) {
                                    sample[idx] = 1;
                                    done = true;
                                    break;
                                }
                            }
                            if (done) break;
                        }
                    }
    }
}