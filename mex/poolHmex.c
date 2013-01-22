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
    int p, np, nh, i, k, i1, j1, i2, j2, idx, idx2, N, K;
    double *max;

    a = prhs[0];
    b = prhs[1];
    c = prhs[2];
    
    dimsa = mxGetDimensions(a);
        
    N = dimsa[0];
    K = dimsa[1];
    nh = dimsa[2];
    bias = mxGetPr(b);
    aa = mxGetPr(c);
    p = (int)(*aa);
    np = nh / p;
    
    dimso = (mwSize*)mxMalloc(sizeof(mwSize)*4);
    dimso[0] = N; dimso[1] = K; dimso[2] = nh; dimso[3] = nh;
    res = plhs[0] = mxCreateNumericArray(4, dimso, mxDOUBLE_CLASS, mxREAL);
    
    if (nlhs > 1) {
        dimso[0] = N; dimso[1] = K; dimso[2] = np; dimso[3] = np;
        poolres = plhs[1] = mxCreateNumericArray(4, dimso, mxDOUBLE_CLASS, mxREAL);
        pvalue = mxGetPr(poolres);
    } else 
        pvalue = (double*)mxMalloc(sizeof(double) * np * np * N * K);
        
    if (nlhs > 2) {
        dimso[0] = N; dimso[1] = K; dimso[2] = nh; dimso[3] = nh;
        poolsample = plhs[2] = mxCreateNumericArray(4, dimso, mxDOUBLE_CLASS, mxREAL);
        sample = mxGetPr(poolsample);
    }
    
    mxFree(dimso);
    
    aa = mxGetPr(a);
    resp = mxGetPr(res);
    bb = resp;
    
    memcpy(bb, aa, sizeof(double) * nh * nh * N * K);
    max = (double*)mxMalloc(sizeof(double)*K*N);
    
    for (i1 = 0; i1 < np; i1++)
        for (j1 = 0; j1 < np; j1++) {
            idx = i1 * np + j1;
            
            for (k = 0; k < K*N; k++)
                max[k] = -1e10;
            
            for (i2 = 0; i2 < p; i2++)
                for (j2 = 0; j2 < p; j2++)
                    for (k = 0; k < K; k++)
                        for (i = 0; i < N; i++) {
                            double t = bb[i + N * (k + K * ((i1 * p + i2) * nh + j1 * p + j2))] + bias[k];
                            if (t > max[i + N * k])
                                max[i + N * k] = t;
                        }
            
            for (k = 0; k < K; k++)
                for (i = 0; i < N; i++)
                    pvalue[i + N * (k + K * idx)] = exp(-max[i + N * k]);
            
            for (i2 = 0; i2 < p; i2++)
                for (j2 = 0; j2 < p; j2++) 
                    for (k = 0; k < K; k++)
                        for (i = 0; i < N; i++) {
                            idx2 = i + N * (k + K * ((i1 * p + i2) * nh + j1 * p + j2));
                            bb[idx2] = exp(bb[idx2] + bias[k] - max[i + N * k]);
                            pvalue[i + N * (k + K * idx)] += bb[idx2];
                        }
                    
            for (i2 = 0; i2 < p; i2++)
                for (j2 = 0; j2 < p; j2++) 
                    for (k = 0; k < K; k++)
                        for (i = 0; i < N; i++) {
                            idx2 = i + N * (k + K * ((i1 * p + i2) * nh + j1 * p + j2));
                            bb[idx2] /= pvalue[i + N * (k + K * idx)];
                        }
            
            if (nlhs > 1)
                for (k = 0; k < K; k++)
                        for (i = 0; i < N; i++)
                            pvalue[i + N * (k + K * idx)] = pvalue[i + N * (k + K * idx)] * exp(max[i + N * k]);
        }

    mxFree(max);
    
    if (nlhs > 1) {
        for (i = 0; i < np * np * N * K; i++)
            pvalue[i] = (pvalue[i] - 1) / pvalue[i];
    } else
        mxFree(pvalue);
        
    if (nlhs > 2) {
        for (i1 = 0; i1 < np; i1++)
            for (j1 = 0; j1 < np; j1++) 
                for (k = 0; k < K; k++)
                    for (i = 0; i < N; i++) {
                        double rnd = rand() % 10000 / 10000.0, acc = 0;
                        bool done = false;
                        for (i2 = 0; i2 < p; i2++) {
                            for (j2 = 0; j2 < p; j2++) {
                                idx = i + N * (k + K * (j2 + p * j1 + nh * (i2 + p * i1)));
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