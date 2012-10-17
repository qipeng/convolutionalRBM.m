#include <math.h>
#include <matrix.h>
#include <mex.h>
#include <string.h>

__global__ void arrayExp(float* a, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < N) {
        a[i] = exp(a[i]);
    }
}

__global__ void mf(float *aa, float*bb, float* pvalue, int p, int np, int nh, bool notLast) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x, j, k, jj, kk, idx2;
    
    if (idx < np * np) {
        j = idx / np; k = idx % np;
        pvalue[idx] = 1;
                
        for (jj = 0; jj < p; jj++)
            for (kk = 0; kk < p; kk++)
                pvalue[idx] += bb[(j * p + jj) * nh + k * p + kk];
                
        for (jj = 0; jj < p; jj++)
            for (kk = 0; kk < p; kk++) {
                idx2 = (j * p + jj) * nh + k * p + kk;
                bb[idx2] /= pvalue[idx];
                if (notLast) {
                    bb[idx2] += aa[idx2];
                    bb[idx2] /= 2;
                }
            }
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const mxArray *a, *b, *c;
    mxArray *res, *poolres;
    const mwSize *dimsa;
    double *aa, *resp, *poolresp;
    float *bb, *pvalue;
    float *d_aa, *d_bb, *d_pvalue;
    int p, np, nh, iter, i;//, j, jj, k, kk, idx, idx2;
    int threadsPerBlock = 256, blocksPerGrid1, blocksPerGrid2;

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
    bb = new float[nh * nh];
    pvalue = new float[np * np];
    
    aa = mxGetPr(a);
    resp = mxGetPr(res);
    if (nlhs > 1) poolresp = mxGetPr(poolres);
    
    for (i = 0; i < nh * nh; i++)
        bb[i] = (float)aa[i];
        
    blocksPerGrid1 = (nh * nh + threadsPerBlock - 1) / threadsPerBlock;
    blocksPerGrid2 = (np * np + threadsPerBlock - 1) / threadsPerBlock;
    
    cudaMalloc(&d_aa, sizeof(float)*nh*nh);
    cudaMalloc(&d_bb, sizeof(float)*nh*nh);
    cudaMalloc(&d_pvalue, sizeof(float)*np*np);
    
    cudaMemcpy(d_aa, bb, sizeof(float)*nh*nh, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bb, d_aa, sizeof(float)*nh*nh, cudaMemcpyDeviceToDevice);
    
    
    for (i = 0; i < iter; i++) {
        // for (j = 0; j < nh * nh; j++)
            // bb[j] = exp(bb[j]);
        arrayExp<<<threadsPerBlock, blocksPerGrid1>>>(d_bb, nh*nh);
            
        // for (j = 0; j < np; j++)
            // for (k = 0; k < np; k++) {
                // idx = j * np + k;
                // pvalue[idx] = 1;
                
                // for (jj = 0; jj < p; jj++)
                    // for (kk = 0; kk < p; kk++)
                        // pvalue[idx] += bb[(j * p + jj) * nh + k * p + kk];
                        
                // for (jj = 0; jj < p; jj++)
                    // for (kk = 0; kk < p; kk++) {
                        // idx2 = (j * p + jj) * nh + k * p + kk;
                        // bb[idx2] /= pvalue[idx];
                        // if (i < iter - 1) {
                            // bb[idx2] += aa[idx2];
                            // bb[idx2] /= 2;
                        // }
                    // }
            // }
        mf<<<threadsPerBlock, blocksPerGrid2>>>(d_aa, d_bb, d_pvalue, p, np, nh, (i < iter - 1));
    }
    
    cudaMemcpy(bb, d_bb, sizeof(float)*nh*nh, cudaMemcpyDeviceToHost);
    for (i = 0; i < nh * nh; i++)
        resp[i] = (double)bb[i];
        
    if (nlhs > 1) {
        cudaMemcpy(pvalue, d_pvalue, sizeof(float)*np*np, cudaMemcpyDeviceToHost);
        for (i = 0; i < np * np; i++)
            poolresp[i] = (double)(1 / pvalue[i]);
    }
    
    cudaFree(d_aa); cudaFree(d_bb); cudaFree(d_pvalue);
    
    delete bb;
    delete pvalue;
}