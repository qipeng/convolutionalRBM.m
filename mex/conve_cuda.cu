//#include <math.h>
#include <matrix.h>
#include <mex.h>

#include "include/utils.cuh"
#include "include/settings.h"

// #include "cutil.h"

__global__ void convolve(float *a, float *b, float *c, int N, int K0, int K, int n, int m, int nz, int gridOffset) {
    int i = blockIdx.y, j = blockIdx.x, ni = gridOffset + threadIdx.z, k0 = threadIdx.x, k = threadIdx.y, ii, jj;
    float res = 0;
    
    if (ni >= N) return;
    
    for (ii = max(0, i - n + 1); ii < min(m, i + 1); ii++)
        for (jj = max(0, j - n + 1); jj < min(m, j + 1); jj++) 
            res += a[ni + N * k + N * K * ((i - ii) * n + j - jj)] * b[k0 + K0 * k + K0 * K * (ii * m + jj)];
                     
    c[ni + N * k0 +  N * K0 * k + N * K0 * K * ((i) * nz + j)] = res;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const mxArray *a, *b;
    mxArray *c;
    const mwSize *dimsa, *dimsb;
    mwSize *dimsc;
    double *aa, *bb, *cc;
    float *fa, *fb, *fc, *da, *db, *dc;
    int n, m, i, nz, ndima, ndimb, K, K0, N;
    cudaStream_t stream[2];
    dim3 threads, blocks;
    int grids, nPerGrid;

    a = prhs[0];
    b = prhs[1];
    
    ndima = mxGetNumberOfDimensions(a);
    ndimb = mxGetNumberOfDimensions(b);
    
    dimsa = mxGetDimensions(a);
    dimsb = mxGetDimensions(b);
    
    N = dimsa[0]; K0 = dimsb[0]; K = dimsb[1];
    
    if (ndima <= 2) 
        n = 1;
    else
        n = dimsa[2];
    if (ndimb <= 2)
        m = 1;
    else
        m = dimsb[2];
    nz = n + m - 1;
    dimsc = (mwSize*)mxMalloc(sizeof(mwSize)*5);
    dimsc[0] = N; dimsc[1] = K0; dimsc[2] = K; dimsc[3] = nz; dimsc[4] = nz;
    c = plhs[0] = mxCreateNumericArray(5, dimsc, mxDOUBLE_CLASS, mxREAL);
    mxFree(dimsc);
    
    aa = mxGetPr(a);
    bb = mxGetPr(b);
    cc = mxGetPr(c);
    
    cudaSetDevice(DEVICE);
    cudaSetDeviceFlags(cudaDeviceMapHost);
    
    cudaMallocHost(&fa, sizeof(float) * N * K * n * n);
    cudaMallocHost(&fb, sizeof(float) * K0 * K * m * m);
    cudaMallocHost(&fc, sizeof(float) * N * K0 * K * nz * nz);
    
    for (i = 0; i < N * K * n * n; i++) fa[i] = (float)aa[i];
    for (i = 0; i < K0 * K * m * m; i++) fb[i] = (float)bb[i];
    for (i = 0; i < N * K0 * K * nz * nz; i++) fc[i] = i;
    
    nPerGrid = min(BLOCKSIZE / K0 / K, MAXBLOCKD3);
    grids = (N - 1) / nPerGrid + 1;
    blocks = dim3(nz, nz, 1);
    
    cudaMalloc(&db, sizeof(float) * K0 * K * m * m);
    
    cudaMemcpy(db, fb, sizeof(float) * K0 * K * m * m, cudaMemcpyHostToDevice);
    
    if (grids > 1) {
        cudaStreamCreate(&stream[0]);
        cudaStreamCreate(&stream[1]);
    
        threads = dim3(K0, K, nPerGrid);
        cudaMalloc(&da, sizeof(float) * nPerGrid * K * n * n * 2);
        cudaMalloc(&dc, sizeof(float) * nPerGrid * K0 * K * nz * nz * 2);
        cudaMemcpy(da, fa, sizeof(float) * nPerGrid * K * n * n, cudaMemcpyHostToDevice);
        
        for (i = 0; i < grids; i++) {
            int stm_cur = i % 2, stm_next = 1 - i % 2;
            convolve<<<blocks, threads, 0, stream[stm_cur]>>>(da, db, dc, N - (i - stm_cur) * nPerGrid, K0, K, n, m, nz, stm_cur * nPerGrid);
            if (i < grids - 1)
                cudaMemcpyAsync(da + stm_next * nPerGrid * K * n * n, fa + (i + 1) * nPerGrid * K * n * n, sizeof(float) * nPerGrid * K * n * n, cudaMemcpyHostToDevice, stream[stm_next]);
            
            cudaMemcpyAsync(fc + i * nPerGrid * K0 * K * nz * nz, dc + stm_cur * nPerGrid * K0 * K * nz * nz, sizeof(float) * nPerGrid * K0 * K * nz * nz, cudaMemcpyDeviceToHost, stream[stm_cur]);
            cudaDeviceSynchronize();
        }
    } else {
        threads = dim3(K0, K, N);
        cudaMalloc(&da, sizeof(float) * N * K * n * n);
        cudaMalloc(&dc, sizeof(float) * N * K0 * K * nz * nz);
        cudaMemcpy(da, fa, sizeof(float) * N * K * n * n, cudaMemcpyHostToDevice);
        
        convolve<<<blocks, threads>>>(da, db, dc, N, K0, K, n, m, nz, 0);
        cudaMemcpy(fc, dc, sizeof(float) * N * K0 * K * nz * nz, cudaMemcpyDeviceToHost);
    }
    
    for (i = 0; i < N * K0 * K * nz * nz; i++)
        cc[i] = (double)fc[i];
        
    cudaFreeHost(fa);
    cudaFreeHost(fb);
    cudaFreeHost(fc);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
}
