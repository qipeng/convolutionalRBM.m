#include <math.h>
#include <matrix.h>
#include <mex.h>

#include <time.h>

#include "include/utils.cuh"
#include "include/settings.h"

__global__ void convolve(float *aa, float *bb, float *cc, int N, int K0, int K, int n, int m, int nz, int gridOffset) {
    int i = blockIdx.y, j = blockIdx.x, ni = gridOffset + threadIdx.z, k0 = threadIdx.x, k = threadIdx.y, ii, jj;
    float res = 0;
    
    if (ni >= N) return;
    
    for (ii = 0; ii < m; ii++)
        for (jj = 0; jj < m; jj++)
            // for (k = 0; k < K; k++)
                res += aa[ni + N * k0 + N * K0 * ((i + ii) * n + j + jj)] * bb[ni + N * k + N * K * (ii * m + jj)];
                
    cc[ni + N * k0 + N * K0 * k + N * K0 * K * (i * nz + j)] = res;
}

__global__ void sum(float *a, float *b, int N) {
    int idx = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));
    float res = 0;
    
    for (int i = 0; i < N; i++)
        res += b[idx * N + i];
        
    a[idx] = res;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const mxArray *a, *b;
    mxArray *c;
    const mwSize *dimsa, *dimsb;
    int ndima, ndimb;
    mwSize *dimsc;
    double *aa, *bb, *cc;
    float *fa, *fb, *fc, *da, *db, *dc, *dd;
    int n, m, i, nz, K, K0, N;
    dim3 blocks, threads;
    int grids, nPerGrid;
    cudaStream_t stream[2];
    
    //long t0 = clock(), t1, t2, t3;

    a = prhs[0];
    b = prhs[1];
    
    dimsa = mxGetDimensions(a);
    dimsb = mxGetDimensions(b);
    
    ndima = mxGetNumberOfDimensions(a);
    ndimb = mxGetNumberOfDimensions(b);
    
    N = dimsa[0]; K0 = dimsa[1]; K = dimsb[1];
    
    if (ndima <= 2) 
        n = 1;
    else
        n = dimsa[2];
    if (ndimb <= 2)
        m = 1;
    else
        m = dimsb[2];
    nz = n - m + 1;
    dimsc = (mwSize*)mxMalloc(sizeof(mwSize)*4);
    //dimsc[0] = N; dimsc[1] = K0; dimsc[2] = K; dimsc[3] = nz; dimsc[4] = nz;
    dimsc[0] = K0; dimsc[1] = K; dimsc[2] = nz; dimsc[3] = nz;
    c = plhs[0] = mxCreateNumericArray(4, dimsc, mxDOUBLE_CLASS, mxREAL);
    mxFree(dimsc);
    
    aa = mxGetPr(a);
    bb = mxGetPr(b);
    cc = mxGetPr(c);
    
    cudaSetDevice(DEVICE);
    cudaSetDeviceFlags(cudaDeviceMapHost);
    
    cudaMallocHost(&fa, sizeof(float) * N * K0 * n * n);
    cudaMallocHost(&fb, sizeof(float) * N * K * m * m);
    cudaMallocHost(&fc, sizeof(float) * K0 * K * nz * nz);
    
    for (i = 0; i < N * K0 * n * n; i++) fa[i] = (float)aa[i];
    for (i = 0; i < N * K * m * m; i++) fb[i] = (float)bb[i];
    for (i = 0; i < K0 * K * nz * nz; i++) fc[i] = i;
    
    nPerGrid = min(BLOCKSIZE / K0 / K, MAXBLOCKD3);
    grids = (N - 1) / nPerGrid + 1;
    blocks = dim3(nz, nz, 1);
    
    //t1 = clock();
    
    if (grids > 1) {
        cudaStreamCreate(&stream[0]);
        cudaStreamCreate(&stream[1]);
    
        threads = dim3(K0, K, nPerGrid);
        cudaMalloc(&da, sizeof(float) * nPerGrid * K0 * n * n * 2);
        cudaMalloc(&db, sizeof(float) * nPerGrid * K * m * m * 2);
        cudaMalloc(&dc, sizeof(float) * nPerGrid * K0 * K * nz * nz * 2);
        cudaMalloc(&dd, sizeof(float) * K0 * K * nz * nz * 2);
        cudaMemcpy(da, fa, sizeof(float) * nPerGrid * K0 * n * n, cudaMemcpyHostToDevice);
        cudaMemcpy(db, fb, sizeof(float) * nPerGrid * K * m * m, cudaMemcpyHostToDevice);
        
        for (i = 0; i < grids; i++) {
            int stm_cur = i % 2, stm_next = 1 - i % 2;
            convolve<<<blocks, threads, 0, stream[stm_cur]>>>(da, db, dc, N - (i - stm_cur) * nPerGrid, K0, K, n, m, nz, stm_cur * nPerGrid);
            if (i < grids - 1) {
                cudaMemcpyAsync(da + stm_next * nPerGrid * K0 * n * n, fa + (i + 1) * nPerGrid * K0 * n * n, sizeof(float) * nPerGrid * K0 * n * n, cudaMemcpyHostToDevice, stream[stm_next]);
                cudaMemcpyAsync(db + stm_next * nPerGrid * K * m * m, fb + (i + 1) * nPerGrid * K * m * m, sizeof(float) * nPerGrid * K * m * m, cudaMemcpyHostToDevice, stream[stm_next]);
            }
            
            //@ Todo: add sum code here
            
            cudaMemcpyAsync(fc + i * nPerGrid * K0 * K * nz * nz, dc + stm_cur * nPerGrid * K0 * K * nz * nz, sizeof(float) * nPerGrid * K0 * K * nz * nz, cudaMemcpyDeviceToHost, stream[stm_cur]);
            cudaDeviceSynchronize();
        }
    } else {
        threads = dim3(K0, K, N);
        cudaMalloc(&da, sizeof(float) * N * K0 * n * n);
        cudaMalloc(&db, sizeof(float) * N * K * m * m);
        cudaMalloc(&dc, sizeof(float) * N * K0 * K * nz * nz);
        cudaMalloc(&dd, sizeof(float) * K0 * K * nz * nz);
        cudaMemcpy(da, fa, sizeof(float) * N * K0 * n * n, cudaMemcpyHostToDevice);
        cudaMemcpy(db, fb, sizeof(float) * N * K * m * m, cudaMemcpyHostToDevice);
        
        convolve<<<blocks, threads>>>(da, db, dc, N, K0, K, n, m, nz, 0);
        
        threads = dim3(K0, K, 1);
        sum<<<blocks, threads>>>(dd, dc, N);
        
        cudaMemcpy(fc, dd, sizeof(float) * K0 * K * nz * nz, cudaMemcpyDeviceToHost);
    }
    
    //t2 = clock();
    
    for (i = 0; i < K0 * K * nz * nz; i++)
        cc[i] = (double)fc[i];
        
    cudaFreeHost(fa);
    cudaFreeHost(fb);
    cudaFreeHost(fc);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    //t3 = clock();
    
    //printf("initialize:%d\nkernel:%d\nfinalize:%d\n", t1-t0, t2-t1, t3-t2);
    
    // for (i = 0; i < nz; i++)
        // for (j = 0; j < nz; j++)
            // for (ii = 0; ii < m; ii++)
                // for (jj = 0; jj < m; jj++)
                    // for (k = 0; k < K; k++) 
                        // for (k0 = 0; k0 < K0; k0++)
                            // for (ni = 0; ni < N; ni++)
                                // cc[ni + N * k0 + N * K0 * k + N * K0 * K * (i * nz + j)] += aa[ni + N * k0 + N * K0 * ((i + ii) * n + j + jj )] * bb[k0 + K0 * k + K0 * K * (ii * m + jj)];
}
