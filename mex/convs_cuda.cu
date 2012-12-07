#include <math.h>
#include <matrix.h>
#include <mex.h>
#include "include/utils.cuh"
#include <time.h>

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))
#define size3d(sz, is3d, K) ((is3d)?(sz*K):(sz))
#define GRID_LIMITX 32
#define GRID_LIMITY 16

__global__ void convolve(float* a, float* b, float* c, const int n, const int m, const int nz, const int K0, const int K, const int gridOffsetX, const int gridOffsetY) {
    int ii = blockIdx.x, jj = blockIdx.y, k0 = threadIdx.x + gridOffsetX, k = threadIdx.y + gridOffsetY, i, j;
    
    if (k0 >= K0 || k >= K) return;
    
    for (i = 0; i < m; i++)
        for (j = 0; j < m; j++)
            c[(ii * nz + jj) * K0 * K + (k0 * K + k)] += a[((i + ii) * n + (j + jj)) * K0 + k0] * b[(i * m + j) * K + k];
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const mxArray *a, *b;
    mxArray *c;
    const mwSize *dimsa, *dimsb;
    mwSize *dimsc;
    double *aa, *bb, *cc;
    float *fa = NULL, *fb = NULL, *fc = NULL,
        *da = NULL, *db = NULL, *dc = NULL;//, *dd = NULL;
    int n, m, i, j, nz, ndima, ndimb, K, K0, szc, sza, szb, 
        a3d, b3d, c4d = 1;
    // long t0 = clock(), t1, t2, t3, t4, t5, t6, t7, t8;
    dim3 threadsPerBlock, blocksPerGrid;
    int gridsx, gridsy;

    a = prhs[0];
    b = prhs[1];
    
    dimsa = mxGetDimensions(a);
    dimsb = mxGetDimensions(b);
    
    ndima = mxGetNumberOfDimensions(a);
    ndimb = mxGetNumberOfDimensions(b);
    K0 = 1; K = 1;
    // if (nrhs > 2) {
        // clear = prhs[2];
        // cclear = mxGetPr(clear);
        // c4d = (int)*cclear;
    // }
    
    switch (ndima) {
        case 3: a3d=1;K0=dimsa[2];break;
        default: a3d=0;break;
    }
    
    switch (ndimb) {
        case 3: b3d=1;K=dimsb[2];break;
        default: b3d=0;break;
    }
        
    n = dimsa[0];
    m = dimsb[0];
    nz = n - m + 1;
    dimsc = new mwSize[4];
    if (c4d) {
        dimsc[0] = nz; dimsc[1] = nz; dimsc[2] = K; dimsc[3] = K0;
        c = plhs[0] = mxCreateNumericArray(4, dimsc, mxDOUBLE_CLASS, mxREAL);
    } else {
        c = plhs[0] = mxCreateDoubleMatrix(nz, nz, mxREAL);
    }
    
    szc = nz * nz;
    szb = m * m;
    sza = n * n;
    
    aa = mxGetPr(a);
    bb = mxGetPr(b);
    cc = mxGetPr(c);
    
    cudaSetDeviceFlags(cudaDeviceMapHost);
    fa = new float[size3d(sza, a3d, K0)];
    fb = new float[size3d(szb, b3d, K)];
    //fc = new float[szc*K*K0];
    cudaHostAlloc(&fc, sizeof(float)*size3d(szc, c4d, K*K0), cudaHostAllocPortable);   
    
    for (i = 0; i < K0; i++)
        for (j = 0; j < sza; j++)
            fa[i + j * K0] = (float)aa[i * sza + j];
    for (i = 0; i < K; i++)
        for (j = 0; j < szb; j++)
            fb[i + j * K] = (float)bb[i * szb + j];
        
    // t1 = clock();printf("1 ");
    
    cudaMalloc(&da, sizeof(float)*size3d(sza, a3d, K0));
    cudaMalloc(&db, sizeof(float)*size3d(szb, b3d, K));
    cudaMalloc(&dc, sizeof(float)*size3d(szc, c4d, K0*K));
    //cudaHostGetDevicePointer(&dc, fc, 0);
    // t2 = clock();printf("2 ");
    
    cudaMemcpy(da, fa, sizeof(float)*size3d(sza, a3d, K0), cudaMemcpyHostToDevice);
    cudaMemcpy(db, fb, sizeof(float)*size3d(szb, b3d, K), cudaMemcpyHostToDevice);
    // t3 = clock();printf("3 ");
    
    threadsPerBlock = dim3(min(K0, GRID_LIMITX), min(K, GRID_LIMITY));
    gridsx = (int)ceil(K0 * 1.0 / GRID_LIMITX);
    gridsy = (int)ceil(K * 1.0 / GRID_LIMITY);
    
    blocksPerGrid = dim3(nz, nz);
    
    for (int gx = 0; gx < gridsx; gx++)
        for (int gy = 0; gy < gridsy; gy++)
            convolve<<<blocksPerGrid, threadsPerBlock>>>(da, db, dc, n, m, nz, K0, K, GRID_LIMITX * gx, GRID_LIMITY * gy);
                    
    // t4 = clock();printf("4 ");
    cudaMemcpy(fc, dc, sizeof(float)*size3d(szc, c4d, K*K0), cudaMemcpyDeviceToHost);
    // t5 = clock();printf("5 ");
    
    if (fc) {
        for (i = 0; i < szc; i++)
            for (j = 0; j < K * K0; j++)
                cc[i + j * szc] = (double)fc[i * K * K0 + j];
    }else
        mexErrMsgTxt("Error reading back.");
    // t6 = clock();printf("6 ");
        
    // if (nrhs > 3) {
        // clear = prhs[2];
        // cclear = mxGetPr(clear);
        // if ((int)*cclear == 1) {
            cudaFree(da); cudaFree(db); cudaFree(dc);
            
            delete fa; delete fb; 
            cudaFreeHost(fc);
        // }
    // }
    // t7 = clock();printf("7 ");
    // t8 = clock();printf("8\n");
    // printf("host init:%d\ndevice init:%d\ntransfer h->d:%d\nkernel:%d\n\
// transfer d->h:%d\nhost finalize:%d\ndevice free:%d\nhost free:%d\n",\
    // t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6, t8-t7);
}
