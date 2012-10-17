#include <math.h>
#include <matrix.h>
#include <mex.h>

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

__global__ void convolve(int n, int m, int nz, float* a, float* b, float* c) {
    int i = blockDim.x * blockIdx.x + threadIdx.x, j, ii, jj;

    j = i % nz; i = i / nz;
    
    for (ii = 0; ii < m; ii++)
        for (jj = 0; jj < m; jj++)
            c[i * nz + j] += a[(i + ii) * n + j + jj] * b[ii * m + jj];
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const mxArray *a, *b;
    mxArray *c;
    const mwSize *dimsa, *dimsb;
    double *aa, *bb, *cc;
    int n, m, i, nz;
    float *fa, *fb, *fc, *da, *db, *dc;
    int threadsPerBlock = 256, blocksPerGrid;

    a = prhs[0];
    b = prhs[1];
    
    dimsa = mxGetDimensions(a);
    dimsb = mxGetDimensions(b);
        
    n = dimsa[0];
    m = dimsb[0];
    nz = n - m + 1;
    c = plhs[0] = mxCreateDoubleMatrix(nz, nz, mxREAL);
    
    aa = mxGetPr(a);
    bb = mxGetPr(b);
    cc = mxGetPr(c);
    
    fa = new float[n*n];
    fb = new float[m*m];
    fc = new float[nz*nz];
    
    cudaMalloc(&da, sizeof(float)*n*n);
    cudaMalloc(&db, sizeof(float)*m*m);
    cudaMalloc(&dc, sizeof(float)*nz*nz);
    
    for (i = 0; i < n * n; i++)
        fa[i] = (float)aa[i];
    for (i = 0; i < m * m; i++)
        fb[i] = (float)bb[i];
    
    cudaMemcpy(da, fa, sizeof(float)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(db, fb, sizeof(float)*m*m, cudaMemcpyHostToDevice);
    
    blocksPerGrid = (nz * nz + threadsPerBlock - 1) / threadsPerBlock;
    
    convolve<<<threadsPerBlock, blocksPerGrid>>>(n, m, nz, da, db, dc);
    
    // for (i = 0; i < nz; i++)
        // for (j = 0; j < nz; j++)
            // for (ii = 0; ii < MIN(m, i); ii++)
                // for (jj = 0; jj < MIN(m, j); jj++)
                    // cc[i * nz + j] += aa[(i - ii) * n + j - jj] * bb[ii * m + jj];
                    
    cudaMemcpy(fc, dc, sizeof(float)*nz*nz, cudaMemcpyDeviceToHost);
    
    for (i = 0; i < nz * nz; i++)
        cc[i] = (double)fc[i];
        
    cudaFree(da); cudaFree(db); cudaFree(dc);
    delete fa; delete fb; delete fc;
}
