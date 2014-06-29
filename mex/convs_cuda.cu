#include <math.h>
#include <matrix.h>
#include <mex.h>

#include "include/utils.cuh"
#include "include/settings.h"

__global__ void convs(float *fa, float *fb, float *fc, int grid_images, int grid_filters, 
    int H, int W, int Wfilter, int Hres, int Wres, int colors, int filterBlocks) {
    
    __shared__ float imgdata[BLOCKSIZE*BLOCKSIZE][IMAGES_PER_GRID];
    __shared__ float filtdata[BLOCKSIZE*BLOCKSIZE][FILTERS_PER_GRID];
    __shared__ float res[FILTERS_PER_GRID][IMAGES_PER_GRID];
    
    int tx = threadIdx.x, ty = threadIdx.y, bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
    int resy = bx / filterBlocks, resx = by / filterBlocks;
    int offsetx = (bx % filterBlocks) * BLOCKSIZE, offsety = (by % filterBlocks) * BLOCKSIZE;
    int imy = (resx + offsetx) + tx, imx = (resy + offsety) + ty;
    int fty = offsetx + tx, ftx = offsety + ty;
    int colorbase = bz * COLORS_PER_BLOCK;
    
    if (tx < FILTERS_PER_GRID && ty < IMAGES_PER_GRID)
        res[tx][ty] = 0;
    
    for (int c = colorbase; c < colors; c++) {
        // Load data into shared memory
        for (int im = 0; im < grid_images; im++) {
            if (imy < H && imx < W)
                imgdata[ty + BLOCKSIZE * tx][im] = 
                    fa[imy + H * imx + W * H * c + W * H * colors * im];
            else 
                imgdata[ty + BLOCKSIZE * tx][im] = 0;
        }
        
        for (int ft = 0; ft < grid_filters; ft++) {
            if (fty < Wfilter && ftx < Wfilter)
                filtdata[ty + BLOCKSIZE * tx][ft] = 
                    fb[fty + Wfilter * ftx + Wfilter * Wfilter * c + Wfilter * Wfilter * colors * ft];
            else 
                filtdata[ty + BLOCKSIZE * tx][ft] = 0;
        }
        
        __syncthreads();
        
        // Compute convolution
        if (tx < FILTERS_PER_GRID && ty < IMAGES_PER_GRID)
            for (int t = 0; t < BLOCKSIZE*BLOCKSIZE; t++)
                res[tx][ty] += filtdata[t][tx] * imgdata[t][ty];
                
        __syncthreads();
    }
    
    if (tx < FILTERS_PER_GRID && ty < IMAGES_PER_GRID)
        atomicAdd(&fc[resy + Hres * resx + Wres * Hres * tx + FILTERS_PER_GRID * Wres * Hres * ty], res[tx][ty]);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    float *fa, *fb, *fc, *da, *db, *dc;
    dim3 blocks, threads;
    
    const mxArray *a, *b;
    mxArray *c;
    const mwSize *dimsa, *dimsb;
    mwSize *dimsc;
    double *aa, *bb, *cc;
    int H, W, ndima, ndimb, colors, Nfilters, N, Wfilter, Wres, Hres, SIZE_IMAGE, SIZE_FILTER;
    int imggrids, filtgrids;

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
    
    Wfilter = dimsb[0];
    if (ndimb <= 3) Nfilters = 1;
    else Nfilters = dimsb[3];
    
    Wres = W - Wfilter + 1;
    Hres = H - Wfilter + 1;
   
    dimsc = (mwSize*)mxMalloc(sizeof(mwSize)*4);
    dimsc[0] = Hres; dimsc[1] = Wres; dimsc[2] = Nfilters; dimsc[3] = N;
    c = plhs[0] = mxCreateNumericArray(4, dimsc, mxDOUBLE_CLASS, mxREAL);
    mxFree(dimsc);
    
    SIZE_IMAGE = H * W * colors;
    SIZE_FILTER = Wfilter * Wfilter * colors;
    
    aa = mxGetPr(a);
    bb = mxGetPr(b);
    cc = mxGetPr(c);
    
    cudaSetDevice(DEVICE);
    cudaSetDeviceFlags(cudaDeviceMapHost);
    
    CUDA_SAFE_CALL(cudaMallocHost(&fa, sizeof(float) * SIZE_IMAGE * IMAGES_PER_GRID));
    CUDA_SAFE_CALL(cudaMallocHost(&fb, sizeof(float) * SIZE_FILTER * FILTERS_PER_GRID));
    CUDA_SAFE_CALL(cudaMallocHost(&fc, sizeof(float) * Hres * Wres * FILTERS_PER_GRID * IMAGES_PER_GRID));
    
    CUDA_SAFE_CALL(cudaMalloc(&da, sizeof(float) * SIZE_IMAGE * IMAGES_PER_GRID));
    CUDA_SAFE_CALL(cudaMalloc(&db, sizeof(float) * SIZE_FILTER * FILTERS_PER_GRID));
    CUDA_SAFE_CALL(cudaMalloc(&dc, sizeof(float) * Hres * Wres * FILTERS_PER_GRID * IMAGES_PER_GRID));
    imggrids = (N - 1) / IMAGES_PER_GRID + 1; filtgrids = (Nfilters - 1) / FILTERS_PER_GRID + 1;
    
    for (int ig = 0; ig < imggrids; ig++) {
        int grid_images = min(IMAGES_PER_GRID, N - ig * IMAGES_PER_GRID);
        for (int imgidx = 0; imgidx < grid_images; imgidx++)
            for (int t = 0; t < SIZE_IMAGE; t++) {
                fa[t + SIZE_IMAGE * imgidx] = (float)aa[t + SIZE_IMAGE * (imgidx + ig * IMAGES_PER_GRID)];
            }
        CUDA_SAFE_CALL(cudaMemcpy(da, fa, sizeof(float) * SIZE_IMAGE * grid_images, cudaMemcpyHostToDevice));

        for (int fg = 0; fg < filtgrids; fg++) {
            int filter_div = (Wfilter + BLOCKSIZE - 1) / Wfilter;
            int grid_filters = min(FILTERS_PER_GRID, Nfilters - fg * FILTERS_PER_GRID);
            for (int fx = 0; fx < filter_div; fx++)
                for (int fy = 0; fy < filter_div; fy++) {    
                    memset(fb, 0, sizeof(float) * SIZE_FILTER * FILTERS_PER_GRID);
                    for (int filtidx = 0; filtidx < grid_filters; filtidx++)
                        for (int t = 0; t < SIZE_FILTER; t++)
                            fb[t + SIZE_FILTER * filtidx] = (float)bb[t + SIZE_FILTER * (filtidx + fg * FILTERS_PER_GRID)];
                            
                    CUDA_SAFE_CALL(cudaMemcpy(db, fb, sizeof(float) * SIZE_FILTER * grid_filters, cudaMemcpyHostToDevice));
                    memset(fc, 0, sizeof(float) * Hres * Wres * FILTERS_PER_GRID * IMAGES_PER_GRID);
                    CUDA_SAFE_CALL(cudaMemcpy(dc, fc, sizeof(float) * Hres * Wres * FILTERS_PER_GRID * IMAGES_PER_GRID, cudaMemcpyHostToDevice));
                    int filterBlocks = ((Wfilter - 1) / BLOCKSIZE + 1);
                    blocks = dim3(Hres * filterBlocks, Wres * filterBlocks, (colors - 1) / COLORS_PER_BLOCK + 1);
                    threads = dim3(BLOCKSIZE, BLOCKSIZE, 1);

                    convs<<<blocks, threads>>>(da, db, dc, grid_images, grid_filters, H, W, Wfilter, Hres, Wres, colors, filterBlocks);

                    CUDA_SAFE_CALL(cudaMemcpy(fc, dc, sizeof(float) * Hres * Wres * IMAGES_PER_GRID * FILTERS_PER_GRID, cudaMemcpyDeviceToHost));

                    for (int imgidx = 0; imgidx < grid_images; imgidx++)
                        for (int filtidx = 0; filtidx < grid_filters; filtidx++)
                            for (int xx = 0; xx < Wres; xx++)
                                for (int yy = 0; yy < Hres; yy++)
                                    cc[yy + Hres * xx + Wres * Hres * (filtidx + fg * FILTERS_PER_GRID) + Nfilters * Wres * Hres * (imgidx + ig * IMAGES_PER_GRID)] = (double)fc[yy + Hres * xx + Wres * Hres * filtidx + FILTERS_PER_GRID * Wres * Hres * imgidx];     
                }
        }
    }
        
    cudaFreeHost(fa);
    cudaFreeHost(fb);
    cudaFreeHost(fc);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
}
