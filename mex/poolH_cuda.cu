#include <math.h>
#include <matrix.h>
#include <mex.h>
#include <string.h>
#include <curand.h>

__global__ void pool(float *h, float *pool, float *bias, int np, int nh, int p, int N, int K) {
    int i1 = blockIdx.y, j1 = blockIdx.x, i = threadIdx.x, k = threadIdx.y, i2, j2;
    float max = -1e10, biask = bias[k];
    
    int idx = i1 * np + j1, idx2;
    
    for (i2 = 0; i2 < p; i2++)
        for (j2 = 0; j2 < p; j2++) {
            float t = h[i + N * (k + K * ((i1 * p + i2) * nh + j1 * p + j2))] + biask;
            if (t > max)
                max = t;
        }
        
    pool[i + N * (k + K * idx)] = exp(-max);
    
    for (i2 = 0; i2 < p; i2++)
        for (j2 = 0; j2 < p; j2++) {
            int idx2 = i + N * (k + K * ((i1 * p + i2) * nh + j1 * p + j2));
            h[idx2] = exp(h[idx2] + biask - max);
            pool[i + N * (k + K * idx)] += h[idx2];
        }
        
    for (i2 = 0; i2 < p; i2++)
        for (j2 = 0; j2 < p; j2++) {
            idx2 = i + N * (k + K * ((i1 * p + i2) * nh + j1 * p + j2));
            h[idx2] /= pool[i + N * (k + K * idx)];
        }
        
    pool[i + N * (k + K * idx)] = pool[i + N * (k + K * idx)] * exp(max);
}

__global__ void sampleHidden(float *h, float *sample, float *rand, int np, int nh, int p, int N, int K) {
    int i1 = blockIdx.y, j1 = blockIdx.x, i = threadIdx.x, k = threadIdx.y, i2, j2, idx;
    float rnd = rand[i + N * k + N * K * (i1 * np + j1)], acc = 0;
    bool done = false;
    
	for (i2 = 0; i2 < p; i2++) 
        for (j2 = 0; j2 < p; j2++) {
            idx = i + N * (k + K * (j2 + p * j1 + nh * (i2 + p * i1)));
			sample[idx] = 0;
		}
	
    for (i2 = 0; i2 < p; i2++) {
        for (j2 = 0; j2 < p; j2++) {
            idx = i + N * (k + K * (j2 + p * j1 + nh * (i2 + p * i1)));
            acc += h[idx];

            if (acc > rnd) {
                sample[idx] = 1;
                done = true;
                break;
            }
        }
        if (done) break;
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const mxArray *a, *b, *c;
    mxArray *res, *poolres, *poolsample;
    float *fres, *fpool, *fsample, *fbias, *dres, *dpool, *dsample, *drnd, *frnd, *dbias;
    const mwSize *dimsa;
    mwSize *dimso;
    double *aa, *bb, *resp, *pvalue, *bias, *sample;
    int p, np, nh, i, N, K;
    dim3 grid, block;

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
    
    cudaMallocHost(&fbias, sizeof(float) * K);
    cudaMalloc(&dbias, sizeof(float) * K);
    
    dimso = (mwSize*)mxMalloc(sizeof(mwSize)*4);
    dimso[0] = N; dimso[1] = K; dimso[2] = nh; dimso[3] = nh;
    res = plhs[0] = mxCreateNumericArray(4, dimso, mxDOUBLE_CLASS, mxREAL);
    
    if (nlhs > 1) {
        dimso[0] = N; dimso[1] = K; dimso[2] = np; dimso[3] = np;
        poolres = plhs[1] = mxCreateNumericArray(4, dimso, mxDOUBLE_CLASS, mxREAL);
        pvalue = mxGetPr(poolres);
    } else {
        pvalue = (double*)mxMalloc(sizeof(double) * np * np * N * K);
    }
    
    cudaMallocHost(&fres, sizeof(float) * N * K * nh * nh);
    cudaMallocHost(&fpool, sizeof(float) * N * K * np * np);
    cudaMalloc(&dres, sizeof(float) * N * K * nh * nh);
    cudaMalloc(&dpool, sizeof(float) * N * K * np * np);
        
    if (nlhs > 2) {
        dimso[0] = N; dimso[1] = K; dimso[2] = nh; dimso[3] = nh;
        poolsample = plhs[2] = mxCreateNumericArray(4, dimso, mxDOUBLE_CLASS, mxREAL);
        sample = mxGetPr(poolsample);
        cudaMallocHost(&fsample, sizeof(float) * N * K * nh * nh);
        cudaMalloc(&dsample, sizeof(float) * N * K * nh * nh);
        cudaMalloc(&drnd, sizeof(float) * N * K * np * np);
    }
    
    mxFree(dimso);
    
    aa = mxGetPr(a);
    resp = mxGetPr(res);
    bb = resp;
    
    for (i = 0; i < nh * nh * N * K; i++) fres[i] = (float)aa[i];
    for (i = 0; i < K; i++) fbias[i] = (float)bias[i];
    cudaMemcpy(dres, fres, sizeof(float) * nh * nh * N * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dbias, fbias, sizeof(float) * K, cudaMemcpyHostToDevice);
    
    grid = dim3(np, np, 1);
    block = dim3(N, K, 1);
    
    pool<<<grid, block>>>(dres, dpool, dbias, np, nh, p, N, K);
    
    cudaMemcpy(fres, dres, sizeof(float) * nh * nh * N * K, cudaMemcpyDeviceToHost);
    for (i = 0; i < nh * nh * N * K; i++) bb[i] = (double)fres[i];
       
    if (nlhs > 1) {
        cudaMemcpy(fpool, dpool, sizeof(float) * np * np * N * K, cudaMemcpyDeviceToHost);
        for (i = 0; i < np * np * N * K; i++) {
            pvalue[i] = (double)fpool[i];
            pvalue[i] = (pvalue[i] - 1) / pvalue[i];
        }
    } else
        mxFree(pvalue);
        
    if (nlhs > 2) {
        curandGenerator_t gen;        
        curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT);
		unsigned long long seed = rand();
		cudaMalloc(&drnd, sizeof(float) * np * np * N * K );
		cudaMallocHost(&frnd, sizeof(float) * np * np * N * K);
		curandSetPseudoRandomGeneratorSeed(gen, seed);
        curandStatus_t curandRes = 
			curandGenerateUniform(gen, frnd, np * np * N * K);
		cudaMemcpy(drnd, frnd, sizeof(float) * np * np * N * K, cudaMemcpyHostToDevice);

        sampleHidden<<<grid, block>>>(dres, dsample, drnd, np, nh, p, N, K);
        cudaMemcpy(fsample, dsample, sizeof(float) * nh * nh * N * K, cudaMemcpyDeviceToHost);
		// cudaMemcpy(fsample, drnd, sizeof(float) * np * np * N * K, cudaMemcpyDeviceToHost);
        for (i = 0; i < nh * nh * N * K; i++)
            sample[i] = (double)fsample[i];

		curandDestroyGenerator(gen);
		cudaFreeHost(frnd);
        cudaFree(drnd);
        cudaFreeHost(fsample);
        cudaFree(dsample);
    }
    
    cudaFreeHost(fres);
    cudaFreeHost(fpool);
    cudaFreeHost(fbias);
    cudaFree(dres);
    cudaFree(dpool);
    cudaFree(dbias);
}