#ifndef FASTPLS_CUDA_RESIDENT_METRICS_CUH
#define FASTPLS_CUDA_RESIDENT_METRICS_CUH
#include "cuda_resident_component.cuh"
namespace fastpls_device {
// Three rows per response: SSE, training-mean SST, observed-mean SST.
template<class T> __global__ void response_sums(const T* pred,const T* observed,
        const int* labels,const T* training_mean,int n,int q,T* out,int* invalid) {
    __shared__ T work[3][256];
    for(int col=blockIdx.x;col<q;col+=gridDim.x) {
        T sum=0;
        for(int i=threadIdx.x;i<n;i+=blockDim.x) {
            if(labels&&(labels[i]<1||labels[i]>q))atomicExch(invalid,1);
            T y=labels?T(labels[i]==col+1):observed[i+size_t(col)*n];
            if(!isfinite(y))atomicExch(invalid,1);
            sum+=y;
        }
        work[0][threadIdx.x]=sum;__syncthreads();
        for(int stride=128;stride;stride/=2){
            if(threadIdx.x<stride)work[0][threadIdx.x]+=work[0][threadIdx.x+stride];
            __syncthreads();
        }
        T mean=work[0][0]/T(n),sse=0,train_ss=0,observed_ss=0;
        __syncthreads();
        for(int i=threadIdx.x;i<n;i+=blockDim.x){
            T y=labels?T(labels[i]==col+1):observed[i+size_t(col)*n];
            T prediction=pred[i+size_t(col)*n];
            if(!isfinite(prediction))atomicExch(invalid,1);
            T e=y-prediction,d=y-training_mean[col],v=y-mean;
            sse+=e*e;train_ss+=d*d;observed_ss+=v*v;
        }
        work[0][threadIdx.x]=sse;work[1][threadIdx.x]=train_ss;work[2][threadIdx.x]=observed_ss;
        __syncthreads();
        for(int stride=128;stride;stride/=2){
            if(threadIdx.x<stride)for(int j=0;j<3;++j)work[j][threadIdx.x]+=work[j][threadIdx.x+stride];
            __syncthreads();
        }
        if(threadIdx.x==0)for(int j=0;j<3;++j)out[j+3*size_t(col)]=work[j][0];
        __syncthreads();
    }
}
}
#endif
