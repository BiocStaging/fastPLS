#ifndef FASTPLS_CUDA_RESIDENT_VARIANCE_CUH
#define FASTPLS_CUDA_RESIDENT_VARIANCE_CUH

#include "cuda_resident_rsvd.cuh"

namespace fastpls_device {
template<class T> __global__ void variance_loadings(const T* cross,const T* gram,
        int p,int a,T* loadings,T* sequential) {
    for(int f=blockIdx.x*blockDim.x+threadIdx.x;f<p;f+=blockDim.x*gridDim.x) {
        for(int j=0;j<a;++j) {
            const T denom=gram[j+size_t(j)*a];
            T value=cross[f+size_t(j)*p];
            loadings[f+size_t(j)*p]=(isfinite(denom)&&denom>T(0))?value/denom:T(0);
            // X_residual' t_j = X' t_j - sum_i p_i (t_i' t_j).
            for(int i=0;i<j;++i)
                value-=sequential[f+size_t(i)*p]*gram[i+size_t(j)*a];
            sequential[f+size_t(j)*p]=(isfinite(denom)&&denom>T(0))?value/denom:T(0);
        }
    }
}

template<class T> __global__ void variance_column_ss(const T* x,int n,int p,T* ss) {
    __shared__ T work[256];
    for(int col=blockIdx.x;col<p;col+=gridDim.x) {
        T sum=0;
        for(int row=threadIdx.x;row<n;row+=blockDim.x) {
            T v=x[row+size_t(col)*n];sum+=v*v;
        }
        work[threadIdx.x]=sum;__syncthreads();
        for(int stride=128;stride;stride/=2) {
            if(threadIdx.x<stride)work[threadIdx.x]+=work[threadIdx.x+stride];
            __syncthreads();
        }
        if(threadIdx.x==0)ss[col]=work[0];
        __syncthreads();
    }
}

template<class T> __global__ void variance_finish(const T* column_ss,const T* gains,
        const T* gram,int p,int a,T* result) {
    __shared__ T work[256];
    T sum=0;
    for(int f=threadIdx.x;f<p;f+=blockDim.x)sum+=column_ss[f];
    work[threadIdx.x]=sum;__syncthreads();
    for(int stride=128;stride;stride/=2) {
        if(threadIdx.x<stride)work[threadIdx.x]+=work[threadIdx.x+stride];
        __syncthreads();
    }
    if(threadIdx.x==0)result[a]=work[0];
    for(int j=threadIdx.x;j<a;j+=blockDim.x) {
        T gain=gains[j]*gram[j+size_t(j)*a];
        result[j]=(isfinite(gain)&&gain>T(0))?gain:T(0);
    }
}

// No n-by-p residual copy: only p-by-a loadings and a-by-a score products.
template<class T> class VarianceWorkspace {
    int n,p,a;
    cudaStream_t stream;
    cublasHandle_t blas=nullptr;
    T *cross=nullptr,*gram=nullptr,*loadings=nullptr,*sequential=nullptr,
      *column_ss=nullptr,*gains=nullptr,*result=nullptr;
    void alloc(T*& v,size_t count){require_cuda(cudaMalloc(&v,count*sizeof(T)));}
    void release() noexcept {
        cudaFree(cross);cudaFree(gram);cudaFree(loadings);cudaFree(sequential);
        cudaFree(column_ss);cudaFree(gains);cudaFree(result);
        if(blas)cublasDestroy(blas);
    }
public:
    VarianceWorkspace(int rows,int cols,int components,cudaStream_t s):n(rows),p(cols),a(components),stream(s) {
        if(n<1||p<1||a<1)throw std::invalid_argument("invalid resident variance dimensions");
        try {
            require_blas(cublasCreate(&blas));require_blas(cublasSetStream(blas,s));
            alloc(cross,size_t(p)*a);alloc(gram,size_t(a)*a);
            alloc(loadings,size_t(p)*a);alloc(sequential,size_t(p)*a);
            alloc(column_ss,p);alloc(gains,a);alloc(result,a+1);
        }catch(...){release();throw;}
    }
    ~VarianceWorkspace(){release();}
    VarianceWorkspace(const VarianceWorkspace&)=delete;
    VarianceWorkspace& operator=(const VarianceWorkspace&)=delete;
    void compute(const T* X,const T* scores) {
        const T one=1,zero=0;
        require_blas(Decomposition<T>::gemm(blas,CUBLAS_OP_T,CUBLAS_OP_N,p,a,n,&one,X,n,scores,n,&zero,cross,p));
        require_blas(Decomposition<T>::gemm(blas,CUBLAS_OP_T,CUBLAS_OP_N,a,a,n,&one,scores,n,scores,n,&zero,gram,a));
        variance_loadings<<<std::min(65535,1+(p-1)/256),256,0,stream>>>(cross,gram,p,a,loadings,sequential);
        variance_column_ss<<<std::min(p,65535),256,0,stream>>>(X,n,p,column_ss);
        variance_column_ss<<<std::min(a,65535),256,0,stream>>>(sequential,p,a,gains);
        variance_finish<<<1,256,0,stream>>>(column_ss,gains,gram,p,a,result);
        require_cuda(cudaGetLastError());
    }
    const T* predictor_loadings()const{return loadings;}
    // Component sums of squares, followed by total predictor sum of squares.
    const T* sums_of_squares()const{return result;}
};
}
#endif
