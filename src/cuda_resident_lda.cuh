// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore

#ifndef FASTPLS_CUDA_RESIDENT_LDA_CUH
#define FASTPLS_CUDA_RESIDENT_LDA_CUH
#include "cuda_resident_plssvd.cuh"

namespace fastpls_device {
template<class T> __global__ void lda_means(T* sums,T* weighted,int a,int classes,const int* offsets) {
    for(size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;i<size_t(a)*classes;i+=size_t(blockDim.x)*gridDim.x){
        int c=i/a;T count=T(offsets[c+1]-offsets[c]);
        sums[i]/=count;weighted[i]=sums[i]*sqrt(count);
    }
}
template<class T> __global__ void lda_scale_covariance(T* cov,int a,int denominator) {
    for(size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;i<size_t(a)*a;i+=size_t(blockDim.x)*gridDim.x)cov[i]/=T(denominator);
}
template<class T> __global__ void lda_regularize(const T* cov,int a,int k,T rho,T* factor,T* ridge) {
    __shared__ T diagonal[256];
    T sum=0;for(int i=threadIdx.x;i<k;i+=blockDim.x)sum+=cov[size_t(i)*a+i];
    diagonal[threadIdx.x]=sum;__syncthreads();
    for(int s=128;s;s/=2){if(threadIdx.x<s)diagonal[threadIdx.x]+=diagonal[threadIdx.x+s];__syncthreads();}
    T scale=diagonal[0]/T(k);if(!(scale>T(0))||!isfinite(scale))scale=T(1);
    T lambda=rho*scale;if(threadIdx.x==0)*ridge=lambda;
    for(size_t i=threadIdx.x;i<size_t(k)*k;i+=blockDim.x){
        int row=i%k,col=i/k;factor[i]=cov[size_t(col)*a+row]+(row==col?lambda:T(0));
    }
}
template<class T> __global__ void lda_constants(const T* means,int a,const T* linear,int k,int classes,const T* prior,T* constants){
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=classes)return;
    T dot=0;for(int i=0;i<k;++i)dot+=means[size_t(c)*a+i]*linear[size_t(c)*k+i];
    constants[c]=T(-.5)*dot+log(prior[c]);
}
template<class T> __global__ void lda_decode(T* scores,int n,int classes,const T* constants,int* predicted){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    int best=0;T value=0;
    for(int c=0;c<classes;++c){T s=scores[size_t(c)*n+i]+constants[c];scores[size_t(c)*n+i]=s;if(c==0||s>value){value=s;best=c;}}
    if(predicted)predicted[i]=best+1;
}

template<class T> class ResidentLda {
    int a,c,n,work_size=0,cached=0;
    cudaStream_t stream;cublasHandle_t blas=nullptr;cusolverDnHandle_t solver=nullptr;
    T *means=nullptr,*weighted=nullptr,*cov=nullptr,*factor=nullptr,*linear=nullptr,*constants=nullptr,*work=nullptr,*ridge=nullptr;
    const T* priors;int* info=nullptr;
    template<class U> void allocate(U*& v,size_t size){require_cuda(cudaMalloc(&v,size*sizeof(U)));}
    void release()noexcept{
        for(T* p:{means,weighted,cov,factor,linear,constants,work,ridge})cudaFree(p);cudaFree(info);
        if(solver)cusolverDnDestroy(solver);if(blas)cublasDestroy(blas);
    }
public:
    ResidentLda(const T* scores,int n_,int a_,int classes,const int* rows,const int* offsets,const T* priors_,cudaStream_t s)
      :a(a_),c(classes),n(n_),stream(s),priors(priors_){
        if(n<2||a<1||c<2||c>n||!rows||!offsets||!priors)throw std::invalid_argument("invalid resident LDA input");
        try{
            require_blas(cublasCreate(&blas));require_blas(cublasSetStream(blas,s));
            require_solver(cusolverDnCreate(&solver));require_solver(cusolverDnSetStream(solver,s));
            allocate(means,size_t(a)*c);allocate(weighted,size_t(a)*c);allocate(cov,size_t(a)*a);
            allocate(factor,size_t(a)*a);allocate(linear,size_t(a)*c);allocate(constants,c);allocate(ridge,1);allocate(info,1);
            for(int k=1;k<=a;++k){int size;require_solver(Cholesky<T>::size(solver,k,factor,&size));work_size=std::max(work_size,size);}allocate(work,work_size);
            class_sums<<<dim3(c,a),256,0,stream>>>(scores,n,a,rows,offsets,means);
            lda_means<<<256,256,0,stream>>>(means,weighted,a,c,offsets);
            T one=1,zero=0,minus=-1;
            require_blas(Decomposition<T>::gemm(blas,CUBLAS_OP_T,CUBLAS_OP_N,a,a,n,&one,scores,n,scores,n,&zero,cov,a));
            require_blas(Decomposition<T>::gemm(blas,CUBLAS_OP_N,CUBLAS_OP_T,a,a,c,&minus,weighted,a,weighted,a,&one,cov,a));
            lda_scale_covariance<<<256,256,0,stream>>>(cov,a,std::max(1,n-c));
            require_cuda(cudaGetLastError());
        }catch(...){release();throw;}
    }
    ResidentLda(const T* score_gram,const T* class_sums,int n_,int a_,
                int classes,const int* offsets,const T* priors_,
                cudaStream_t s)
      :a(a_),c(classes),n(n_),stream(s),priors(priors_){
        if(n<2||a<1||c<2||c>n||!score_gram||!class_sums||!offsets||!priors)
            throw std::invalid_argument("invalid resident LDA moment input");
        try{
            require_blas(cublasCreate(&blas));require_blas(cublasSetStream(blas,s));
            require_solver(cusolverDnCreate(&solver));require_solver(cusolverDnSetStream(solver,s));
            allocate(means,size_t(a)*c);allocate(weighted,size_t(a)*c);allocate(cov,size_t(a)*a);
            allocate(factor,size_t(a)*a);allocate(linear,size_t(a)*c);allocate(constants,c);allocate(ridge,1);allocate(info,1);
            for(int k=1;k<=a;++k){int size;require_solver(Cholesky<T>::size(solver,k,factor,&size));work_size=std::max(work_size,size);}allocate(work,work_size);
            require_cuda(cudaMemcpyAsync(means,class_sums,size_t(a)*c*sizeof(T),
                                         cudaMemcpyDeviceToDevice,stream));
            lda_means<<<256,256,0,stream>>>(means,weighted,a,c,offsets);
            require_cuda(cudaMemcpyAsync(cov,score_gram,size_t(a)*a*sizeof(T),
                                         cudaMemcpyDeviceToDevice,stream));
            T one=1,minus=-1;
            require_blas(Decomposition<T>::gemm(blas,CUBLAS_OP_N,CUBLAS_OP_T,
                a,a,c,&minus,weighted,a,weighted,a,&one,cov,a));
            lda_scale_covariance<<<256,256,0,stream>>>(cov,a,std::max(1,n-c));
            require_cuda(cudaGetLastError());
        }catch(...){release();throw;}
    }
    ~ResidentLda(){release();}
    ResidentLda(const ResidentLda&)=delete;ResidentLda& operator=(const ResidentLda&)=delete;
    void prepare(int k){
        if(k<1||k>a)throw std::invalid_argument("invalid LDA prefix");if(k==cached)return;
        bool success=false;
        for(double rho:{1e-8,1e-6,1e-5,1e-4,1e-3,1e-2}){
            lda_regularize<<<1,256,0,stream>>>(cov,a,k,T(rho),factor,ridge);
            require_solver(Cholesky<T>::factor(solver,k,factor,work,work_size,info));
            int status;require_cuda(cudaMemcpyAsync(&status,info,sizeof(int),cudaMemcpyDeviceToHost,stream));require_cuda(cudaStreamSynchronize(stream));
            if(status<0)throw std::runtime_error("invalid resident LDA Cholesky argument");
            if(status==0){success=true;break;}
        }
        if(!success)throw std::runtime_error("resident LDA Cholesky failed for every regularization value");
        require_cuda(cudaMemcpy2DAsync(linear,k*sizeof(T),means,a*sizeof(T),k*sizeof(T),c,cudaMemcpyDeviceToDevice,stream));
        require_solver(Cholesky<T>::solve(solver,k,c,factor,linear,info));
        int status;require_cuda(cudaMemcpyAsync(&status,info,sizeof(int),cudaMemcpyDeviceToHost,stream));require_cuda(cudaStreamSynchronize(stream));
        if(status)throw std::runtime_error("resident LDA triangular solve failed");
        lda_constants<<<(c+255)/256,256,0,stream>>>(means,a,linear,k,c,priors,constants);
        require_cuda(cudaGetLastError());cached=k;
    }
    void predict(const T* scores,int rows,int k,T* out,int* labels=nullptr){
        if(rows<1)throw std::invalid_argument("invalid LDA test row count");prepare(k);
        T one=1,zero=0;
        require_blas(Decomposition<T>::gemm(blas,CUBLAS_OP_N,CUBLAS_OP_N,rows,c,k,&one,scores,rows,linear,k,&zero,out,rows));
        lda_decode<<<(rows+255)/256,256,0,stream>>>(out,rows,c,constants,labels);require_cuda(cudaGetLastError());
    }
    const T* ridge_value()const{return ridge;}
};
} // namespace fastpls_device
#endif
