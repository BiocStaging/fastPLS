// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore

#ifndef FASTPLS_CUDA_RESIDENT_COMPONENT_CUH
#define FASTPLS_CUDA_RESIDENT_COMPONENT_CUH

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include "cuda_resident_labels.cuh"

namespace fastpls_device {
inline void require_cuda(cudaError_t s) {
    if(s!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
inline void require_blas(cublasStatus_t s) {
    if(s!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("resident SIMPLS cuBLAS operation failed");
}
template<class T> struct Blas;
#define FASTPLS_TYPED_BLAS(T, P) \
template<> struct Blas<T> { \
    static auto gemv(cublasHandle_t h,cublasOperation_t op,int m,int n,const T* a,const T* x,int ld,const T* v,const T* b,T* out) {return cublas##P##gemv(h,op,m,n,a,x,ld,v,1,b,out,1);} \
    static auto norm(cublasHandle_t h,int n,const T* x,T* out) {return cublas##P##nrm2(h,n,x,1,out);} \
    static auto dot(cublasHandle_t h,int n,const T* x,const T* y,T* out) {return cublas##P##dot(h,n,x,1,y,1,out);} \
    static auto scale(cublasHandle_t h,int n,const T* a,T* x) {return cublas##P##scal(h,n,a,x,1);} \
    static auto ger(cublasHandle_t h,int m,int n,const T* a,const T* x,const T* y,T* out) {return cublas##P##ger(h,m,n,a,x,1,y,1,out,m);} \
};
FASTPLS_TYPED_BLAS(float,S)
FASTPLS_TYPED_BLAS(double,D)
#undef FASTPLS_TYPED_BLAS

template<class T> __global__ void component_constants(T* s) {s[0]=1;s[1]=-1;s[2]=0;}
template<class T> __global__ void reciprocal_norm(T* s,int* invalid) {
    if(!(s[3]>T(0)) || !isfinite(s[3])) {atomicExch(invalid,1);s[3]=0;}
    else s[3]=T(1)/s[3];
}
template<class T> __global__ void reciprocal_sqrt_norm(T* s,int* invalid) {
    if(!(s[3]>T(0)) || !isfinite(s[3])) {atomicExch(invalid,1);s[3]=0;}
    else s[3]=rsqrt(s[3]);
}
template<class T> __global__ void correction_gate(T* s,T eps) {
    s[5]=s[4]>T(32)*eps*s[3]?T(-1):T(0);
}

// Owns handles and scratch once; all factors and scalar decisions stay on-device.
template<class T> class ComponentWorkspace {
    cublasHandle_t h=nullptr;
    cudaStream_t stream;
    T *scalars=nullptr,*projection=nullptr,*loading=nullptr,*row=nullptr;
    int n,p,q,capacity;
    bool owns_handle=true;
    void mv(cublasOperation_t op,int m,int k,const T* a,const T* x,
            const T* v,const T* b,T* out) {
        require_blas(Blas<T>::gemv(h,op,m,k,a,x,m,v,b,out));
    }
    void norm(int size,const T* x,T* value) {require_blas(Blas<T>::norm(h,size,x,value));}
    void normalize(int size,T* x,int* invalid) {
        norm(size,x,scalars+3);
        reciprocal_norm<<<1,1,0,stream>>>(scalars,invalid);
        require_blas(Blas<T>::scale(h,size,scalars+3,x));
    }
public:
    ComponentWorkspace(int n_,int p_,int q_,int a,cudaStream_t s,
                       cublasHandle_t shared_handle=nullptr)
      :stream(s),n(n_),p(p_),q(q_),capacity(a),
       owns_handle(shared_handle==nullptr) {
        if(n<1||p<1||q<1||a<1)throw std::invalid_argument("invalid resident component dimensions");
        try {
            if(shared_handle)h=shared_handle;
            else {
                require_blas(cublasCreate(&h));
                require_blas(cublasSetStream(h,s));
                require_blas(cublasSetPointerMode(
                    h,CUBLAS_POINTER_MODE_DEVICE));
            }
            require_cuda(cudaMalloc(&scalars,6*sizeof(T)));
            require_cuda(cudaMalloc(&projection,a*sizeof(T)));
            require_cuda(cudaMalloc(&loading,p*sizeof(T)));
            require_cuda(cudaMalloc(&row,q*sizeof(T)));
            component_constants<<<1,1,0,stream>>>(scalars);
            require_cuda(cudaGetLastError());
        } catch(...) {release();throw;}
    }
    void release() noexcept {
        cudaFree(scalars);cudaFree(projection);cudaFree(loading);cudaFree(row);
        if(h&&owns_handle)cublasDestroy(h);
        scalars=projection=loading=row=nullptr;h=nullptr;
    }
    ~ComponentWorkspace(){release();}
    ComponentWorkspace(const ComponentWorkspace&)=delete;
    ComponentWorkspace& operator=(const ComponentWorkspace&)=delete;

    void step(const T* X,const T* Y,T* S,const T* candidate,int used,
              T* R,T* scores,T* V,T* Q,int* invalid,
              const int* class_rows=nullptr,const int* offsets=nullptr,
              const T* priors=nullptr,bool deflate_operator=true) {
        if(used<0||used>=capacity)throw std::invalid_argument("invalid component prefix");
        if(!Y&&(!class_rows||!offsets||!priors))throw std::invalid_argument("missing response representation");
        T* r=R+size_t(used)*p;T* t=scores+size_t(used)*n;
        T* v=V+size_t(used)*p;T* response=Q+size_t(used)*q;
        require_cuda(cudaMemcpyAsync(r,candidate,p*sizeof(T),cudaMemcpyDeviceToDevice,stream));
        if(used) {
            mv(CUBLAS_OP_T,p,used,scalars,V,r,scalars+2,projection);
            norm(p,r,scalars+3);norm(used,projection,scalars+4);
            correction_gate<<<1,1,0,stream>>>(scalars,std::numeric_limits<T>::epsilon());
            for(int pass=0;pass<2;++pass) {
                mv(CUBLAS_OP_N,p,used,scalars+5,V,projection,scalars,r);
                mv(CUBLAS_OP_T,p,used,scalars,V,r,scalars+2,projection);
            }
        }
        mv(CUBLAS_OP_N,n,p,scalars,X,r,scalars+2,t);
        normalize(n,t,invalid);
        require_blas(Blas<T>::scale(h,p,scalars+3,r));
        mv(CUBLAS_OP_T,n,p,scalars,X,t,scalars+2,loading);
        if(Y)mv(CUBLAS_OP_T,n,q,scalars,Y,t,scalars+2,response);
        else require_cuda(class_product(t,n,1,q,class_rows,offsets,priors,response,stream));
        require_cuda(cudaMemcpyAsync(v,loading,p*sizeof(T),cudaMemcpyDeviceToDevice,stream));
        if(used)for(int pass=0;pass<2;++pass) {
            mv(CUBLAS_OP_T,p,used,scalars,V,v,scalars+2,projection);
            mv(CUBLAS_OP_N,p,used,scalars+1,V,projection,scalars,v);
        }
        normalize(p,v,invalid);
        if(deflate_operator) {
            if(!S)throw std::invalid_argument("missing explicit SIMPLS operator");
            mv(CUBLAS_OP_T,p,q,scalars,S,v,scalars+2,row);
            require_blas(Blas<T>::ger(h,p,q,scalars+1,v,row,S));
        }
        require_cuda(cudaGetLastError());
    }
    void step_crossprod(const T* XtX,const T* Sxy,T* S,
                        const T* candidate,int used,T* R,T* V,T* Q,
                        int* invalid) {
        if(used<0||used>=capacity||!XtX||!Sxy||!S)
            throw std::invalid_argument("invalid cached-cross-product component request");
        T* r=R+size_t(used)*p;
        T* v=V+size_t(used)*p;
        T* response=Q+size_t(used)*q;
        require_cuda(cudaMemcpyAsync(r,candidate,p*sizeof(T),
                                     cudaMemcpyDeviceToDevice,stream));
        if(used) {
            mv(CUBLAS_OP_T,p,used,scalars,V,r,scalars+2,projection);
            norm(p,r,scalars+3);norm(used,projection,scalars+4);
            correction_gate<<<1,1,0,stream>>>(
                scalars,std::numeric_limits<T>::epsilon());
            for(int pass=0;pass<2;++pass) {
                mv(CUBLAS_OP_N,p,used,scalars+5,V,projection,scalars,r);
                mv(CUBLAS_OP_T,p,used,scalars,V,r,scalars+2,projection);
            }
        }
        mv(CUBLAS_OP_N,p,p,scalars,XtX,r,scalars+2,loading);
        require_blas(Blas<T>::dot(h,p,r,loading,scalars+3));
        reciprocal_sqrt_norm<<<1,1,0,stream>>>(scalars,invalid);
        require_blas(Blas<T>::scale(h,p,scalars+3,r));
        require_blas(Blas<T>::scale(h,p,scalars+3,loading));
        mv(CUBLAS_OP_T,p,q,scalars,Sxy,r,scalars+2,response);
        require_cuda(cudaMemcpyAsync(v,loading,p*sizeof(T),
                                     cudaMemcpyDeviceToDevice,stream));
        if(used)for(int pass=0;pass<2;++pass) {
            mv(CUBLAS_OP_T,p,used,scalars,V,v,scalars+2,projection);
            mv(CUBLAS_OP_N,p,used,scalars+1,V,projection,scalars,v);
        }
        normalize(p,v,invalid);
        mv(CUBLAS_OP_T,p,q,scalars,S,v,scalars+2,row);
        require_blas(Blas<T>::ger(h,p,q,scalars+1,v,row,S));
        require_cuda(cudaGetLastError());
    }
    const T* deflation_row() const{return row;}
};
} // namespace fastpls_device
#endif
