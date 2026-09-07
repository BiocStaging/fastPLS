#ifndef FASTPLS_CUDA_RESIDENT_RSVD_CUH
#define FASTPLS_CUDA_RESIDENT_RSVD_CUH

#include "cuda_resident_component.cuh"
#include <cusolverDn.h>
#include <curand.h>
#include <algorithm>
#include <type_traits>

namespace fastpls_device {
inline void require_solver(cusolverStatus_t s) {
    if(s!=CUSOLVER_STATUS_SUCCESS)throw std::runtime_error("resident rSVD cuSOLVER operation failed");
}
inline void require_random(curandStatus_t s) {
    if(s!=CURAND_STATUS_SUCCESS)throw std::runtime_error("resident rSVD cuRAND operation failed");
}
template<class T> inline void configure_blas_math(cublasHandle_t) {}
template<> inline void configure_blas_math<float>(cublasHandle_t handle) {
    require_blas(cublasSetMathMode(handle,CUBLAS_TF32_TENSOR_OP_MATH));
}
template<class T> struct Decomposition;
#define FASTPLS_DEVICE_DECOMPOSITION(T,P,NORMAL) \
template<> struct Decomposition<T> { \
 static auto gemm(cublasHandle_t h,cublasOperation_t a,cublasOperation_t b,int m,int n,int k,const T* alpha,const T* x,int lx,const T* y,int ly,const T* beta,T* z,int lz){return cublas##P##gemm(h,a,b,m,n,k,alpha,x,lx,y,ly,beta,z,lz);} \
 static auto qr_size(cusolverDnHandle_t h,int m,int n,T* x,int* s){return cusolverDn##P##geqrf_bufferSize(h,m,n,x,m,s);} \
 static auto q_size(cusolverDnHandle_t h,int m,int n,T* x,T* tau,int* s){return cusolverDn##P##orgqr_bufferSize(h,m,n,n,x,m,tau,s);} \
 static auto qr(cusolverDnHandle_t h,int m,int n,T* x,T* tau,T* work,int size,int* info){return cusolverDn##P##geqrf(h,m,n,x,m,tau,work,size,info);} \
 static auto form_q(cusolverDnHandle_t h,int m,int n,T* x,T* tau,T* work,int size,int* info){return cusolverDn##P##orgqr(h,m,n,n,x,m,tau,work,size,info);} \
 static auto svd_size(cusolverDnHandle_t h,int m,int n,int* size){return cusolverDn##P##gesvd_bufferSize(h,m,n,size);} \
 static auto svd(cusolverDnHandle_t h,int m,int n,T* x,T* d,T* u,T* vt,T* work,int size,T* rwork,int* info){return cusolverDn##P##gesvd(h,'S','S',m,n,x,m,d,u,m,vt,n,work,size,rwork,info);} \
 static auto random(curandGenerator_t g,T* out,size_t n){return NORMAL(g,out,n,T(0),T(1));} \
};
FASTPLS_DEVICE_DECOMPOSITION(float,S,curandGenerateNormal)
FASTPLS_DEVICE_DECOMPOSITION(double,D,curandGenerateNormalDouble)
#undef FASTPLS_DEVICE_DECOMPOSITION

__global__ void collect_solver_status(const int* info,int* invalid) {
    if(*info)atomicExch(invalid,1);
}

// Matrix buffers remain resident; callers copy only returned factors if required.
template<class T> class RsvdWorkspace {
    int p,q,k,l,power,work_size=0;
    cudaStream_t stream;
    cublasHandle_t blas=nullptr;
    cusolverDnHandle_t solver=nullptr;
    curandGenerator_t rng=nullptr;
    T *omega=nullptr,*basis=nullptr,*right=nullptr,*small=nullptr,*small_u=nullptr,
      *small_vt=nullptr,*singular=nullptr,*tau=nullptr,*work=nullptr,*rwork=nullptr,
      *sample=nullptr,*projection=nullptr;
    int observations=0,max_prefix=0;
    int* info=nullptr;
    bool owns_handles=true;
    void allocate(T*& ptr,size_t size){require_cuda(cudaMalloc(&ptr,size*sizeof(T)));}
    void multiply(cublasOperation_t opA,cublasOperation_t opB,int m,int n,int inner,
                  const T* a,int lda,const T* b,int ldb,T* out,int ldc) {
        const T one=1,zero=0;
        require_blas(Decomposition<T>::gemm(blas,opA,opB,m,n,inner,&one,a,lda,b,ldb,&zero,out,ldc));
    }
    void project_left(T* values,const T* vectors,int used) {
        if(used<=0)return;
        const T one=1,zero=0,minus_one=-1;
        require_blas(Decomposition<T>::gemm(
            blas,CUBLAS_OP_T,CUBLAS_OP_N,used,l,p,&one,
            vectors,p,values,p,&zero,projection,used));
        require_blas(Decomposition<T>::gemm(
            blas,CUBLAS_OP_N,CUBLAS_OP_N,p,l,used,&minus_one,
            vectors,p,projection,used,&one,values,p));
    }
    void implicit_forward(const T* x,const T* y,const T* input,T* output,
                          const T* vectors,int used) {
        multiply(CUBLAS_OP_N,CUBLAS_OP_N,observations,l,q,
                 y,observations,input,q,sample,observations);
        multiply(CUBLAS_OP_T,CUBLAS_OP_N,p,l,observations,
                 x,observations,sample,observations,output,p);
        project_left(output,vectors,used);
    }
    void implicit_transpose(const T* x,const T* y,T* input,T* output,
                            const T* vectors,int used) {
        project_left(input,vectors,used);
        multiply(CUBLAS_OP_N,CUBLAS_OP_N,observations,l,p,
                 x,observations,input,p,sample,observations);
        multiply(CUBLAS_OP_T,CUBLAS_OP_N,q,l,observations,
                 y,observations,sample,observations,output,q);
    }
    void orthonormalize(T* x,int rows,int* invalid) {
        require_solver(Decomposition<T>::qr(solver,rows,l,x,tau,work,work_size,info));
        collect_solver_status<<<1,1,0,stream>>>(info,invalid);
        require_solver(Decomposition<T>::form_q(solver,rows,l,x,tau,work,work_size,info));
        collect_solver_status<<<1,1,0,stream>>>(info,invalid);
    }
public:
    RsvdWorkspace(int rows,int cols,int rank,int oversample,int iterations,
                  cudaStream_t s,int observations_=0,int max_prefix_=0,
                  cublasHandle_t shared_blas=nullptr,
                  cusolverDnHandle_t shared_solver=nullptr,
                  curandGenerator_t shared_rng=nullptr)
      :p(rows),q(cols),k(rank),l(0),power(iterations),stream(s),
       observations(observations_),max_prefix(max_prefix_) {
        if(p<1||q<1||k<1||k>std::min(p,q)||oversample<0||power<0)
            throw std::invalid_argument("invalid resident rSVD dimensions or controls");
        const int shared_count=(shared_blas?1:0)+(shared_solver?1:0)+
            (shared_rng?1:0);
        if(shared_count!=0&&shared_count!=3)
            throw std::invalid_argument("provide all shared CUDA solver handles or none");
        l=k+std::min(oversample,std::min(p,q)-k);
        try {
            if(shared_count==3) {
                blas=shared_blas;solver=shared_solver;rng=shared_rng;
                owns_handles=false;
            } else {
                require_blas(cublasCreate(&blas));
                require_blas(cublasSetStream(blas,s));
                configure_blas_math<T>(blas);
                require_solver(cusolverDnCreate(&solver));
                require_solver(cusolverDnSetStream(solver,s));
                require_random(curandCreateGenerator(
                    &rng,CURAND_RNG_PSEUDO_DEFAULT));
                require_random(curandSetStream(rng,s));
            }
            allocate(omega,(size_t(q)*l+1)/2*2);allocate(basis,size_t(p)*l);
            allocate(right,size_t(q)*l);allocate(small,size_t(q)*l);
            allocate(small_u,size_t(q)*l);allocate(small_vt,size_t(l)*l);
            allocate(singular,l);allocate(tau,l);allocate(rwork,std::max(1,l-1));
            if(observations>0) {
                if(max_prefix<1)throw std::invalid_argument("invalid implicit rSVD prefix capacity");
                allocate(sample,size_t(observations)*l);
                allocate(projection,size_t(max_prefix)*l);
            }
            require_cuda(cudaMalloc(&info,sizeof(int)));
            int size;
            for(int rows_:{p,q}) {
                T* matrix=rows_==p?basis:right;
                require_solver(Decomposition<T>::qr_size(solver,rows_,l,matrix,&size));work_size=std::max(work_size,size);
                require_solver(Decomposition<T>::q_size(solver,rows_,l,matrix,tau,&size));work_size=std::max(work_size,size);
            }
            require_solver(Decomposition<T>::svd_size(solver,q,l,&size));work_size=std::max(work_size,size);
            allocate(work,work_size);
        } catch(...) {release();throw;}
    }
    void release() noexcept {
        for(T* ptr:{omega,basis,right,small,small_u,small_vt,singular,tau,work,rwork,
                    sample,projection})cudaFree(ptr);
        cudaFree(info);
        if(owns_handles) {
            if(rng)curandDestroyGenerator(rng);
            if(solver)cusolverDnDestroy(solver);
            if(blas)cublasDestroy(blas);
        }
        omega=basis=right=small=small_u=small_vt=singular=tau=work=rwork=
            sample=projection=nullptr;
        info=nullptr;rng=nullptr;solver=nullptr;blas=nullptr;
    }
    ~RsvdWorkspace(){release();}
    RsvdWorkspace(const RsvdWorkspace&)=delete;
    RsvdWorkspace& operator=(const RsvdWorkspace&)=delete;
    void solve(const T* matrix,unsigned long long seed,T* u,T* v,T* d,int* invalid) {
        require_random(curandSetPseudoRandomGeneratorSeed(rng,seed));
        require_random(curandSetGeneratorOffset(rng,0));
        require_random(Decomposition<T>::random(rng,omega,(size_t(q)*l+1)/2*2));
        multiply(CUBLAS_OP_N,CUBLAS_OP_N,p,l,q,matrix,p,omega,q,basis,p);
        orthonormalize(basis,p,invalid);
        for(int i=0;i<power;++i) {
            multiply(CUBLAS_OP_T,CUBLAS_OP_N,q,l,p,matrix,p,basis,p,right,q);
            orthonormalize(right,q,invalid);
            multiply(CUBLAS_OP_N,CUBLAS_OP_N,p,l,q,matrix,p,right,q,basis,p);
            orthonormalize(basis,p,invalid);
        }
        if(k==1&&l==1&&!v&&!d) {
            require_cuda(cudaMemcpyAsync(u,basis,p*sizeof(T),
                                         cudaMemcpyDeviceToDevice,stream));
            return;
        }
        // Decompose B' (q x l), ensuring the tall-matrix cuSOLVER contract.
        multiply(CUBLAS_OP_T,CUBLAS_OP_N,q,l,p,matrix,p,basis,p,small,q);
        require_solver(Decomposition<T>::svd(solver,q,l,small,singular,small_u,small_vt,work,work_size,rwork,info));
        collect_solver_status<<<1,1,0,stream>>>(info,invalid);
        multiply(CUBLAS_OP_N,CUBLAS_OP_T,p,k,l,basis,p,small_vt,l,u,p);
        if(v)require_cuda(cudaMemcpyAsync(v,small_u,size_t(q)*k*sizeof(T),cudaMemcpyDeviceToDevice,stream));
        if(d)require_cuda(cudaMemcpyAsync(d,singular,k*sizeof(T),cudaMemcpyDeviceToDevice,stream));
        require_cuda(cudaGetLastError());
    }
    void solve_implicit_crosscov(const T* x,const T* y,const T* vectors,int used,
                                 unsigned long long seed,T* u,int* invalid) {
        if(!sample||!projection||!x||!y||!u||used<0||used>max_prefix)
            throw std::invalid_argument("invalid implicit cross-covariance rSVD request");
        require_random(curandSetPseudoRandomGeneratorSeed(rng,seed));
        require_random(curandSetGeneratorOffset(rng,0));
        require_random(Decomposition<T>::random(rng,omega,(size_t(q)*l+1)/2*2));
        implicit_forward(x,y,omega,basis,vectors,used);
        orthonormalize(basis,p,invalid);
        for(int i=0;i<power;++i) {
            implicit_transpose(x,y,basis,right,vectors,used);
            orthonormalize(right,q,invalid);
            implicit_forward(x,y,right,basis,vectors,used);
            orthonormalize(basis,p,invalid);
        }
        if(k==1&&l==1) {
            require_cuda(cudaMemcpyAsync(u,basis,p*sizeof(T),
                                         cudaMemcpyDeviceToDevice,stream));
            return;
        }
        implicit_transpose(x,y,basis,small,vectors,used);
        require_solver(Decomposition<T>::svd(
            solver,q,l,small,singular,small_u,small_vt,
            work,work_size,rwork,info));
        collect_solver_status<<<1,1,0,stream>>>(info,invalid);
        multiply(CUBLAS_OP_N,CUBLAS_OP_T,p,k,l,basis,p,small_vt,l,u,p);
        require_cuda(cudaGetLastError());
    }
    void solve_from_right_gram(const T* matrix,const T* gram,
                               unsigned long long seed,T* u,T* v,T* d,
                               int* invalid) {
        require_random(curandSetPseudoRandomGeneratorSeed(rng,seed));
        require_random(curandSetGeneratorOffset(rng,0));
        require_random(Decomposition<T>::random(rng,omega,(size_t(q)*l+1)/2*2));
        T* z=omega;
        T* next=right;
        if(power==0)orthonormalize(z,q,invalid);
        for(int i=0;i<power;++i) {
            multiply(CUBLAS_OP_N,CUBLAS_OP_N,q,l,q,gram,q,z,q,next,q);
            orthonormalize(next,q,invalid);
            std::swap(z,next);
        }
        multiply(CUBLAS_OP_N,CUBLAS_OP_N,p,l,q,matrix,p,z,q,basis,p);
        orthonormalize(basis,p,invalid);
        if(k==1&&l==1&&!v&&!d) {
            require_cuda(cudaMemcpyAsync(u,basis,p*sizeof(T),
                                         cudaMemcpyDeviceToDevice,stream));
            return;
        }
        // The right basis is no longer needed, so `small` can hold B' (q x l).
        multiply(CUBLAS_OP_T,CUBLAS_OP_N,q,l,p,matrix,p,basis,p,small,q);
        require_solver(Decomposition<T>::svd(
            solver,q,l,small,singular,small_u,small_vt,
            work,work_size,rwork,info));
        collect_solver_status<<<1,1,0,stream>>>(info,invalid);
        multiply(CUBLAS_OP_N,CUBLAS_OP_T,p,k,l,basis,p,small_vt,l,u,p);
        if(v)require_cuda(cudaMemcpyAsync(
            v,small_u,size_t(q)*k*sizeof(T),cudaMemcpyDeviceToDevice,stream));
        if(d)require_cuda(cudaMemcpyAsync(
            d,singular,k*sizeof(T),cudaMemcpyDeviceToDevice,stream));
        require_cuda(cudaGetLastError());
    }
};
} // namespace fastpls_device
#endif
