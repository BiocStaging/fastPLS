// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore

#ifndef FASTPLS_CUDA_RESIDENT_SIMPLS_CUH
#define FASTPLS_CUDA_RESIDENT_SIMPLS_CUH

#include "cuda_resident_preprocess.cuh"
#include "cuda_resident_rsvd.cuh"
#include <memory>
#include <type_traits>

namespace fastpls_device {
constexpr int kResidentSimplsMaximumBlock = 64;
constexpr int kResidentSimplsFloatRegressionBlock = 8;

template<class T> __global__ void add_response_mean(T* pred,size_t size,int n,const T* mean) {
    for(size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;i<size;i+=size_t(blockDim.x)*gridDim.x)
        pred[i]+=mean[i/n];
}

// Isolated fitting core. Public R dispatch is connected only after validation.
template<class T> class ResidentSimpls {
    int n,p,q,components,requested_oversample,requested_power;
    int effective_oversample=0,effective_power=0,refresh_block=1;
    cudaStream_t stream;
    cublasHandle_t blas=nullptr;
    cusolverDnHandle_t solver_handle=nullptr;
    curandGenerator_t rng_handle=nullptr;
    T *X=nullptr,*Y=nullptr,*S=nullptr,*meanX=nullptr,*scaleX=nullptr,*meanY=nullptr,*scaleY=nullptr,
      *R=nullptr,*Q=nullptr,*V=nullptr,*scores=nullptr,*candidate=nullptr,
      *right_gram=nullptr,*predictor_gram=nullptr,*original_crosscov=nullptr,
      *lda_score_gram=nullptr,*lda_class_sums=nullptr,*lda_moment_temp=nullptr;
    int *labels=nullptr,*keys=nullptr,*rows=nullptr,*offsets=nullptr,*invalid=nullptr;
    std::unique_ptr<ComponentWorkspace<T>> update;
    std::unique_ptr<RsvdWorkspace<T>> solver,rank_one_solver;
    bool fitted=false,attempted=false,implicit_crosscov=false,
         cached_predictor_crossprod=false,retain_training_scores=true;
    template<class U> void allocate(U*& ptr,size_t size){require_cuda(cudaMalloc(&ptr,size*sizeof(U)));}
    void product(cublasOperation_t oa,cublasOperation_t ob,int m,int cols,int inner,
                 const T* a,int lda,const T* b,int ldb,T* out,int ldc) {
        T one=1,zero=0;
        require_blas(Decomposition<T>::gemm(blas,oa,ob,m,cols,inner,&one,a,lda,b,ldb,&zero,out,ldc));
    }
    void product_accumulate(cublasOperation_t oa,cublasOperation_t ob,
                            int m,int cols,int inner,const T* a,int lda,
                            const T* b,int ldb,T* out,int ldc,bool add) {
        T one=1,zero=0;
        const T* beta=add?&one:&zero;
        require_blas(Decomposition<T>::gemm(
            blas,oa,ob,m,cols,inner,&one,a,lda,b,ldb,beta,out,ldc));
    }
    void release() noexcept {
        rank_one_solver.reset();solver.reset();update.reset();
        for(T* a:{X,Y,S,meanX,scaleX,meanY,scaleY,R,Q,V,scores,candidate,
                  right_gram,predictor_gram,original_crosscov,lda_score_gram,
                  lda_class_sums,lda_moment_temp})cudaFree(a);
        for(int* a:{labels,keys,rows,offsets,invalid})cudaFree(a);
        if(rng_handle)curandDestroyGenerator(rng_handle);
        if(solver_handle)cusolverDnDestroy(solver_handle);
        if(blas)cublasDestroy(blas);
    }
    void fit_loaded(const T* hostY,const int* hostLabels,int scaling,
                    unsigned long long seed,bool predictors_standardized) {
        require_cuda(cudaMemsetAsync(invalid,0,sizeof(int),stream));
        if(predictors_standardized) {
            require_cuda(initialize_identity_statistics(
                meanX,scaleX,p,stream));
        } else {
            require_cuda(preprocess(X,n,p,scaling,meanX,scaleX,stream));
        }
        if(hostLabels) {
            allocate(S,size_t(p)*q);
            allocate(labels,n);allocate(keys,n);allocate(rows,n);allocate(offsets,q+1);
            require_cuda(cudaMemcpyAsync(labels,hostLabels,n*sizeof(int),cudaMemcpyHostToDevice,stream));
            require_cuda(prepare_labels(labels,n,q,keys,rows,offsets,meanY,invalid,stream));
            require_cuda(class_product(X,n,p,q,rows,offsets,meanY,S,stream));
        } else {
            allocate(Y,size_t(n)*q);
            require_cuda(cudaMemcpyAsync(Y,hostY,size_t(n)*q*sizeof(T),cudaMemcpyHostToDevice,stream));
            require_cuda(preprocess(Y,n,q,1,meanY,scaleY,stream));
            implicit_crosscov = double(p)*double(q)*sizeof(T) >
                512.0*1024.0*1024.0;
            if(!implicit_crosscov) {
                allocate(S,size_t(p)*q);
                product(CUBLAS_OP_T,CUBLAS_OP_N,p,q,n,X,n,Y,n,S,p);
            }
        }
        const double crosscov_bytes=double(p)*double(q)*sizeof(T);
        const bool massive=crosscov_bytes>512.0*1024.0*1024.0;
        cached_predictor_crossprod=hostLabels&&components>=8&&n>=8*p;
        if(cached_predictor_crossprod) {
            allocate(predictor_gram,size_t(p)*p);
            allocate(original_crosscov,size_t(p)*q);
            product(CUBLAS_OP_T,CUBLAS_OP_N,p,p,n,X,n,X,n,predictor_gram,p);
            require_cuda(cudaMemcpyAsync(
                original_crosscov,S,size_t(p)*q*sizeof(T),
                cudaMemcpyDeviceToDevice,stream));
        }
        const bool block_classification=hostLabels&&q<=2048&&components>=4&&
            double(n)*double(p)*double(q)>=5.0e8;
        const bool block_massive_regression=
            std::is_same<T,float>::value&&!hostLabels&&massive&&components>=4;
        if(block_classification) {
            refresh_block=std::min({kResidentSimplsMaximumBlock,components,p,q});
        } else if(block_massive_regression) {
            refresh_block=std::min({kResidentSimplsFloatRegressionBlock,
                                    components,p,q});
        } else {
            refresh_block=1;
        }
        effective_oversample=massive?0:requested_oversample;
        effective_power=massive?1:requested_power;
        solver.reset(new RsvdWorkspace<T>(
            p,q,refresh_block,effective_oversample,effective_power,stream,
            implicit_crosscov?n:0,components,blas,solver_handle,rng_handle));
        if(right_gram)product(
            CUBLAS_OP_T,CUBLAS_OP_N,q,q,p,S,p,S,p,right_gram,q);
        for(int a=0;a<components;) {
            RsvdWorkspace<T>* active_solver=solver.get();
            if(implicit_crosscov) {
                active_solver->solve_implicit_crosscov(
                    X,Y,V,a,seed+a,candidate,invalid);
            } else if(right_gram) {
                active_solver->solve_from_right_gram(
                    S,right_gram,seed+a,candidate,nullptr,nullptr,invalid);
            } else {
                active_solver->solve(S,seed+a,candidate,nullptr,nullptr,invalid);
            }
            const int use=std::min(refresh_block,components-a);
            for(int j=0;j<use;++j) {
                if(cached_predictor_crossprod) {
                    update->step_crossprod(
                        predictor_gram,original_crosscov,S,
                        candidate+size_t(j)*p,a+j,R,V,Q,invalid);
                } else {
                    update->step(X,Y,S,candidate+size_t(j)*p,a+j,R,scores,V,Q,
                                 invalid,rows,offsets,meanY,!implicit_crosscov);
                }
                if(right_gram) {
                    const T minus_one=T(-1);
                    require_blas(Blas<T>::ger(
                        blas,q,q,&minus_one,update->deflation_row(),
                        update->deflation_row(),right_gram));
                }
            }
            a+=use;
        }
        if(cached_predictor_crossprod&&scores) {
            product(CUBLAS_OP_N,CUBLAS_OP_N,n,components,p,
                    X,n,R,p,scores,n);
        }
        int bad=0;require_cuda(cudaMemcpyAsync(&bad,invalid,sizeof(int),cudaMemcpyDeviceToHost,stream));
        require_cuda(cudaStreamSynchronize(stream));
        if(bad)throw std::runtime_error("resident SIMPLS failed: invalid labels, decomposition or component norm");
        fitted=true;
    }
public:
    ResidentSimpls(int n_,int p_,int q_,int a,int oversample,int power,
                   cudaStream_t s,bool retain_scores=true,
                   bool classification=false,int=0,int=0,T=T(0),int=0,
                   T=T(0),bool allocate_predictors=true)
      :n(n_),p(p_),q(q_),components(a),requested_oversample(oversample),
       requested_power(power),stream(s),
       retain_training_scores(retain_scores || !classification ||
                              components<8 || n<8*p ||
                              double(n)*components*sizeof(T) <=
                                  256.0*1024.0*1024.0) {
        if(n<2||p<1||q<1||a<1||a>std::min(n-1,p))throw std::invalid_argument("invalid resident SIMPLS dimensions");
        try {
            require_blas(cublasCreate(&blas));require_blas(cublasSetStream(blas,s));
            configure_blas_math<T>(blas);
            require_solver(cusolverDnCreate(&solver_handle));
            require_solver(cusolverDnSetStream(solver_handle,s));
            require_random(curandCreateGenerator(
                &rng_handle,CURAND_RNG_PSEUDO_DEFAULT));
            require_random(curandSetStream(rng_handle,s));
            if(allocate_predictors)allocate(X,size_t(n)*p);
            allocate(meanX,p);allocate(scaleX,p);allocate(meanY,q);allocate(scaleY,q);
            allocate(R,size_t(p)*a);allocate(Q,size_t(q)*a);allocate(V,size_t(p)*a);
            if(retain_training_scores)allocate(scores,size_t(n)*a);
            allocate(candidate,size_t(p)*std::min({kResidentSimplsMaximumBlock,a,p,q}));
            if(q<=p&&q<=512)allocate(right_gram,size_t(q)*q);
            allocate(invalid,1);
            update.reset(new ComponentWorkspace<T>(n,p,q,a,s));
        } catch(...) {release();throw;}
    }
    ~ResidentSimpls(){release();}
    ResidentSimpls(const ResidentSimpls&)=delete;
    ResidentSimpls& operator=(const ResidentSimpls&)=delete;
    void fit(const T* hostX,const T* hostY,const int* hostLabels,int scaling,unsigned long long seed) {
        if(attempted)throw std::logic_error("resident fitting workspace already used");
        if((hostY==nullptr)==(hostLabels==nullptr))throw std::invalid_argument("provide responses or labels, not both");
        attempted=true;
        require_cuda(cudaMemcpyAsync(X,hostX,size_t(n)*p*sizeof(T),cudaMemcpyHostToDevice,stream));
        fit_loaded(hostY,hostLabels,scaling,seed,false);
    }
    void fit_device_predictors(const T* deviceX,const T* hostY,
                               const int* hostLabels,int scaling,
                               unsigned long long seed,
                               bool predictors_standardized=true) {
        if(attempted)throw std::logic_error("resident fitting workspace already used");
        if(!deviceX||(hostY==nullptr)==(hostLabels==nullptr))
            throw std::invalid_argument("invalid resident device fitting input");
        attempted=true;
        require_cuda(cudaMemcpyAsync(X,deviceX,size_t(n)*p*sizeof(T),
                                     cudaMemcpyDeviceToDevice,stream));
        fit_loaded(hostY,hostLabels,scaling,seed,predictors_standardized);
    }
    void adopt_device_predictors(T*& deviceX,const T* hostY,
                                 const int* hostLabels,int scaling,
                                 unsigned long long seed,
                                 bool predictors_standardized=true) {
        if(attempted)throw std::logic_error("resident fitting workspace already used");
        if(!deviceX||(hostY==nullptr)==(hostLabels==nullptr))
            throw std::invalid_argument("invalid resident device fitting input");
        attempted=true;
        cudaFree(X);
        X=deviceX;
        deviceX=nullptr;
        fit_loaded(hostY,hostLabels,scaling,seed,predictors_standardized);
    }
    void project_device(T* test,int test_n,int prefix,T* test_scores) {
        if(!fitted||test_n<1||prefix<1||prefix>components)throw std::invalid_argument("invalid resident prediction request");
        standardize_device(test,test_n);
        project_standardized_device(test,test_n,prefix,test_scores);
    }
    void standardize_device(T* test,int test_n) {
        if(!fitted||test_n<1)throw std::invalid_argument("invalid resident standardization request");
        standardize<<<256,256,0,stream>>>(test,size_t(test_n)*p,test_n,meanX,scaleX);
        require_cuda(cudaGetLastError());
    }
    void project_standardized_device(const T* test,int test_n,int prefix,T* test_scores) {
        if(!fitted||test_n<1||prefix<1||prefix>components)throw std::invalid_argument("invalid resident projection request");
        product(CUBLAS_OP_N,CUBLAS_OP_N,test_n,prefix,p,test,test_n,R,p,test_scores,test_n);
        require_cuda(cudaGetLastError());
    }
    void predict_increment_device(const T* test_scores,int test_n,int first,
                                  int prefix,T* predictions) {
        if(!fitted||test_n<1||first<0||prefix<=first||prefix>components)
            throw std::invalid_argument("invalid resident prediction increment");
        product_accumulate(
            CUBLAS_OP_N,CUBLAS_OP_T,test_n,q,prefix-first,
            test_scores+size_t(first)*test_n,test_n,
            Q+size_t(first)*q,q,predictions,test_n,first>0);
        if(first==0)
            add_response_mean<<<256,256,0,stream>>>(
                predictions,size_t(test_n)*q,test_n,meanY);
        require_cuda(cudaGetLastError());
    }
    void predict_projected_device(const T* test_scores,int test_n,int prefix,
                                  T* predictions) {
        if(!fitted||test_n<1||prefix<1||prefix>components)
            throw std::invalid_argument("invalid resident projected prediction request");
        product(CUBLAS_OP_N,CUBLAS_OP_T,test_n,q,prefix,test_scores,test_n,
                Q,q,predictions,test_n);
        add_response_mean<<<256,256,0,stream>>>(
            predictions,size_t(test_n)*q,test_n,meanY);
        require_cuda(cudaGetLastError());
    }
    void predict_device(T* test,int test_n,int prefix,T* test_scores,T* predictions) {
        project_device(test,test_n,prefix,test_scores);
        product(CUBLAS_OP_N,CUBLAS_OP_T,test_n,q,prefix,test_scores,test_n,Q,q,predictions,test_n);
        add_response_mean<<<256,256,0,stream>>>(predictions,size_t(test_n)*q,test_n,meanY);
        require_cuda(cudaGetLastError());
    }
    const T* weights()const{return R;}
    const T* loadings()const{return Q;}
    const T* training_scores()const{return scores;}
    const T* standardized_predictors()const{return X;}
    const int* label_rows()const{return rows;}
    const int* label_offsets()const{return offsets;}
    const T* class_priors()const{return meanY;}
    int solver_oversample()const{return effective_oversample;}
    int solver_power()const{return effective_power;}
    int solver_block()const{return refresh_block;}
    int solver_block_limit()const{return 0;}
    bool implicit_operator()const{return implicit_crosscov;}
    bool predictor_crossprod_cache()const{return cached_predictor_crossprod;}
    bool has_lda_moments()const{return cached_predictor_crossprod;}
    void prepare_lda_moments() {
        if(!cached_predictor_crossprod)
            throw std::logic_error("resident SIMPLS LDA moments require cached cross-products");
        if(lda_score_gram)return;
        allocate(lda_score_gram,size_t(components)*components);
        allocate(lda_class_sums,size_t(components)*q);
        allocate(lda_moment_temp,size_t(p)*components);
        product(CUBLAS_OP_N,CUBLAS_OP_N,p,components,p,
                predictor_gram,p,R,p,lda_moment_temp,p);
        product(CUBLAS_OP_T,CUBLAS_OP_N,components,components,p,
                R,p,lda_moment_temp,p,lda_score_gram,components);
        product(CUBLAS_OP_T,CUBLAS_OP_N,components,q,p,
                R,p,original_crosscov,p,lda_class_sums,components);
    }
    const T* lda_gram()const{return lda_score_gram;}
    const T* lda_sums()const{return lda_class_sums;}
    void compact_training() {
        require_cuda(cudaStreamSynchronize(stream));
        solver.reset();update.reset();
        for(T** value:{&X,&Y,&S,&V,&scores,&candidate,&right_gram,
                       &predictor_gram,&original_crosscov}) {
            cudaFree(*value);*value=nullptr;
        }
        for(int** value:{&labels,&keys,&rows,&offsets}) {
            cudaFree(*value);*value=nullptr;
        }
    }
    const T* exported_field(int field)const{
        switch(field){case 0:return R;case 1:return Q;case 2:return scores;
            case 3:return meanX;case 4:return scaleX;case 5:return meanY;}
        throw std::invalid_argument("invalid resident model field");
    }
    int input_columns()const{return p;}
    int feature_columns()const{return p;}
};
} // namespace fastpls_device
#endif
