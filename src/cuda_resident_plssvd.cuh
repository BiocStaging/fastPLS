#ifndef FASTPLS_CUDA_RESIDENT_PLSSVD_CUH
#define FASTPLS_CUDA_RESIDENT_PLSSVD_CUH
#include "cuda_resident_simpls.cuh"

namespace fastpls_device {
template<class T> __global__ void plssvd_right_vectors(
    const T* cross,int components,int responses,T* right,int* invalid) {
    const int component=blockIdx.x;
    __shared__ T partial[256];
    T sum=0;
    for(int response=threadIdx.x;response<responses;response+=blockDim.x) {
        const T value=cross[size_t(response)*components+component];
        sum+=value*value;
    }
    partial[threadIdx.x]=sum;
    __syncthreads();
    for(int step=128;step;step/=2) {
        if(threadIdx.x<step)partial[threadIdx.x]+=partial[threadIdx.x+step];
        __syncthreads();
    }
    const T norm=sqrt(partial[0]);
    if(!(norm>T(0))||!isfinite(norm)) {
        if(threadIdx.x==0)atomicExch(invalid,1);
        return;
    }
    for(int response=threadIdx.x;response<responses;response+=blockDim.x)
        right[size_t(component)*responses+response]=
            cross[size_t(response)*components+component]/norm;
}

template<class T> struct Cholesky;
#define FASTPLS_DEVICE_CHOLESKY(T,P) \
template<> struct Cholesky<T> { \
 static auto size(cusolverDnHandle_t h,int n,T* a,int* s){return cusolverDn##P##potrf_bufferSize(h,CUBLAS_FILL_MODE_LOWER,n,a,n,s);} \
 static auto factor(cusolverDnHandle_t h,int n,T* a,T* w,int size,int* info){return cusolverDn##P##potrf(h,CUBLAS_FILL_MODE_LOWER,n,a,n,w,size,info);} \
 static auto solve(cusolverDnHandle_t h,int n,int rhs,T* a,T* b,int* info){return cusolverDn##P##potrs(h,CUBLAS_FILL_MODE_LOWER,n,rhs,a,n,b,n,info);} \
};
FASTPLS_DEVICE_CHOLESKY(float,S)
FASTPLS_DEVICE_CHOLESKY(double,D)
#undef FASTPLS_DEVICE_CHOLESKY

template<class T> class ResidentPlssvd {
    int n,p,q,a,oversample_value,power_value,work_size=0,cached_prefix=0;
    cudaStream_t stream;
    cublasHandle_t blas=nullptr;cusolverDnHandle_t chol=nullptr;
    std::unique_ptr<RsvdWorkspace<T>> solver;
    T *X=nullptr,*Y=nullptr,*S=nullptr,*R=nullptr,*Q=nullptr,*scores=nullptr,*gram=nullptr,*cross=nullptr,
      *factor=nullptr,*weights=nullptr,*work=nullptr,*meanX=nullptr,*scaleX=nullptr,*meanY=nullptr,*scaleY=nullptr;
    T *svd_gram=nullptr,*predictor_gram=nullptr,*moment_temp=nullptr;
    int *labels=nullptr,*keys=nullptr,*rows=nullptr,*offsets=nullptr,*invalid=nullptr,*info=nullptr;
    bool attempted=false,fitted=false,implicit_crosscov=false,
         classification=false,retain_training_scores=true;
    template<class U> void allocate(U*& v,size_t size){require_cuda(cudaMalloc(&v,size*sizeof(U)));}
    void multiply(cublasOperation_t oa,cublasOperation_t ob,int m,int cols,int k,const T* x,int lx,const T* y,int ly,T* out,int ld){
        const T one=1,zero=0;
        require_blas(Decomposition<T>::gemm(blas,oa,ob,m,cols,k,&one,x,lx,y,ly,&zero,out,ld));
    }
    void check_status(){
        int bad=0;require_cuda(cudaMemcpyAsync(&bad,invalid,sizeof(int),cudaMemcpyDeviceToHost,stream));
        require_cuda(cudaStreamSynchronize(stream));
        if(bad)throw std::runtime_error("resident PLS-SVD decomposition or score-Gram solve failed");
    }
    void release()noexcept{
        solver.reset();
        for(T* x:{X,Y,S,R,Q,scores,gram,cross,factor,weights,work,meanX,
                  scaleX,meanY,scaleY,svd_gram,predictor_gram,
                  moment_temp})cudaFree(x);
        for(int* x:{labels,keys,rows,offsets,invalid,info})cudaFree(x);
        if(chol)cusolverDnDestroy(chol);if(blas)cublasDestroy(blas);
    }
public:
    ResidentPlssvd(int n_,int p_,int q_,int components,int oversample,
                   int power,cudaStream_t s,bool retain_scores=true,
                   bool classification_=false,int=0,int=0,T=T(0),int=0,
                   T=T(0))
      :n(n_),p(p_),q(q_),a(components),oversample_value(oversample),
       power_value(power),stream(s),classification(classification_),
       retain_training_scores(retain_scores || !classification_ || n<8*p ||
                              2*components<p ||
                              double(n)*components*sizeof(T) <=
                                  256.0*1024.0*1024.0){
        if(n<2||p<1||q<1||a<1||a>std::min(n-1,std::min(p,q)))throw std::invalid_argument("invalid resident PLS-SVD dimensions");
        try{
            require_blas(cublasCreate(&blas));require_blas(cublasSetStream(blas,s));
            configure_blas_math<T>(blas);
            require_solver(cusolverDnCreate(&chol));require_solver(cusolverDnSetStream(chol,s));
            allocate(X,size_t(n)*p);allocate(R,size_t(p)*a);
            if(retain_training_scores)allocate(scores,size_t(n)*a);
            allocate(Q,size_t(q)*a);
            allocate(gram,size_t(a)*a);allocate(cross,size_t(a)*q);allocate(factor,size_t(a)*a);allocate(weights,size_t(a)*q);
            allocate(meanX,p);allocate(scaleX,p);allocate(meanY,q);allocate(scaleY,q);allocate(invalid,1);allocate(info,1);
            // cuSOLVER work requirements are queried for every eligible prefix.
            for(int k=1;k<=a;++k){int size;require_solver(Cholesky<T>::size(chol,k,factor,&size));work_size=std::max(work_size,size);}
            allocate(work,work_size);
        }catch(...){release();throw;}
    }
    ~ResidentPlssvd(){release();}
    ResidentPlssvd(const ResidentPlssvd&)=delete;
    ResidentPlssvd& operator=(const ResidentPlssvd&)=delete;
    void fit(const T* hx,const T* hy,const int* hl,int scaling,unsigned long long seed){
        if(attempted)throw std::logic_error("resident PLS-SVD workspace already used");
        if((hy==nullptr)==(hl==nullptr))throw std::invalid_argument("provide responses or labels, not both");
        if(hl&&a>=q)throw std::invalid_argument("classification PLS-SVD components must be below class count");
        attempted=true;require_cuda(cudaMemsetAsync(invalid,0,sizeof(int),stream));
        require_cuda(cudaMemcpyAsync(X,hx,size_t(n)*p*sizeof(T),cudaMemcpyHostToDevice,stream));
        require_cuda(preprocess(X,n,p,scaling,meanX,scaleX,stream));
        if(hl){
            allocate(S,size_t(p)*q);
            allocate(labels,n);allocate(keys,n);allocate(rows,n);allocate(offsets,q+1);
            require_cuda(cudaMemcpyAsync(labels,hl,n*sizeof(int),cudaMemcpyHostToDevice,stream));
            require_cuda(prepare_labels(labels,n,q,keys,rows,offsets,meanY,invalid,stream));
            require_cuda(class_product(X,n,p,q,rows,offsets,meanY,S,stream));
        }else{
            allocate(Y,size_t(n)*q);require_cuda(cudaMemcpyAsync(Y,hy,size_t(n)*q*sizeof(T),cudaMemcpyHostToDevice,stream));
            require_cuda(preprocess(Y,n,q,1,meanY,scaleY,stream));
            implicit_crosscov=double(p)*double(q)*sizeof(T)>
                512.0*1024.0*1024.0;
            if(!implicit_crosscov) {
                allocate(S,size_t(p)*q);
                multiply(CUBLAS_OP_T,CUBLAS_OP_N,p,q,n,X,n,Y,n,S,p);
            }
        }
        solver.reset(new RsvdWorkspace<T>(
            p,q,a,oversample_value,power_value,stream,
            implicit_crosscov?n:0,implicit_crosscov?1:0));
        if(implicit_crosscov) {
            solver->solve_implicit_crosscov(
                X,Y,nullptr,0,seed,R,invalid);
        } else if(q<=p&&q<=512) {
            allocate(svd_gram,size_t(q)*q);
            multiply(CUBLAS_OP_T,CUBLAS_OP_N,q,q,p,S,p,S,p,svd_gram,q);
            solver->solve_from_right_gram(
                S,svd_gram,seed,R,Q,nullptr,invalid);
        } else {
            solver->solve(S,seed,R,Q,nullptr,invalid);
        }
        if(scores) {
            multiply(CUBLAS_OP_N,CUBLAS_OP_N,n,a,p,X,n,R,p,scores,n);
            multiply(CUBLAS_OP_T,CUBLAS_OP_N,a,a,n,scores,n,scores,n,gram,a);
        } else {
            allocate(predictor_gram,size_t(p)*p);
            allocate(moment_temp,size_t(p)*a);
            multiply(CUBLAS_OP_T,CUBLAS_OP_N,p,p,n,X,n,X,n,predictor_gram,p);
            multiply(CUBLAS_OP_N,CUBLAS_OP_N,p,a,p,predictor_gram,p,R,p,
                     moment_temp,p);
            multiply(CUBLAS_OP_T,CUBLAS_OP_N,a,a,p,R,p,moment_temp,p,gram,a);
        }
        if(implicit_crosscov) {
            multiply(CUBLAS_OP_T,CUBLAS_OP_N,a,q,n,scores,n,Y,n,cross,a);
            plssvd_right_vectors<<<a,256,0,stream>>>(
                cross,a,q,Q,invalid);
            require_cuda(cudaGetLastError());
        } else {
            multiply(CUBLAS_OP_T,CUBLAS_OP_N,a,q,p,R,p,S,p,cross,a);
        }
        check_status();fitted=true;
    }
    void predict_device(T* test,int count,int prefix,T* test_scores,T* pred){
        if(!fitted||count<1||prefix<1||prefix>a)throw std::invalid_argument("invalid resident PLS-SVD prediction request");
        prepare_prediction_weights(prefix);
        project_device(test,count,prefix,test_scores);
        multiply(CUBLAS_OP_N,CUBLAS_OP_N,count,q,prefix,test_scores,count,weights,prefix,pred,count);
        add_response_mean<<<256,256,0,stream>>>(pred,size_t(count)*q,count,meanY);
        require_cuda(cudaGetLastError());
    }
    void prepare_prediction_weights(int prefix) {
        if(!fitted||prefix<1||prefix>a)
            throw std::invalid_argument("invalid resident PLS-SVD prediction prefix");
        if(prefix!=cached_prefix){
            require_cuda(cudaMemsetAsync(invalid,0,sizeof(int),stream));
            require_cuda(cudaMemcpy2DAsync(factor,prefix*sizeof(T),gram,a*sizeof(T),prefix*sizeof(T),prefix,cudaMemcpyDeviceToDevice,stream));
            require_cuda(cudaMemcpy2DAsync(weights,prefix*sizeof(T),cross,a*sizeof(T),prefix*sizeof(T),q,cudaMemcpyDeviceToDevice,stream));
            require_solver(Cholesky<T>::factor(chol,prefix,factor,work,work_size,info));
            collect_solver_status<<<1,1,0,stream>>>(info,invalid);
            require_solver(Cholesky<T>::solve(chol,prefix,q,factor,weights,info));
            collect_solver_status<<<1,1,0,stream>>>(info,invalid);
            check_status();cached_prefix=prefix;
        }
    }
    const T* weights_matrix()const{return R;}
    void project_device(T* test,int count,int prefix,T* test_scores){
        if(!fitted||count<1||prefix<1||prefix>a)throw std::invalid_argument("invalid resident PLS-SVD projection request");
        standardize_device(test,count);
        project_standardized_device(test,count,prefix,test_scores);
    }
    void standardize_device(T* test,int count) {
        if(!fitted||count<1)throw std::invalid_argument("invalid resident PLS-SVD standardization request");
        standardize<<<256,256,0,stream>>>(test,size_t(count)*p,count,meanX,scaleX);
        require_cuda(cudaGetLastError());
    }
    void project_standardized_device(const T* test,int count,int prefix,T* test_scores){
        if(!fitted||count<1||prefix<1||prefix>a)throw std::invalid_argument("invalid resident PLS-SVD projection request");
        multiply(CUBLAS_OP_N,CUBLAS_OP_N,count,prefix,p,test,count,R,p,test_scores,count);
        require_cuda(cudaGetLastError());
    }
    void predict_increment_device(const T* test_scores,int count,int first,
                                  int prefix,T* pred) {
        (void)first;
        prepare_prediction_weights(prefix);
        multiply(CUBLAS_OP_N,CUBLAS_OP_N,count,q,prefix,test_scores,count,
                 weights,prefix,pred,count);
        add_response_mean<<<256,256,0,stream>>>(
            pred,size_t(count)*q,count,meanY);
        require_cuda(cudaGetLastError());
    }
    void predict_projected_device(const T* test_scores,int count,int prefix,
                                  T* pred) {
        predict_increment_device(test_scores,count,0,prefix,pred);
    }
    const T* training_scores()const{return scores;}
    const T* standardized_predictors()const{return X;}
    const int* label_rows()const{return rows;}
    const int* label_offsets()const{return offsets;}
    const T* class_priors()const{return meanY;}
    int solver_oversample()const{return oversample_value;}
    int solver_power()const{return power_value;}
    int solver_block()const{return a;}
    int solver_block_limit()const{return 0;}
    bool implicit_operator()const{return implicit_crosscov;}
    bool predictor_crossprod_cache()const{return false;}
    bool has_lda_moments()const{return classification;}
    void prepare_lda_moments() {}
    const T* lda_gram()const{return gram;}
    const T* lda_sums()const{return cross;}
    void compact_training() {
        require_cuda(cudaStreamSynchronize(stream));
        solver.reset();
        for(T** value:{&X,&Y,&S,&scores,&svd_gram,&predictor_gram,
                       &moment_temp}) {
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
