#include <Rcpp.h>
#include "cuda_resident_api.h"

namespace {
#ifdef FASTPLS_HAS_CUDA
struct ResidentRModel {
    void* handle;
    int n,p,feature_p,q,a,precision;
    ResidentRModel(void* h,int n_,int p_,int feature_p_,int q_,int a_,int bits)
      :handle(h),n(n_),p(p_),feature_p(feature_p_),q(q_),a(a_),precision(bits){}
    ~ResidentRModel(){fastpls_resident_simpls_destroy(handle);}
};
Rcpp::IntegerVector dimensions(SEXP x,int precision) {
    if(TYPEOF(x)!=(precision==32?INTSXP:REALSXP)||!Rf_isMatrix(x))
        Rcpp::stop("resident input must be a numeric matrix or float32 integer-bit matrix of the requested precision");
    Rcpp::IntegerVector d=Rf_getAttrib(x,R_DimSymbol);
    if(d.size()!=2||d[0]<1||d[1]<1)Rcpp::stop("resident input dimensions must be positive");
    return d;
}
const void* data(SEXP x,int precision){return precision==32?static_cast<const void*>(INTEGER(x)):static_cast<const void*>(REAL(x));}
Rcpp::XPtr<ResidentRModel> resident_state(Rcpp::List object){
    if(!object.containsElementNamed("state"))Rcpp::stop("resident CUDA state is missing");
    SEXP ptr=object["state"];
    if(TYPEOF(ptr)!=EXTPTRSXP||R_ExternalPtrTag(ptr)!=Rf_install("fastPLS_cuda_resident_simpls"))
        Rcpp::stop("invalid resident CUDA model pointer");
    Rcpp::XPtr<ResidentRModel> state(ptr);
    if(!state.get()||!state->handle)Rcpp::stop("resident CUDA state is unavailable; refit the model");
    return state;
}
#endif
}

// [[Rcpp::export]]
SEXP cuda_resident_project_cpp(Rcpp::List object,SEXP X,int ncomp){
#ifdef FASTPLS_HAS_CUDA
    auto state=resident_state(object);auto dx=dimensions(X,state->precision);
    if(dx[1]!=state->p||ncomp<1||ncomp>state->a)Rcpp::stop("invalid resident score dimensions");
    Rcpp::Shield<SEXP> result(Rf_allocMatrix(state->precision==32?INTSXP:REALSXP,dx[0],ncomp));
    void* out=state->precision==32?static_cast<void*>(INTEGER(result)):static_cast<void*>(REAL(result));
    char error[1024];
    if(fastpls_resident_project(state->handle,data(X,state->precision),dx[0],ncomp,out,error,sizeof(error)))Rcpp::stop("%s",error);
    return result;
#else
    Rcpp::stop("CUDA resident projection is unavailable in this build; no CPU fallback is performed");
#endif
}

// [[Rcpp::export]]
SEXP cuda_resident_response_sums_cpp(Rcpp::List object,SEXP X,SEXP Y,SEXP labels,int ncomp){
#ifdef FASTPLS_HAS_CUDA
    auto state=resident_state(object);auto dx=dimensions(X,state->precision);
    if(dx[1]!=state->p)Rcpp::stop("test predictor dimension differs from the fitted model");
    const void* y=nullptr;const int* ylabels=nullptr;
    if(labels!=R_NilValue){
        if(Y!=R_NilValue||TYPEOF(labels)!=INTSXP||Rf_xlength(labels)!=dx[0])Rcpp::stop("invalid observed class labels");
        ylabels=INTEGER(labels);
    }else{
        auto dy=dimensions(Y,state->precision);
        if(dy[0]!=dx[0]||dy[1]!=state->q)Rcpp::stop("observed response dimensions differ from predictions");
        y=data(Y,state->precision);
    }
    Rcpp::Shield<SEXP> result(Rf_allocMatrix(state->precision==32?INTSXP:REALSXP,3,state->q));
    void* out=state->precision==32?static_cast<void*>(INTEGER(result)):static_cast<void*>(REAL(result));
    char error[1024];
    if(fastpls_resident_response_sums(state->handle,data(X,state->precision),y,ylabels,dx[0],ncomp,out,error,sizeof(error)))Rcpp::stop("%s",error);
    return result;
#else
    Rcpp::stop("CUDA resident metrics are unavailable in this build; no CPU fallback is performed");
#endif
}

// [[Rcpp::export]]
Rcpp::List cuda_resident_simpls_fit_cpp(SEXP X,SEXP Y,SEXP labels,int classes,
    int precision,int ncomp,int scaling,int oversample,int power,int seed,
    bool retain_scores=true,int method=3,int north=1,int kernel=2,
    double gamma=1.0,int degree=3,double coef0=1.0) {
#ifdef FASTPLS_HAS_CUDA
    if(precision!=32&&precision!=64)Rcpp::stop("precision must be 32 or 64");
    auto dx=dimensions(X,precision);
    const int* ylabels=nullptr;const void* response=nullptr;int q=classes;
    if(labels!=R_NilValue) {
        if(Y!=R_NilValue||TYPEOF(labels)!=INTSXP||Rf_xlength(labels)!=dx[0]||classes<2)
            Rcpp::stop("provide one integer class label per row and no dense response");
        ylabels=INTEGER(labels);
    } else {
        auto dy=dimensions(Y,precision);q=dy[1];
        if(dy[0]!=dx[0])Rcpp::stop("response and predictor row counts differ");
        response=data(Y,precision);
    }
    char error[1024];
    if(method!=1&&method!=3&&method!=4&&method!=5)
        Rcpp::stop("resident core supports PLS-SVD, SIMPLS, OPLS, or nonlinear kernel PLS");
    void* handle=nullptr;
    if(method==1)handle=fastpls_resident_plssvd_create(data(X,precision),
        response,ylabels,precision,dx[0],dx[1],q,ncomp,scaling,oversample,
        power,retain_scores?1:0,static_cast<unsigned int>(seed),error,
        sizeof(error));
    else if(method==3)handle=fastpls_resident_simpls_create(data(X,precision),
        response,ylabels,precision,dx[0],dx[1],q,ncomp,scaling,oversample,
        power,retain_scores?1:0,static_cast<unsigned int>(seed),error,
        sizeof(error));
    else if(method==4)handle=fastpls_resident_opls_create(data(X,precision),
        response,ylabels,precision,dx[0],dx[1],q,ncomp,scaling,oversample,
        power,retain_scores?1:0,static_cast<unsigned int>(seed),north,error,
        sizeof(error));
    else handle=fastpls_resident_kernelpls_create(data(X,precision),response,
        ylabels,precision,dx[0],dx[1],q,ncomp,scaling,oversample,power,
        retain_scores?1:0,static_cast<unsigned int>(seed),kernel,gamma,degree,
        coef0,error,sizeof(error));
    if(!handle)Rcpp::stop("%s",error);
    int effective_oversample=0,effective_power=0,refresh_block=0;
    int refresh_block_limit=0;
    int implicit_operator=0,predictor_crossprod_cache=0;
    if(fastpls_resident_controls(
        handle,&effective_oversample,&effective_power,&refresh_block,
        &refresh_block_limit,&implicit_operator,&predictor_crossprod_cache,
        error,sizeof(error))){
        fastpls_resident_simpls_destroy(handle);Rcpp::stop("%s",error);
    }
    const int feature_p=method==5?dx[0]:dx[1];
    Rcpp::XPtr<ResidentRModel> state(new ResidentRModel(handle,dx[0],dx[1],
        feature_p,q,ncomp,precision),true,
        Rf_install("fastPLS_cuda_resident_simpls"));
    return Rcpp::List::create(Rcpp::_["state"]=state,Rcpp::_["ncomp"]=ncomp,
        Rcpp::_["precision"]=precision,Rcpp::_["resident"]=true,
        Rcpp::_["effective_oversample"]=effective_oversample,
        Rcpp::_["effective_power"]=effective_power,
        Rcpp::_["refresh_block"]=refresh_block,
        Rcpp::_["refresh_block_limit"]=refresh_block_limit,
        Rcpp::_["implicit_crosscovariance"]=implicit_operator==1,
        Rcpp::_["predictor_crossprod_cache"]=predictor_crossprod_cache==1);
#else
    Rcpp::stop("CUDA resident fitting is unavailable in this build; no CPU fallback is performed");
#endif
}

// [[Rcpp::export]]
Rcpp::List cuda_resident_export_cpp(Rcpp::List object,bool loadings=false,
                                     bool variance=false,bool scores=true) {
#ifdef FASTPLS_HAS_CUDA
    if(!object.containsElementNamed("state"))Rcpp::stop("resident CUDA state is missing");
    SEXP ptr=object["state"];
    if(TYPEOF(ptr)!=EXTPTRSXP||R_ExternalPtrTag(ptr)!=Rf_install("fastPLS_cuda_resident_simpls"))Rcpp::stop("invalid resident CUDA model pointer");
    Rcpp::XPtr<ResidentRModel> state(ptr);
    if(!state.get())Rcpp::stop("resident CUDA state is unavailable; refit the model");
    int rows[]={state->feature_p,state->q,state->n,1,1,1,state->feature_p,1};
    int cols[]={state->a,state->a,state->a,state->p,state->p,state->q,state->a,state->a+1};
    const char* names[]={"R","Q","Ttrain","mX","vX","mY","P","predictor_ss"};
    Rcpp::List result;
    for(int field=0;field<8;++field){
        if((field==2&&!scores)||(field==6&&!loadings)||
           (field==7&&!variance))continue;
        Rcpp::Shield<SEXP> value(Rf_allocMatrix(state->precision==32?INTSXP:REALSXP,rows[field],cols[field]));
        void* out=state->precision==32?static_cast<void*>(INTEGER(value)):static_cast<void*>(REAL(value));
        char error[1024];
        if(fastpls_resident_export(state->handle,field,out,size_t(rows[field])*cols[field],error,sizeof(error)))Rcpp::stop("%s",error);
        result[names[field]]=value;
    }
    return result;
#else
    Rcpp::stop("CUDA resident model export is unavailable in this build");
#endif
}

// [[Rcpp::export]]
void cuda_resident_compact_cpp(Rcpp::List object,bool prepare_lda=false) {
#ifdef FASTPLS_HAS_CUDA
    auto state=resident_state(object);char error[1024];
    if(fastpls_resident_compact(state->handle,prepare_lda?1:0,error,sizeof(error)))
        Rcpp::stop("%s",error);
#else
    Rcpp::stop("CUDA resident compaction is unavailable in this build; no CPU fallback is performed");
#endif
}

// [[Rcpp::export]]
Rcpp::IntegerMatrix cuda_resident_classify_cpp(Rcpp::List object,SEXP X,int ncomp,int classifier,int top) {
#ifdef FASTPLS_HAS_CUDA
    if(!object.containsElementNamed("state"))Rcpp::stop("resident CUDA state is missing");
    SEXP ptr=object["state"];
    if(TYPEOF(ptr)!=EXTPTRSXP||R_ExternalPtrTag(ptr)!=Rf_install("fastPLS_cuda_resident_simpls"))Rcpp::stop("invalid resident CUDA model pointer");
    Rcpp::XPtr<ResidentRModel> state(ptr);
    if(!state.get())Rcpp::stop("resident CUDA state is unavailable; refit the model");
    auto dx=dimensions(X,state->precision);
    if(dx[1]!=state->p||top<1||top>state->q)Rcpp::stop("invalid predictor dimension or top-k request");
    Rcpp::IntegerMatrix result(dx[0],top);char error[1024];
    if(fastpls_resident_classify(state->handle,data(X,state->precision),dx[0],ncomp,classifier,top,result.begin(),error,sizeof(error)))Rcpp::stop("%s",error);
    return result;
#else
    Rcpp::stop("CUDA resident classification is unavailable in this build; no CPU fallback is performed");
#endif
}

// [[Rcpp::export]]
Rcpp::IntegerVector cuda_resident_classify_path_cpp(
    Rcpp::List object,SEXP X,Rcpp::IntegerVector ncomp,int classifier,int top) {
#ifdef FASTPLS_HAS_CUDA
    auto state=resident_state(object);
    auto dx=dimensions(X,state->precision);
    if(dx[1]!=state->p||ncomp.size()<1||top<1||top>state->q)
        Rcpp::stop("invalid predictor dimension, component path, or top-k request");
    int previous=0;
    for(int value:ncomp) {
        if(value<=previous||value>state->a)
            Rcpp::stop("ncomp must be strictly increasing and within the fitted path");
        previous=value;
    }
    Rcpp::IntegerVector result(
        static_cast<R_xlen_t>(dx[0])*top*ncomp.size());
    result.attr("dim")=Rcpp::IntegerVector::create(dx[0],top,ncomp.size());
    char error[1024];
    if(fastpls_resident_classify_path(
        state->handle,data(X,state->precision),dx[0],ncomp.begin(),ncomp.size(),
        classifier,top,result.begin(),error,sizeof(error)))
        Rcpp::stop("%s",error);
    return result;
#else
    Rcpp::stop("CUDA resident classification is unavailable in this build; no CPU fallback is performed");
#endif
}

// [[Rcpp::export]]
Rcpp::List cuda_resident_classify_response_path_cpp(
    Rcpp::List object,SEXP X,Rcpp::IntegerVector ncomp,int classifier,int top) {
#ifdef FASTPLS_HAS_CUDA
    auto state=resident_state(object);auto dx=dimensions(X,state->precision);
    if(dx[1]!=state->p||ncomp.size()<1||top<1||top>state->q)
        Rcpp::stop("invalid predictor dimension, component path, or top-k request");
    int previous=0;
    for(int value:ncomp) {
        if(value<=previous||value>state->a)
            Rcpp::stop("ncomp must be strictly increasing and within the fitted path");
        previous=value;
    }
    if(classifier!=0&&classifier!=1)Rcpp::stop("invalid resident classifier");
    Rcpp::IntegerVector labels(
        static_cast<R_xlen_t>(dx[0])*top*ncomp.size());
    labels.attr("dim")=Rcpp::IntegerVector::create(dx[0],top,ncomp.size());
    Rcpp::Shield<SEXP> predictions(Rf_allocVector(
        state->precision==32?INTSXP:REALSXP,
        static_cast<R_xlen_t>(dx[0])*state->q*ncomp.size()));
    Rf_setAttrib(predictions,R_DimSymbol,Rcpp::IntegerVector::create(
        dx[0],state->q,ncomp.size()));
    void* output=state->precision==32?static_cast<void*>(INTEGER(predictions)):
        static_cast<void*>(REAL(predictions));
    char error[1024];
    if(fastpls_resident_classify_response_path(
        state->handle,data(X,state->precision),dx[0],ncomp.begin(),ncomp.size(),
        classifier,top,labels.begin(),output,error,sizeof(error)))
        Rcpp::stop("%s",error);
    return Rcpp::List::create(
        Rcpp::Named("labels")=labels,
        Rcpp::Named("predictions")=predictions);
#else
    Rcpp::stop("CUDA resident classification-response path is unavailable in this build; no CPU fallback is performed");
#endif
}

// [[Rcpp::export]]
SEXP cuda_resident_simpls_predict_cpp(Rcpp::List object,SEXP X,int ncomp,int classifier=0) {
#ifdef FASTPLS_HAS_CUDA
    if(!object.containsElementNamed("state"))Rcpp::stop("resident CUDA state is missing");
    SEXP ptr=object["state"];
    if(TYPEOF(ptr)!=EXTPTRSXP||R_ExternalPtrTag(ptr)!=Rf_install("fastPLS_cuda_resident_simpls"))
        Rcpp::stop("invalid resident CUDA model pointer");
    Rcpp::XPtr<ResidentRModel> state(ptr);
    if(!state.get()||!state->handle)Rcpp::stop("resident CUDA state is unavailable; refit the model");
    auto dx=dimensions(X,state->precision);
    if(dx[1]!=state->p)Rcpp::stop("test predictor dimension differs from the fitted model");
    Rcpp::Shield<SEXP> result(Rf_allocMatrix(state->precision==32?INTSXP:REALSXP,dx[0],state->q));
    void* out=state->precision==32?static_cast<void*>(INTEGER(result)):static_cast<void*>(REAL(result));
    char error[1024];
    if(classifier!=0&&classifier!=1)Rcpp::stop("invalid resident classifier");
    auto predict=classifier==1?fastpls_resident_lda_predict:fastpls_resident_simpls_predict;
    if(predict(state->handle,data(X,state->precision),dx[0],ncomp,out,error,sizeof(error)))
        Rcpp::stop("%s",error);
    return result;
#else
    Rcpp::stop("CUDA resident prediction is unavailable in this build; no CPU fallback is performed");
#endif
}

// [[Rcpp::export]]
SEXP cuda_resident_predict_path_cpp(
    Rcpp::List object,SEXP X,Rcpp::IntegerVector ncomp,int classifier=0) {
#ifdef FASTPLS_HAS_CUDA
    auto state=resident_state(object);auto dx=dimensions(X,state->precision);
    if(dx[1]!=state->p||ncomp.size()<1)
        Rcpp::stop("invalid predictor dimension or empty component path");
    int previous=0;
    for(int value:ncomp) {
        if(value<=previous||value>state->a)
            Rcpp::stop("ncomp must be strictly increasing and within the fitted path");
        previous=value;
    }
    if(classifier!=0&&classifier!=1)Rcpp::stop("invalid resident classifier");
    Rcpp::Shield<SEXP> result(Rf_allocVector(
        state->precision==32?INTSXP:REALSXP,
        static_cast<R_xlen_t>(dx[0])*state->q*ncomp.size()));
    Rf_setAttrib(result,R_DimSymbol,Rcpp::IntegerVector::create(
        dx[0],state->q,ncomp.size()));
    void* out=state->precision==32?static_cast<void*>(INTEGER(result)):
        static_cast<void*>(REAL(result));
    char error[1024];
    if(fastpls_resident_predict_path(
        state->handle,data(X,state->precision),dx[0],ncomp.begin(),ncomp.size(),
        classifier,out,error,sizeof(error)))Rcpp::stop("%s",error);
    return result;
#else
    Rcpp::stop("CUDA resident prediction path is unavailable in this build; no CPU fallback is performed");
#endif
}
