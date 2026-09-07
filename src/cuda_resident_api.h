#ifndef FASTPLS_CUDA_RESIDENT_API_H
#define FASTPLS_CUDA_RESIDENT_API_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
// Precision is 32 or 64. Matrices are column-major; labels are one-based.
void* fastpls_resident_simpls_create(const void* x,const void* y,const int* labels,
    int precision,int n,int p,int q,int components,int scaling,int oversample,
    int power,int retain_scores,unsigned long long seed,char* error,
    size_t error_capacity);
void* fastpls_resident_plssvd_create(const void* x,const void* y,const int* labels,
    int precision,int n,int p,int q,int components,int scaling,int oversample,
    int power,int retain_scores,unsigned long long seed,char* error,
    size_t error_capacity);
void* fastpls_resident_opls_create(const void* x,const void* y,const int* labels,
    int precision,int n,int p,int q,int components,int scaling,int oversample,
    int power,int retain_scores,unsigned long long seed,int north,char* error,
    size_t error_capacity);
void* fastpls_resident_kernelpls_create(const void* x,const void* y,
    const int* labels,int precision,int n,int p,int q,int components,
    int scaling,int oversample,int power,int retain_scores,
    unsigned long long seed,int kernel,double gamma,int degree,double coef0,
    char* error,size_t error_capacity);
int fastpls_resident_simpls_predict(void* model,const void* x,int rows,int prefix,
    void* predictions,char* error,size_t error_capacity);
int fastpls_resident_lda_predict(void* model,const void* x,int rows,int prefix,
    void* predictions,char* error,size_t error_capacity);
int fastpls_resident_predict_path(void* model,const void* x,int rows,
    const int* prefixes,int prefix_count,int lda,void* predictions,
    char* error,size_t error_capacity);
void fastpls_resident_simpls_destroy(void* model);
int fastpls_resident_export(void* model,int field,void* out,size_t elements,
    char* error,size_t error_capacity);
int fastpls_resident_classify(void* model,const void* x,int rows,int prefix,int lda,int top,
    int* labels,char* error,size_t error_capacity);
int fastpls_resident_classify_path(void* model,const void* x,int rows,
    const int* prefixes,int prefix_count,int lda,int top,int* labels,
    char* error,size_t error_capacity);
int fastpls_resident_classify_response_path(void* model,const void* x,int rows,
    const int* prefixes,int prefix_count,int lda,int top,int* labels,
    void* predictions,char* error,size_t error_capacity);
int fastpls_resident_response_sums(void* model,const void* x,const void* y,
    const int* labels,int rows,int prefix,void* out,char* error,size_t error_capacity);
int fastpls_resident_project(void* model,const void* x,int rows,int prefix,
    void* out,char* error,size_t error_capacity);
int fastpls_resident_controls(void* model,int* oversample,int* power,int* block,
    int* block_limit,int* implicit_operator,int* predictor_crossprod_cache,
    char* error,size_t error_capacity);
int fastpls_resident_compact(void* model,int prepare_lda,
    char* error,size_t error_capacity);
#ifdef __cplusplus
}
#endif
#endif
