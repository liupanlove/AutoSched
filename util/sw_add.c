/******************************************
 * Created by Liandeng Li
 * Date: 2017/10/5
 * addtify  functions: (in SPEs)
 * 1. double: sw_add_d(double* src1,double *src2, double* dst, long count)
 * 2. float : sw_add_f(float * src1,float *src2, float * dst, long count)
 * ***************************************/
#include "athread.h"
#include "simd.h"


extern SLAVE_FUN(sw_slave_add_d)();
extern SLAVE_FUN(sw_slave_add_i)();
extern SLAVE_FUN(sw_slave_add_4cg_d)();
extern SLAVE_FUN(sw_slave_add_4cg_f)();
extern SLAVE_FUN(sw_slave_add_4cg_i)();
extern SLAVE_FUN(sw_slave_add_f)();
typedef struct addTransPara_st {
  void *src1;
  void *src2;
  void *src3;
  void *src4;
  void *dst;
  long count;
}addPara;
// Precondition: already athread_init()
static void *do_slave_4cg_add_f(void * lParam)
{
     addPara *param = (addPara*)lParam;
     if(!athread_get_num_threads()) athread_init();
		 athread_spawn(sw_slave_add_4cg_f,(void*)param);
		 athread_join();
     pthread_exit(0);
}
static void *do_slave_4cg_add_i(void * lParam)
{
     addPara *param = (addPara*)lParam;
     if(!athread_get_num_threads()) athread_init();
		 athread_spawn(sw_slave_add_4cg_i,(void*)param);
		 athread_join();
     pthread_exit(0);
}
static void *do_slave_add_f(void * lParam)
{
     addPara *param = (addPara*)lParam;
     if(!athread_get_num_threads()) athread_init();
		 athread_spawn(sw_slave_add_f,(void*)param);
		 athread_join();
     pthread_exit(0);
}
static void *do_slave_add_i(void * lParam)
{
     addPara *param = (addPara*)lParam;
     if(!athread_get_num_threads()) athread_init();
		 athread_spawn(sw_slave_add_i,(void*)param);
		 athread_join();
     pthread_exit(0);
}
void sw_add_4cg_f(float* src1,float *src2, float * src3,
                  float *src4,float* dst,const long count) {
  long min_size = 8192;
  if(count < min_size){
    int i = 0;
    floatv4 va1,va2,va3,va4;
    const int simdsize = 4;
    for(i=0; i+simdsize-1<count; i+=simdsize) {
       simd_load(va1,src1+i);
       simd_load(va2,src2+i);
       simd_load(va3,src3+i);
       va1 = va1 + va2;
       simd_load(va4,src4+i);
       va1 = va1 + va3; 
       va1 = va1 + va4;
       simd_store(va1,dst+i);
    }
    for(;i<count;i++)
      dst[i] = src1[i] + src2[i] + src3[i] + src4[i];
    return;
  }
#ifdef USE_4CG
  int i=0,NUM_THREADS=4;
  pthread_t pthread_handler[NUM_THREADS];
  addPara *para = (addPara*)malloc(sizeof(addPara)*NUM_THREADS);
  int block_size = 0 , offset = 0; 
  for(i=0; i<NUM_THREADS; i++)
  { 
     block_size = count/NUM_THREADS + (i<(count%NUM_THREADS));
     offset = i*(count/NUM_THREADS) + (i<(count%NUM_THREADS) ? i : (count%NUM_THREADS));
     para[i].src1 = src1+offset;
     para[i].src2 = src2+offset;
     para[i].src3 = src3+offset;
     para[i].src4 = src4+offset;
     para[i].dst = dst+offset;
     para[i].count = block_size;
  }
  for(i=0; i<NUM_THREADS; i++)
       pthread_create(&pthread_handler[i],NULL, do_slave_4cg_add_f, (void*)(&para[i]));
  for(i=0; i<NUM_THREADS; i++)
       pthread_join(pthread_handler[i], NULL);
  free(para);
#else
  addPara *para = (addPara*)malloc(sizeof(addPara));
  para->src1 = src1;
  para->src2 = src2;
  para->src3 = src3;
  para->src4 = src4;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_slave_add_4cg_f,para);
  athread_join();
  free(para);
#endif
}
void sw_add_4cg_d(double* src1,double *src2, double * src3,
                  double *src4,double* dst,const long count) {
  long min_size = 8192;
  if(count < min_size){
    int i = 0;
    doublev4 va1,va2,va3,va4;
    const int simdsize = 4;
    for(i=0; i+simdsize-1<count; i+=simdsize) {
       simd_load(va1,src1+i);
       simd_load(va2,src2+i);
       simd_load(va3,src3+i);
       va1 = va1 + va2;
       simd_load(va4,src4+i);
       va1 = va1 + va3; 
       va1 = va1 + va4;
       simd_store(va1,dst+i);
    }
    for(;i<count;i++)
      dst[i] = src1[i] + src2[i] + src3[i] + src4[i];
    return;
  }
  addPara *para = (addPara*)malloc(sizeof(addPara));
  para->src1 = src1;
  para->src2 = src2;
  para->src3 = src3;
  para->src4 = src4;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_slave_add_4cg_d,para);
  athread_join();
  free(para);
}
void sw_add_4cg_i(int* src1,int *src2,int * src3,
                  int *src4,int* dst,const long count) {
  long min_size = 8192;
  if(count < min_size){
    int i = 0;
    intv8 va1,va2,va3,va4;
    const int simdsize = 8;
    for(i=0; i+simdsize-1<count; i+=simdsize) {
       simd_load(va1,src1+i);
       simd_load(va2,src2+i);
       simd_load(va3,src3+i);
       va1 = va1 + va2;
       simd_load(va4,src4+i);
       va1 = va1 + va3; 
       va1 = va1 + va4;
       simd_store(va1,dst+i);
    }
    for(;i<count;i++)
      dst[i] = src1[i] + src2[i] + src3[i] + src4[i];
    return;
  }
#ifdef USE_4CG
  int i=0,NUM_THREADS=4;
  pthread_t pthread_handler[NUM_THREADS];
  addPara *para = (addPara*)malloc(sizeof(addPara)*NUM_THREADS);
  int block_size = 0 , offset = 0; 
  for(i=0; i<NUM_THREADS; i++)
  { 
     block_size = count/NUM_THREADS + (i<(count%NUM_THREADS));
     offset = i*(count/NUM_THREADS) + (i<(count%NUM_THREADS) ? i : (count%NUM_THREADS));
     para[i].src1 = src1+offset;
     para[i].src2 = src2+offset;
     para[i].src3 = src3+offset;
     para[i].src4 = src4+offset;
     para[i].dst = dst+offset;
     para[i].count = block_size;
  }
  for(i=0; i<NUM_THREADS; i++)
       pthread_create(&pthread_handler[i],NULL, do_slave_4cg_add_i, (void*)(&para[i]));
  for(i=0; i<NUM_THREADS; i++)
       pthread_join(pthread_handler[i], NULL);
  free(para);
#else
  addPara *para = (addPara*)malloc(sizeof(addPara));
  para->src1 = src1;
  para->src2 = src2;
  para->src3 = src3;
  para->src4 = src4;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_slave_add_4cg_i,para);
  athread_join();
  free(para);
#endif
}
void sw_add_d(double* src1,double *src2, double* dst, const long count) {
  long min_size = 8192;
  if(count < min_size)
  {
    int simdsize = 4,i=0;
    doublev4 vc1,vc2;
    for(i=0;i+simdsize-1<count;i+=simdsize)
    {
      simd_load(vc1,src1+i);
      simd_load(vc2,src2+i);
      vc1 = vc1 + vc2;
      simd_store(vc1,dst+i);
    }
    for(;i<count;i++)
      dst[i] = src1[i]+src2[i];
    return;
  }
  addPara *para = (addPara*)malloc(sizeof(addPara));
  para->src1 = src1;
  para->src2 = src2;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_slave_add_d,para);
  athread_join();
  free(para);
}
void sw_add_i(int* src1,int *src2, int* dst, const long count) {
  long min_size = 8192;
  if(count < min_size)
  {
    int simdsize = 8,i=0;
    intv8 vc1,vc2;
    for(i=0;i+simdsize-1<count;i+=simdsize)
    {
      simd_load(vc1,src1+i);
      simd_load(vc2,src2+i);
      vc1 = vc1 + vc2;
      simd_store(vc1,dst+i);
    }
    for(;i<count;i++)
      dst[i] = src1[i]+src2[i];
    return;
  }
#ifdef USE_4CG
  int i=0,NUM_THREADS=4;
  pthread_t pthread_handler[NUM_THREADS];
  addPara *para = (addPara*)malloc(sizeof(addPara)*NUM_THREADS);
  int block_size = 0 , offset = 0; 
  for(i=0; i<NUM_THREADS; i++)
  { 
     block_size = count/NUM_THREADS + (i<(count%NUM_THREADS));
     offset = i*(count/NUM_THREADS) + (i<(count%NUM_THREADS) ? i : (count%NUM_THREADS));
     para[i].src1 = src1+offset;
     para[i].src2 = src2+offset;
     para[i].dst = dst+offset;
     para[i].count = block_size;
  }
  for(i=0; i<NUM_THREADS; i++)
       pthread_create(&pthread_handler[i],NULL, do_slave_add_i, (void*)(&para[i]));
  for(i=0; i<NUM_THREADS; i++)
       pthread_join(pthread_handler[i], NULL);
  free(para);
#else
  addPara *para = (addPara*)malloc(sizeof(addPara));
  para->src1 = src1;
  para->src2 = src2;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_slave_add_i,para);
  athread_join();
  free(para);
#endif
}
void sw_add_f(float* src1,float *src2, float* dst,const long count) {
  long min_size = 8192;
  if(count < min_size)
  {
    long simdsize = 4,i=0;
    floatv4 vc1,vc2;
    for(i=0;i+simdsize-1<count;i+=simdsize)
    {
      simd_load(vc1,src1+i);
      simd_load(vc2,src2+i);
      vc1 = vc1 + vc2;
      simd_store(vc1,dst+i);
    }
    for(;i<count;i++)
      dst[i] = src1[i]+src2[i];
    return;
  }
#ifdef USE_4CG
  int i=0,NUM_THREADS=4;
  pthread_t pthread_handler[NUM_THREADS];
  addPara *para = (addPara*)malloc(sizeof(addPara)*NUM_THREADS);
  int block_size = 0 , offset = 0; 
  for(i=0; i<NUM_THREADS; i++)
  { 
     block_size = count/NUM_THREADS + (i<(count%NUM_THREADS));
     offset = i*(count/NUM_THREADS) + (i<(count%NUM_THREADS) ? i : (count%NUM_THREADS));
     para[i].src1 = src1+offset;
     para[i].src2 = src2+offset;
     para[i].dst = dst+offset;
     para[i].count = block_size;
  }
  for(i=0; i<NUM_THREADS; i++)
       pthread_create(&pthread_handler[i],NULL, do_slave_add_f, (void*)(&para[i]));
  for(i=0; i<NUM_THREADS; i++)
       pthread_join(pthread_handler[i], NULL);
  free(para);
#else
  addPara *para = (addPara*)malloc(sizeof(addPara));
  para->src1 = src1;
  para->src2 = src2;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_slave_add_f,para);
  athread_join();
  free(para);
#endif
}
