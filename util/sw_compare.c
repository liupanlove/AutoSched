/******************************************
 * Created by Liandeng Li
 * Date: 2017/10/5
 * Division  functions: (in SPEs)
 * 1. double: sw_div_d(double* src, double* dst, int count)
 * 2. float : sw_div_f(float * src, float * dst, int count)
 * ***************************************/
#include "athread.h"
#include "simd.h"

extern SLAVE_FUN(sw_slave_div_d)();
extern SLAVE_FUN(sw_slave_div_f)();
typedef struct DivTransPara_st {
  void *src,*dst,*old_center;
  int *center_num;
  int count,dims;
}DivPara;
// Precondition: already athread_init()
void sw_div_d(double* src,double* dst,double * old,int* center_num,const int count,const int dims) {
  int min_size = 4096,i,j,simd_size=4;
  long total_size = count*dims;
  int num,index;
  doublev4 vsrc,vdst,vc;
  if(total_size < min_size)
  {
    for(i=0;i<count;i++){
      num = center_num[i];
      index = i*dims;
      if(num > 0 ){
        vc = num;
        for(j=0;j+simd_size-1<dims;j+=simd_size){
          simd_load(vsrc,(src+index+j));
          vdst = vsrc / vc;
          simd_store(vdst,dst+index+j);
        }
        for(;j<dims;j++)
          dst[index+j] = src[index+j] / num;
      }
      else{
        for(j=0;j+simd_size-1<dims;j+=simd_size){
          simd_load(vsrc,(old+index+j));
          simd_store(vsrc,dst+index+j);
        }
        for(;j<dims;j++)
          dst[index+j] = old[index+j];
      }
    }
    return;
  }
  DivPara *para = (DivPara*)malloc(sizeof(DivPara));
  para->src = src;
  para->dst = dst;
  para->old_center = old;
  para->center_num = center_num;
  para->count = count;
  para->dims = dims;
  athread_spawn(sw_slave_div_d,para);
  athread_join();
  free(para);
}

static void *do_slave_div(void * lParam)
{
     DivPara *param = (DivPara*)lParam;
     if(!athread_get_num_threads()) athread_init();
		 athread_spawn(sw_slave_div_f,(void*)param);
		 athread_join();
     pthread_exit(0);
}
void sw_div_f(float* src,float * dst,float * old,int* center_num,const int count,const int dims) {
  int min_size = 4096,i,j,simd_size=4,NUM_THREADS = 4;
  long total_size = count*dims;
  int num,index;
  floatv4 vsrc,vdst,vc;
  if(total_size < min_size){
    for(i=0;i<count;i++){
      num = center_num[i];
      index = i*dims;
      if(num > 0 ){
        vc = num;
        for(j=0;j+simd_size-1<dims;j+=simd_size){
          simd_load(vsrc,(src+index+j));
          vdst = vsrc / vc;
          simd_store(vdst,dst+index+j);
        }
        for(;j<dims;j++)
          dst[index+j] = src[index+j] / num;
      }
      else{
        for(j=0;j+simd_size-1<dims;j+=simd_size){
          simd_load(vsrc,(old+index+j));
          simd_store(vsrc,dst+index+j);
        }
        for(;j<dims;j++)
          dst[index+j] = old[index+j];
      }
    }
    return;
  }
#ifdef USE_4CG
  pthread_t pthread_handler[NUM_THREADS];
  DivPara *para = (DivPara*)malloc(sizeof(DivPara)*NUM_THREADS);
  int block_size = 0 , offset = 0; 
  for(i=0; i<NUM_THREADS; i++)
  { 
     block_size = count/NUM_THREADS + (i<(count%NUM_THREADS));
     offset = i*(count/NUM_THREADS) + (i<(count%NUM_THREADS) ? i : (count%NUM_THREADS));
     para[i].center_num = center_num+offset;
     offset *= dims;
     para[i].src = src+offset;
     para[i].dst = dst+offset;
     para[i].old_center = old+offset;
     para[i].count = block_size;
     para[i].dims = dims;
  }
  for(i=0; i<NUM_THREADS; i++)
       pthread_create(&pthread_handler[i],NULL, do_slave_div, (void*)(&para[i]));
  for(i=0; i<NUM_THREADS; i++)
       pthread_join(pthread_handler[i], NULL);
  free(para);
#else
  DivPara *para = (DivPara*)malloc(sizeof(DivPara));
  para->src = src;
  para->dst = dst;
  para->old_center = old;
  para->center_num = center_num;
  para->count = count;
  para->dims = dims;
  athread_spawn(sw_slave_div_f,para);
  athread_join();
  free(para);
#endif
}

