/******************************************
 * Created by Xin You
 * Date: 2017/8/7
 * Memory copy functions: (in SPEs)
 * 1. double: sw_memcpy_d(double* src, double* dst, int count)
 * 2. float : sw_memcpy_f(float * src, float * dst, int count)
 * ***************************************/
#include "athread.h"
#define NUM_THREADS 4

extern SLAVE_FUN(sw_slave_memcpy_d)();
extern SLAVE_FUN(sw_slave_memcpy_f)();
extern SLAVE_FUN(sw_slave_memcpy_i)();
extern SLAVE_FUN(sw_slave_memcpy_c2d)();
extern SLAVE_FUN(sw_slave_memcpy_c2f)();
typedef struct MemcpyTransPara_st {
  void *src;
  void *dst;
  long count;
}MemcpyPara;
// Precondition: already athread_init()
void sw_memcpy_d(double* src, double* dst,const long count) {
  MemcpyPara *para = (MemcpyPara*)malloc(sizeof(MemcpyPara));
  para->src = src;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_slave_memcpy_d,para);
  athread_join();
  free(para);
}
static void *do_slave_memcpy(void * lParam)
{
     MemcpyPara *param = (MemcpyPara*)lParam;
     if(!athread_get_num_threads()) athread_init();
		 athread_spawn(sw_slave_memcpy_f,(void*)param);
		 athread_join();
     pthread_exit(0);
}
void sw_memcpy_f(float* src, float* dst,const long count) {
#ifdef USE_4CG
  pthread_t pthread_handler[NUM_THREADS];
  MemcpyPara *para = (MemcpyPara*)malloc(sizeof(MemcpyPara)*NUM_THREADS);
  int block_size = 0 , offset = 0,i; 
  for(i=0; i<NUM_THREADS; i++)
  { 
     block_size = count/NUM_THREADS + (i<(count%NUM_THREADS));
     offset = i*(count/NUM_THREADS) + (i<(count%NUM_THREADS) ? i : (count%NUM_THREADS));
     para[i].src = src+offset;
     para[i].dst = dst+offset;
     para[i].count = block_size;
  }
  for(i=0; i<NUM_THREADS; i++)
       pthread_create(&pthread_handler[i],NULL, do_slave_memcpy, (void*)(&para[i]));
  for(i=0; i<NUM_THREADS; i++)
       pthread_join(pthread_handler[i], NULL);
  free(para);
#else
  MemcpyPara *para = (MemcpyPara*)malloc(sizeof(MemcpyPara));
  para->src = src;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_slave_memcpy_f,para);
  athread_join();
  free(para);
#endif
}
void sw_memcpy_i(int* src, int* dst,const long count) {
  MemcpyPara *para = (MemcpyPara*)malloc(sizeof(MemcpyPara));
  para->src = src;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_slave_memcpy_i,para);
  athread_join();
  free(para);
}
void sw_memcpy_c2d(unsigned char* src, double* dst,const long count) {
  MemcpyPara *para = (MemcpyPara*)malloc(sizeof(MemcpyPara));
  para->src = src;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_slave_memcpy_c2d,para);
  athread_join();
  free(para);
}
void sw_memcpy_c2f(unsigned char* src, float* dst,const long count) {
  MemcpyPara *para = (MemcpyPara*)malloc(sizeof(MemcpyPara));
  para->src = src;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_slave_memcpy_c2f,para);
  athread_join();
  free(para);
}
