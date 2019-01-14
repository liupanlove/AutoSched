/******************************************
 * Created by Liandeng Li
 * Date: 2017/8/7
 * Memory copy functions: (in SPEs)
 * 1. double: sw_memset_d(double* src, long count)
 * 2. float : sw_memset_f(float * src, long count)
 * ***************************************/
#include "athread.h"
#define NUM_THREADS 4

extern SLAVE_FUN(sw_slave_memset_d)();
extern SLAVE_FUN(sw_slave_memset_f)();
extern SLAVE_FUN(sw_slave_memset_i)();
typedef struct memsetTransPara_st {
  void *src;
  long count;
}MemsetPara;
// Precondition: already athread_init()
void sw_memset_d(double* src,const long count) {
  MemsetPara *para = (MemsetPara*)malloc(sizeof(MemsetPara));
  para->src = src;
  para->count = count;
  athread_spawn(sw_slave_memset_d,para);
  athread_join();
  free(para);
}
static void *do_slave_memset(void * lParam)
{
     MemsetPara *param = (MemsetPara*)lParam;
     if(!athread_get_num_threads()) athread_init();
		 athread_spawn(sw_slave_memset_f,(void*)param);
		 athread_join();
     pthread_exit(0);
}
void sw_memset_f(float* src,const long count) {
#ifdef USE_4CG
  pthread_t pthread_handler[NUM_THREADS];
  MemsetPara *para = (MemsetPara*)malloc(sizeof(MemsetPara)*NUM_THREADS);
  int block_size = 0 , offset = 0,i; 
  for(i=0; i<NUM_THREADS; i++)
  { 
     block_size = count/NUM_THREADS + (i<(count%NUM_THREADS));
     offset = i*(count/NUM_THREADS) + (i<(count%NUM_THREADS) ? i : (count%NUM_THREADS));
     para[i].src = src+offset;
     para[i].count = block_size;
  }
  for(i=0; i<NUM_THREADS; i++)
       pthread_create(&pthread_handler[i],NULL, do_slave_memset, (void*)(&para[i]));
  for(i=0; i<NUM_THREADS; i++)
       pthread_join(pthread_handler[i], NULL);
  free(para);
#else
  MemsetPara *para = (MemsetPara*)malloc(sizeof(MemsetPara));
  para->src = src;
  para->count = count;
  athread_spawn(sw_slave_memset_f,para);
  athread_join();
  free(para);
#endif
}
void sw_memset_i(long* src,const long count) {
  MemsetPara *para = (MemsetPara*)malloc(sizeof(MemsetPara));
  para->src = src;
  para->count = count;
  athread_spawn(sw_slave_memset_i,para);
  athread_join();
  free(para);
}
