#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>  
#include <fcntl.h>
#include <string.h>  
#include <time.h>  
#include <math.h>  
#include <athread.h>
#include <pthread.h>
#include <simd.h>
#include <assert.h>
#include "mpi.h"
#include <float.h>

#define SIMDSIZE    4
#define NUM_THREADS 4
#define MAX_ROUND 5000 //最大的聚类次数
#define MAX_SUPER_NODES 256 //感觉应该是MAX_NODES

#define CLUSTER_COUNT (2000)
#define DIMS          (196608)

//#define USE_IMAGENET

extern SLAVE_FUN(sw_slave_kmeans_f)();
extern SLAVE_FUN(get_data_group)();
extern SLAVE_FUN(caculate_radius)();
//extern SLAVE_FUN(caculate_evaluation_function)();
//extern SLAVE_FUN(test)();

typedef struct _KmeansPara{
  int dims;
  int data_size;
  int cluster_count;
  int *cluster_center_num;
  float*data;
  int rank;
  //int * data_group; // record the data belongs to which cluster
  float* cluster_center;
  float* cluster_center_out;
  //float * all_data; // each cg has all data
} KmeansPara;
typedef struct _tagNodeInfo{
  int rank;
  int mid_id;
}NodeInfo;
typedef struct _tagSuperNodes{
  int root;
  int comm_size;
}SuperNodes;

typedef struct _EvaluationPara{
  char * out_filename;
  int rank;
  int dims;
  int data_size;
  int all_data_size;
  int cluster_count;
  int data_start; // the start index of data in all_data
  int * all_data_group;
  float * cluster_center;
  float * data;
  int * data_group;
  float * all_data;
  float * cluster_distance_count;
  float * non_cluster_distance_count;
  float * radius;
  float * distance;
} EvaluationPara;
//Custom Allreduce

int   root_index = 0;
int   root_ranks[MAX_SUPER_NODES];
SuperNodes root_nodes[MAX_SUPER_NODES];
MPI_Group  prime_group[MAX_SUPER_NODES];
MPI_Comm   prime_comm[MAX_SUPER_NODES];

inline void mb()
{
    asm volatile("memb");
    asm volatile("":::"memory");
}
int caffe_mpi_bcast_f( void *buffer, long count, int root,MPI_Comm comm ) {
  return MPI_Bcast(buffer,count , MPI_FLOAT, root, comm);
  /*int comm_size,rank;
  MPI_Status status;
  MPI_Request send_req,recv_req;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &rank);
  int start = 0,mid = 0,tag = 1;
  int end = comm_size -1;
  while(1){
    if(start == end) break;
    mid = (start + end +1)>>1;
    if(rank >= start && rank <= mid -1)//front half
    {
      if(start == rank){
	      MPI_Isend(buffer, count,MPI_FLOAT,mid, tag,comm,&send_req);
      }
      end = mid - 1;
    }
    else if(rank >= mid && rank <= end){
      if(rank == mid){
        MPI_Irecv(buffer,count,MPI_FLOAT,start,tag,comm,&recv_req);
        MPI_Wait(&recv_req,&status);
      }
      start = mid;
    }
  }*/
}

int caffe_mpi_reduce_f( void *sendbuf, void *recvbuf, int count,MPI_Op op, int root, MPI_Comm comm  ){
    int comm_size,rank;
    MPI_Status status;
    MPI_Request send_req,recv_req;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &rank);
    int mask = 0x1,source=0,tag = 10;
    int relrank = (rank - root + comm_size) % comm_size;

    sw_memcpy_f((float*)sendbuf,(float*)recvbuf,count);
    
    float * tmp_buff = (float*)malloc(count * sizeof(float));
    assert(tmp_buff != NULL);
    while(mask < comm_size){
      // Receive
      if ((mask & relrank) == 0) {
        source = (relrank | mask);
        if (source < comm_size) {
          source = (source + root) % comm_size;
	        MPI_Irecv(tmp_buff,count,MPI_FLOAT,source,tag,comm,&recv_req);
          MPI_Wait(&recv_req,&status);
          sw_add_f((float*)tmp_buff,(float*)recvbuf,(float*)recvbuf,count);
          
        }
      }
      else {
         //I've received all that I'm going to.  Send my result to my parent 
         source = ((relrank & (~ mask)) + root) % comm_size;
	       MPI_Isend(recvbuf, count,MPI_FLOAT, source, tag,comm,&send_req);
         break;
      }
      mask = mask << 1;
    }
    free(tmp_buff);
    return 0;
}
int caffe_mpi_supernode_allreduce_f( void *sendbuf, void *recvbuf, int count, int root_count,int* ranks, MPI_Comm comm  ){
  int i = 0,index = -1;
  int pof2 = 1,dest=0,tag=1;
  int mpi_rank;
  MPI_Request recv_req,send_req;
  MPI_Status  statue;
  MPI_Comm_rank(comm, &mpi_rank);
  for(i = 0;i < root_count;i++)
  {
     if(mpi_rank != ranks[i])
       continue;
     index = i;
     break;
  }
  if(index < 0) return 0;

  sw_memcpy_f((float*)sendbuf,(float*)recvbuf,count);
  
  while(pof2 < root_count){
     dest = index ^ pof2;
     if(dest < root_count){
        MPI_Irecv(sendbuf, count,MPI_FLOAT, ranks[dest], tag,comm,&recv_req);
        MPI_Isend(recvbuf, count,MPI_FLOAT, ranks[dest], tag,comm,&send_req);
        MPI_Wait(&recv_req,&statue);
        MPI_Wait(&send_req,&statue);
        sw_add_f((float*)sendbuf,(float*)recvbuf,(float*)recvbuf,count);
        
     }
     pof2 = pof2 << 1;
  }
  return 0;
}
int caffe_mpi_reduce_i( void *sendbuf, void *recvbuf, int count,MPI_Op op, int root, MPI_Comm comm  ){
    int comm_size,rank;
    MPI_Status status;
    MPI_Request send_req,recv_req;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &rank);
    int mask = 0x1,source=0,tag = 11;
    int relrank = (rank - root + comm_size) % comm_size;

    sw_memcpy_i((int*)sendbuf,(int*)recvbuf,count);
    
    int * tmp_buff = (int*)malloc(count * sizeof(int));
    assert(tmp_buff != NULL);
    while(mask < comm_size){
      // Receive
      if ((mask & relrank) == 0) {
        source = (relrank | mask);
        if (source < comm_size) {
          source = (source + root) % comm_size;
	        MPI_Irecv(tmp_buff,count,MPI_INT,source,tag,comm,&recv_req);
          MPI_Wait(&recv_req,&status);
          sw_add_i((int*)tmp_buff,(int*)recvbuf,(int*)recvbuf,count);
          
        }
      }
      else {
         //I've received all that I'm going to.  Send my result to my parent 
         source = ((relrank & (~ mask)) + root) % comm_size;
	       MPI_Isend(recvbuf, count,MPI_INT, source, tag,comm,&send_req);
         break;
      }
      mask = mask << 1;
    }
    free(tmp_buff);
    return 0;
}
int caffe_mpi_supernode_allreduce_i( void *sendbuf, void *recvbuf, int count, int root_count,int* ranks, MPI_Comm comm  ){
  int i = 0,index = -1;
  int pof2 = 1,dest=0,tag=2;
  int mpi_rank;
  MPI_Request recv_req,send_req;
  MPI_Status  statue;
  MPI_Comm_rank(comm, &mpi_rank);
  for(i = 0;i < root_count;i++)
  {
     if(mpi_rank != ranks[i])
       continue;
     index = i;
     break;
  }
  if(index < 0) return 0;

  sw_memcpy_i((int*)sendbuf,(int*)recvbuf,count);
  
  while(pof2 < root_count){
     dest = index ^ pof2;
     if(dest < root_count){
        MPI_Irecv(sendbuf, count,MPI_INT, ranks[dest], tag,comm,&recv_req);
        MPI_Isend(recvbuf, count,MPI_INT, ranks[dest], tag,comm,&send_req);
        MPI_Wait(&recv_req,&statue);
        MPI_Wait(&send_req,&statue);
        sw_add_i((int*)sendbuf,(int*)recvbuf,(int*)recvbuf,count);
        
     }
     pof2 = pof2 << 1;
  }
  return 0;
}
int caffe_mpi_reduce( void *sendbuf, void *recvbuf,long pack_size,long count,long num,int root, MPI_Comm comm){
    MPI_Reduce((void *)sendbuf, (void *)recvbuf, count, MPI_FLOAT, MPI_SUM, root, comm);
    MPI_Reduce((void *)(sendbuf + count*sizeof(float)), (void *)(recvbuf + count*sizeof(float)), num, MPI_INT, MPI_SUM, root, comm);
    /*int comm_size,rank;
    MPI_Status status;
    MPI_Request send_req,recv_req;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &rank);
    int mask = 0x1,source=0,tag = 11;
    int relrank = (rank - root + comm_size) % comm_size;
    sw_memcpy_f((float*)sendbuf,(float*)recvbuf,count);
    sw_memcpy_i((int*)(sendbuf + count*sizeof(float)),(int*)(recvbuf + count*sizeof(float)),num);
    long buff_size = pack_size >> 2;
    //printf("rank=%d root=%d comm_size=%d buff_size=%d\n", rank, root, comm_size, buff_size);
    while(mask < comm_size){
      // Receive
      if ((mask & relrank) == 0) {
        source = (relrank | mask);
        if (source < comm_size) {
          source = (source + root) % comm_size;
	        //MPI_Irecv(sendbuf,pack_size,MPI_CHAR,source,tag,comm,&recv_req);
	        MPI_Irecv(sendbuf,buff_size,MPI_INT,source,tag,comm,&recv_req);
          if(rank == 0)
          {
            printf("rank=%d test MPI_Wait start\n", rank);
          }
          MPI_Wait(&recv_req,&status);
          if(rank == 0)
          {
            printf("rank=%d test MPI_Wait end\n", rank);
          }

          sw_add_f((float*)sendbuf,(float*)recvbuf,(float*)recvbuf,count);
          sw_add_i((int*)(sendbuf+count*sizeof(float)),(int*)(recvbuf+count*sizeof(float)),(int*)(recvbuf+count*sizeof(float)));
        }
      }
      else {
         //I've received all that I'm going to.  Send my result to my parent 
         source = ((relrank & (~ mask)) + root) % comm_size;
	       //MPI_Isend(recvbuf,pack_size,MPI_CHAR, source, tag,comm,&send_req);
	       MPI_Isend(recvbuf,buff_size,MPI_INT, source, tag,comm,&send_req);
         break;
      }
      mask = mask << 1;
    }*/

    /*if(rank == 0)
    {
    int i;
    float * tmp = recvbuf;
    for(i = 0; i < count; ++i)
    {
      printf("%f ", tmp[i]);
    }
    printf("\n");
    int * tmp1 = recvbuf + count * sizeof(float);
    for(i = 0; i < num; ++i)
      printf("%d ", tmp1[i]);
    printf("\n");
    }*/
    return 0;
}
int caffe_mpi_supernode_allreduce( void *sendbuf, void *recvbuf,long pack_size,long count,long num, int root_count,int* ranks, MPI_Comm comm){
  int i = 0,index = -1;
  int pof2 = 1,dest=0,tag=2;
  int mpi_rank;
  MPI_Request recv_req,send_req;
  MPI_Status  statue;
  MPI_Comm_rank(comm, &mpi_rank);
  for(i = 0;i < root_count;i++)
  {
     if(mpi_rank != ranks[i])
       continue;
     index = i;
     break;
  }
  if(index < 0) return 0;

  sw_memcpy_f((float*)sendbuf,(float*)recvbuf,count);
  sw_memcpy_i((int*)(sendbuf + count*sizeof(float)),(int*)(recvbuf + count*sizeof(float)),num);
  long buff_size = pack_size >> 2;
  
  while(pof2 < root_count){
     dest = index ^ pof2;
     if(dest < root_count){
        //MPI_Irecv(sendbuf, pack_size,MPI_CHAR, ranks[dest], tag,comm,&recv_req);
        //MPI_Isend(recvbuf, pack_size,MPI_CHAR, ranks[dest], tag,comm,&send_req);
        MPI_Irecv(sendbuf, buff_size,MPI_INT, ranks[dest], tag,comm,&recv_req);
        MPI_Isend(recvbuf, buff_size,MPI_INT, ranks[dest], tag,comm,&send_req);
        MPI_Wait(&recv_req,&statue);
        MPI_Wait(&send_req,&statue);
        
        sw_add_f((float*)sendbuf,(float*)recvbuf,(float*)recvbuf,count);
        sw_add_i((int*)(sendbuf+count*sizeof(float)),(int*)(recvbuf+count*sizeof(float)),(int*)(recvbuf+count*sizeof(float)));

     }
     pof2 = pof2 << 1;
  }
  return 0;
}
void init_net_topology(){
   char node_info[20];
   int node_id = 0,i=0;
   int rank, numprocs;
   MPI_Comm_rank(MPI_COMM_WORLD,&rank);
   MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
   //Custom struct start
   int blockLength[] = {1,1};
   MPI_Datatype oldTypes[] = {MPI_INT,MPI_INT};
   MPI_Aint addressOffsets[] = {0,sizeof(int)};
   MPI_Datatype newType;
   MPI_Type_struct(
       sizeof(blockLength)/sizeof(int),
       blockLength,
       addressOffsets,
       oldTypes,&newType);
   MPI_Type_commit(&newType);
   //Custom struct end
   //Get process node id
   MPI_Get_processor_name(node_info, &node_id);
   sscanf(node_info,"vn%d",&node_id);
   printf("rank=%d node_info=%s\n", rank,  node_info);
   NodeInfo node;
   NodeInfo* nodes = (NodeInfo*)malloc(sizeof(NodeInfo)*numprocs);
   node.rank = rank;
   node.mid_id = node_id >> 8;
   //gather per process node info
   MPI_Allgather(&node,1,newType,nodes,1,newType,MPI_COMM_WORLD);
   //Judge the root rank and nodes num of Mid start
   int root = nodes[0].rank;
   int mid_id = nodes[0].mid_id;
   for(i = 1;i < numprocs- 1;i++)
   {
     if(mid_id != nodes[i].mid_id)
     {
       root_nodes[root_index].root = root;
       root_nodes[root_index].comm_size = nodes[i].rank - root;
       root = nodes[i].rank;
       mid_id = nodes[i].mid_id;
       root_index++;
     }
   }
   if(mid_id != nodes[numprocs-1].mid_id)
   {
      root_nodes[root_index].root = root;
      root_nodes[root_index].comm_size = nodes[numprocs-1].rank - root;
      root_index++;
      root_nodes[root_index].root = nodes[numprocs-1].rank;
      root_nodes[root_index].comm_size = 1;
      root_index++;
   }
   else{
      if(nodes[numprocs-1].rank != root){
        root_nodes[root_index].root = root;
        root_nodes[root_index].comm_size = nodes[numprocs-1].rank - root+1;
        root_index++;
      }
   }
   free(nodes);
   printf("rank=%d root_index=%d\n", rank, root_index);
   if(root_index>1)
   {
     MPI_Group world_group;
     MPI_Comm_group(MPI_COMM_WORLD,&world_group);
     int index = 0;
     //Create group and comm
     for(index=0;index < root_index;index++)
     {
       root_ranks[index] = root_nodes[index].root;
       int * ranks = (int*)malloc(root_nodes[index].comm_size*sizeof(int));
       for(i=0;i< root_nodes[index].comm_size;i++)
         ranks[i] = i + root_nodes[index].root;
       MPI_Group_incl(world_group,root_nodes[index].comm_size,ranks,&prime_group[index]);
       MPI_Comm_create_group(MPI_COMM_WORLD,prime_group[index],0,&prime_comm[index]);
       free(ranks);
     }
   }
   MPI_Barrier(MPI_COMM_WORLD);
}
void release_net_topology(){
    int i = 0;
    for(i=0;i<root_index;i++)
    {
       MPI_Group_free(&prime_group[i]);
       if(MPI_COMM_NULL != prime_comm[i])
          MPI_Comm_free(&prime_comm[i]);
    }
}
void format_result(float data_size,unsigned char info[20])
{
    if(data_size<1024)               {sprintf(info, "%4.3f  B", data_size);}
    else if(data_size<1024*1024)     {sprintf(info, "%4.3f KB", data_size/1024);}
    else if(data_size<1024*1024*1024){sprintf(info, "%4.3f MB", data_size/(1024*1024));}
    else                             {sprintf(info, "%4.3f GB", data_size/(1024*1024*1024));}

}
inline unsigned long rpcc_usec()
{
   struct timeval   val;
   gettimeofday(&val,NULL);
   return (val.tv_sec*1000000 + val.tv_usec);
}
int init_data(char *filename,char bRdOrWr)
{
  mode_t mode;

	if(bRdOrWr > 0) 
		mode = O_CREAT |  O_WRONLY;
	else	
	    mode = O_RDONLY;
	
  int fd;

  fd = open(filename,mode,S_IRWXU|S_IRWXG|S_IRWXO);

  if(fd<0) {
		printf("open %s failed\n",filename);
    return -1;
  }
  return fd;
}


void store_data(int fd, float *data, int len)
{
    int ret;
    ret = write(fd,data,len);

    if (ret < 0) {
        perror("write error:");
        exit(-1);
    } 
}  
/* 
 * 从文件中读入x和y数据 
 * */  
#ifdef USE_IMAGENET
void load_data(int fd, unsigned char *data,long len,long offset)
{
    int ret;
	  lseek(fd,offset,SEEK_SET);
    ret = read(fd,data,len);

    if (ret < 0) {
        perror("read error:");
        exit(-1);
    } 
}
int readDataFromFile(int rank, int numprocs,int start,int local_count,int dims,int data_size,unsigned char *filename,float *data){
    
  int rdfd = 0;
	struct stat thestat;
    
  rdfd = init_data(filename,0);
	if(rdfd < 0)
	{
     if(rank == 0) 
       printf("init_data data fileName error filename =%s\n",filename);
	   return;		
	}
	if(fstat(rdfd, &thestat) < 0) {
		 close(rdfd);
     if(rank == 0) 
        printf("fstat error %s\n",filename);
     return;
  }
	long size = thestat.st_size/sizeof(unsigned char);
  long total_data_size = data_size;
  total_data_size *= dims;
	if(size != total_data_size)
	{
    if(rank == 0) 
		  printf("data format and size error:%s %ld %ld %d\n",filename,size,total_data_size,size/dims);
		//return -1;
	}
	long len = local_count*sizeof(unsigned char);
  len *= dims;
  long offset = start*sizeof(unsigned char);
  offset *= dims;
  unsigned char * p_data = (unsigned char *)malloc(len);
  assert(p_data != NULL);
	load_data(rdfd,p_data,len,offset);
  sw_memcpy_c2f(p_data,data,len);
  free(p_data);
	close(rdfd);
	return 1;
}
void initial_cluster(unsigned char * filename,int rank,int numprocs,int dims,int data_size,int cluster_count,float *cluster_center,float *data){
    int i,j,k,s;     
    long max_size = data_size;  
    srand((unsigned int) time(NULL)); 
	  int local_count=data_size/numprocs + (rank<(data_size%numprocs));
    int start=rank*(data_size/numprocs) + (rank<(data_size%numprocs)?rank:(data_size%numprocs));
    int end = start + local_count;
    long index = 0,len = dims * sizeof(unsigned char);   
    floatv4 vsrc,vdst,vsum;
    int param_size = cluster_count * dims;
    int simd_size = 4,data_index = 0;
    int rdfd = init_data(filename,0);
		unsigned char *temp = (unsigned char*)malloc(len);
    float *c_data,arr[4],sum;
    for( i = 0; i < cluster_count; i++ )  
    {  
       index = rand() % max_size;
       c_data = cluster_center + i*dims;
	     if(index >= start && index < end){
         data_index = (index - start)*dims;
         for(j=0;j+simd_size-1<dims;j+=simd_size) {
           simd_load(vsrc,data+data_index+j);
           simd_store(vsrc,c_data+j);
         }  
		     for(;j<dims;j++)   
           c_data[j]= data[data_index+j];   
	     }
	     else
	     {
         index *= dims;
         index *= sizeof(unsigned char);
		     load_data(rdfd,temp,len,index);
		     for(j=0;j<dims;j++)   
           c_data[j]= temp[j];   
	     }
    }
		free(temp);
    close(rdfd);
    //printf("Init cluster center ok.rank = %d\n",rank);
}
#else
void load_data(int fd, float *data,long len,long offset)
{
    int ret;
	  lseek(fd,offset,SEEK_SET);
    ret = read(fd,data,len);

    if (ret < 0) {
        perror("read error:");
        exit(-1);
    }
}
int readDataFromFile(int rank, int numprocs,int start,int local_count,int dims,int data_size, unsigned char *filename, float* all_data)
{

  int rdfd = 0;
	struct stat thestat;

  rdfd = init_data(filename,0);

  printf("%s\n", filename);
	if(rdfd < 0)
	{
     if(rank == 0)
       printf("init_data data fileName error filename =%s\n",filename);
	   return -1;
	}
	if(fstat(rdfd, &thestat) < 0) {
		 close(rdfd);
     if(rank == 0)
        printf("fstat error %s\n",filename);
     return -1;
  }
	long size = thestat.st_size/sizeof(float);
  long total_data_size = data_size;
  total_data_size *= dims;
	if(size != total_data_size)
	{
    if(rank == 0)
		  printf("data format and size error:%s %ld %ld %d\n",filename,size,total_data_size,size/dims);
		//return -1;
	}
	/*long len = local_count;
  len *= dims;
  len *= sizeof(float);
  long offset = start;
  offset *= dims;
  offset *= sizeof(float);*/

	load_data(rdfd, all_data, thestat.st_size, 0); // 0 == offset
  int i = 0;
  /*if(rank == 0) printf("bcast all_data start...\n");
  caffe_mpi_bcast_f(all_data, total_data_size, 0, MPI_COMM_WORLD);
  if(rank == 0) printf("bcast all_data done...\n");*/  // it's not making sense
  printf("%x  %f\n", all_data + 1, *(all_data + 1));
  //*data = all_data + start * dims;//&(((float*)all_data)[start*dims]);
  printf("total_data_size=%d\n", thestat.st_size);
  close(rdfd);
	return 1;
}
void initial_cluster(unsigned char * filename,int rank,int numprocs,int dims,int data_size,int cluster_count,float *cluster_center,float *data){
    int i,j,k,s;
    long max_size = data_size;
    srand((unsigned int) time(NULL));
	  int local_count=data_size/numprocs + (rank<(data_size%numprocs));
    int start=rank*(data_size/numprocs) + (rank<(data_size%numprocs)?rank:(data_size%numprocs));
    int end = start + local_count;
    long index = 0,len = dims * sizeof(float);
    floatv4 vsrc,vdst,vsum;
    int param_size = cluster_count * dims;
    int simd_size = 4,data_index = 0;
    int rdfd = init_data(filename,0);
		float *temp = (float*)malloc(len);
    float *c_data,arr[4],sum;

    //int is_used[local_count];
    int * is_used = (int*)malloc(data_size * sizeof(int));
    for(i = 0; i < data_size; ++i) is_used[i] = 0;
    //printf("max_size=%d\n", max_size);
    for(i = 0; i < cluster_count; i++)
    {
      while(1) //index = (rand() % max_size))
      {
        index = (rand() % max_size);
        if(is_used[index] == 0)
        {
          //if(index == 0) printf("index==0\n");
          is_used[index] = 1;
          //if(index == 0) printf("index==0 %d\n", is_used[index]);
          break;
        }
      }

      //printf("init index=%d\n", index);
      c_data = cluster_center + i*dims;
      if(index >= start && index < end){
        data_index = (index - start)*dims;
        for(j=0;j+simd_size-1<dims;j+=simd_size) {
          simd_load(vsrc,data+data_index+j);
          simd_store(vsrc,c_data+j);
		    }
        for(;j<dims;j++)
        {
          c_data[j] = data[data_index + j];
        }
      }
	    else
	    {
        index *= dims;
        index *= sizeof(float);
		    load_data(rdfd,temp,len,index);
        for(j=0;j+simd_size-1<dims;j+=simd_size) {
          simd_load(vsrc,temp+j);
          simd_store(vsrc,c_data+j);
        }
		    for(;j<dims;j++)
          c_data[j]= temp[j];
	    }
    }

    free(is_used);
		free(temp);
    close(rdfd);
}
#endif

void writeClusterDataToFile(int round,int dims, int cluster_count, float * cluster_center, int data_size, int * data_group, float * cluster_distance_count, float * non_cluster_distance_count, unsigned long all_time, unsigned long round_time, unsigned long cluster_time, float * all_radius){
  int i,j;
  char filename[200];
  FILE* file_write, *file_read;
  float min_radius = 0, pre_radius = FLT_MAX;
  sprintf(filename, "./data/round_%d_cluster.dat", cluster_count);  //sprintf(filename, "./data/round%d_%d_cluster.dat", round, cluster_count);
  /*if(NULL != (file_read = fopen(filename, "r")))
  {
    fscanf(file_read, "%f", &pre_radius);
    printf("min_radius=%f\n", pre_radius);
  }
  fclose(file_read);*/

  char filename1[200];
  FILE* file_write_time;
  sprintf(filename1, "./data/time_%d_cluster.dat", cluster_count);
  if( NULL == (file_write_time = fopen(filename1, "w")) )
  {
    printf("file open(%s) error!", filename1);
    exit(0);
  }

  if( NULL == (file_write = fopen(filename, "a+"))){
    printf("file open(%s) error!", filename);
    exit(0);
  }


  if(cluster_distance_count != NULL){
    for(i = 0; i < cluster_count; i++)
    {
      printf("%f ", cluster_distance_count[i]);
    }
    printf("\n");

    for(i = 0; i < cluster_count; i++)
    {
      printf("%f ", non_cluster_distance_count[i]);
    }
    printf("\n");

    int cluster_cnt[cluster_count]; // the count of cluster
    for(i = 0; i < cluster_count; ++i)  cluster_cnt[i] = 0;
    for(i = 0; i < data_size; ++i)
    {
      cluster_cnt[data_group[i]] ++;
      //fprintf(file, "%d %d\n", i, data_group[i]);
    }
    /*float distance_count = 0.0;
    for(i = 0; i < cluster_count; ++i)
    {
      //printf("cluster_cnt %d  %d\n", i, cluster_cnt[i]);
      fprintf(file_write, "cluster_cnt %d  %d\n", i, cluster_cnt[i]);
      distance_count += non_cluster_distance_count[i];
    }

    printf("distance_count = %f\n", distance_count);*/

    float evaluation_value = 0;
    for(i = 0; i < cluster_count; i++)
    {
      if(cluster_cnt[i] == 0)
        cluster_distance_count[i] = 0;
      else
        cluster_distance_count[i] = cluster_distance_count[i] / cluster_cnt[i];
      if(cluster_cnt[i] == data_size)
      {
        //printf("cluster_count is equal to 1, I didn\'t deal with it.\n");
        //continue;
        non_cluster_distance_count[i] = 0;
      }
      else
        non_cluster_distance_count[i] = non_cluster_distance_count[i] / (data_size - cluster_cnt[i]);
      evaluation_value += abs(non_cluster_distance_count[i] - cluster_distance_count[i]); // * (non_cluster_distance_count[i] - cluster_distance_count[i]);
    }
    printf("evaluation_value=%f\n", evaluation_value);
    fprintf(file_write, "%f\n", evaluation_value);
  }
  printf("all_time=%lu us\n", all_time);
  printf("round_time=%lu us\n", round_time);
  printf("cluster_time=%lu us\n", cluster_time);
  fprintf(file_write_time, "all_time=%lu\n", all_time);
  fprintf(file_write_time, "round_time=%lu\n", round_time);
  fprintf(file_write_time, "cluster_time=%lu\n", cluster_time);
  //float min_radius = FLT_MAX;
  int flag = 0;
  if(all_radius != NULL)
  {
    int cluster;
    float * cluster_radius = (float *) malloc(cluster_count * sizeof(float));
    assert(cluster_radius != NULL);
    for(i = 0; i < cluster_count; ++i)
    {
      cluster_radius[i] = FLT_MAX;
    }
    for(i = 0; i < data_size; ++i)
    {
      cluster = data_group[i];
      //printf("i = %d  cluster = %d\n", i, cluster);
      if(all_radius[i] < cluster_radius[cluster])
      {
        cluster_radius[cluster] = all_radius[i];
      }
    }
    for(i = 0; i < cluster_count; ++i)
    {
      if(min_radius < cluster_radius[i])
      {
        //flag = 1;
        min_radius = cluster_radius[i];
      }
    }
    printf("min_radius = %f\n", min_radius);
    /*if(min_radius > pre_radius)
    {
      //min_radius = pre_radius;
    }
    else
    {
      flag = 1;
    }*/
    free(cluster_radius);
  }
  //printf("pre_radius=%f\n", pre_radius);
  //printf("min_radius=%f\n", min_radius);
  fprintf(file_write, "%f\n", min_radius);
  fclose(file_write);
  printf("end\n");
  if(flag == 1)
  {
    FILE * file_write_center;
    char filename_center[200];
    sprintf(filename_center, "./data/center_%d_cluster.dat", cluster_count);
    if( NULL == (file_write_center = fopen(filename_center, "w")) ){
      printf("file open(%s) error!", filename_center);
      exit(0);
    }

    for(i = 0; i < cluster_count; ++i)
    {
      for(j = 0; j < dims; ++j)
      {
        fprintf(file_write_center, "%f ", cluster_center[i * dims + j]);
      }
      fprintf(file_write_center, "\n");
    }
    fclose(file_write_center);
  }
  fclose(file_write_time);
}

static void *do_slave_kmeans(void * lParam)
{
     KmeansPara *param = (KmeansPara*)lParam;
     if(!athread_get_num_threads()) athread_init();
		 athread_spawn(sw_slave_kmeans_f,(void*)param);
		 athread_join();
     pthread_exit(0);
}

void write_data1(int data_size, int dims, float * data)
{
  char filename[20];
  sprintf(filename, "./data/test_data.dat");

  FILE * file;
  if( NULL == (file = fopen(filename, "w")) ){
    printf("file open(%s) error!", filename);
    exit(0);
  }

  int i, j;
  for(i = 0; i < data_size; ++i)
  {
    for(j = 0; j < dims; ++j)
    {
      fprintf(file, "%f,", data[i * dims + j]);
    }
    fprintf(file, "\n");
  }

  fclose(file);
}

void write_data_group(int data_size, int * data_group)
{
  int i, j;
  FILE * file;

  if(NULL == (file = fopen("./data/remote_sense.dat", "w")))
  {
    printf("file open error\n");
    exit(0);
  }
  int num = 254;
  for(i = 0; i < num; ++i)
  {
    for(j = 0; j < num; ++j)
      fprintf(file, "%d ", *(data_group + i * num + j));
    fprintf(file, "\n");
  }
  fclose(file);
}
int main(int argc, char* argv[]){
  unsigned long all_time;
  all_time = rpcc_usec();
  unsigned long cluster_time;
  cluster_time = all_time;
  if( argc != 9){
    printf("This application need other parameter to run:"
                "\n\t\tthe first is the file name that contain data"  
                "\n\t\tthe second is the size of data set,"  
                "\n\t\tthe third indicate the cluster_count"  
                "\n\t\tthe fourth indicate the data dimension"  
                "\n\t\tthe fifth indicate whether save run result"  
                "\n\t\tthe sixth indicate the distance file"
                "\n\t\tthe seventh indicate whether caculate evaluation function"
                "\n\t\tthe eighth indicate whether it's remote sense data\n");
        exit(0);
  }
  athread_init();
  pthread_t pthread_handler[NUM_THREADS];
  unsigned char filename[200];
  strcpy(filename, argv[1]);
  int data_size = atoi(argv[2]);
  int cluster_count = atoi(argv[3]);
  int dims=atoi(argv[4]);
  int save_status = atoi(argv[5]);
  unsigned char distance_filename[50];
  strcpy(distance_filename, argv[6]);
  int caculate_status = atoi(argv[7]);
  int sense_data = atoi(argv[8]);
  int rank = 0,numprocs = 1;
  //assert(CLUSTER_COUNT == cluster_count);
  //assert(DIMS == dims);

#ifdef USE_MPI
	MPI_Status status;
  MPI_Request request;
  MPI_Init(&argc,&argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm, &numprocs);
  if(rank == 0)
  {
    printf("numprocs=%d \n", numprocs);
  }
  init_net_topology();

  long buff_size = cluster_count*sizeof(float);
  buff_size *= dims;
  buff_size += cluster_count*sizeof(int);

  unsigned char * pack_buff0 = (unsigned char *)malloc(buff_size);
  assert(pack_buff0 != NULL);
  unsigned char * pack_buff1 = (unsigned char *)malloc(buff_size);
  assert(pack_buff1 != NULL);
#endif   
  int local_count=data_size/numprocs + (rank<(data_size%numprocs));
  int start=rank*(data_size/numprocs) + (rank<(data_size%numprocs) ? rank : (data_size%numprocs));
  int end = start + local_count;
  long total_data_size = local_count*sizeof(float);
  total_data_size *= dims;
  //printf("total_data_size=%d", total_data_size);
	int * data_group = (int*)malloc(local_count * sizeof(int));
  assert(data_group != NULL);
  float * radius = (float *)malloc(local_count * sizeof(float));
  assert(radius != NULL);
  float * all_radius = (float *)malloc(data_size * sizeof(float));
  assert(all_radius != NULL);

  float *data = NULL; // = (float*)malloc(total_data_size);
  //assert(data != NULL);

  float * all_data = (float*)malloc(data_size * dims * sizeof(float));
  assert(all_data != NULL);
  int ret = readDataFromFile(rank, numprocs,start,local_count,dims,data_size,filename, all_data);
  data = (all_data + start * dims);
  assert(data != NULL);
  /*if(data != NULL)
  {
    int tmp;
    for(tmp = 0; tmp < dims; ++tmp)
      printf("%f ", data[tmp]);
    //write_data1(data_size, dims, data);
  }*/
  //File Format Error,Or read file error
	if(ret < 1)
	{
		free(data);
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return -1;
	}
	int flag = 1,center_tag = 2,exit_tag = 3,count = 0,i=0,j=0,k=0;
  int is_continue = 0,round = 0,offset=0, block_size=0;
  long param_size = cluster_count;
  param_size *= dims;
  float *cluster_center = NULL,*cluster_center_new = NULL,*cluster_center_old = NULL;
  int * cluster_center_num = NULL;
#ifdef USE_4CG
  cluster_center = (float*)malloc(sizeof(float) * param_size * NUM_THREADS);
	cluster_center_num = (int*)malloc(sizeof(int) * cluster_count *NUM_THREADS);
#else
  cluster_center = (float*)malloc(sizeof(float) * param_size);
#endif

#ifdef USE_MPI
  cluster_center_new = (float*)pack_buff0;
  cluster_center_num = (int*)(pack_buff0 + param_size*sizeof(float));
  cluster_center_old = cluster_center;
#else
	cluster_center_num = (int*)malloc(sizeof(int) * cluster_count);
  cluster_center_new = (float*)malloc(sizeof(float) * param_size);
  cluster_center_old = (float*)malloc(sizeof(float) * param_size);
  /*if(rank == 0)
  {
    int tmp;
    for(tmp = 0; tmp < param_size; ++tmp)
    {
      printf("%f ", cluster_center_new[tmp]);

      if(tmp % 3 == 2)
      {
        printf("\n");
      }
    }
  }*/
#endif
  assert(cluster_center != NULL);
  assert(cluster_center_num != NULL);
  assert(cluster_center_new != NULL);
  assert(cluster_center_old != NULL);

	//Wait Data Read complete
#ifdef USE_MPI
  MPI_Barrier(comm);
#endif
	if(rank == 0){
		//初始化中心点及其索引，中心点可能不在一个节点上，通过重新读文件获取
    /*int tmp;
    for(tmp = 0; tmp < param_size; ++tmp)
    {
      printf("%f ", cluster_center_new[tmp]);
      if(tmp % 3 == 2)
      {
        printf("\n");
      }
    }*/
    initial_cluster(filename,rank,numprocs,dims,data_size,cluster_count,cluster_center_new,data);
	}
#ifdef USE_MPI
	//bcast the center point
  if(rank == 0) printf("bcast center starting...\n");
  caffe_mpi_bcast_f(cluster_center_new,param_size ,0, MPI_COMM_WORLD);
  if(rank == 0) printf("bcast center done.\n");
#endif
    //1、各个计算几点把坐标和值发送到0号进程，0号进程重新计算中心点，并广播
    //2、其他进程接收到最新中心点，判断是否满足形成条件，如果满足退出，否则，继续第一步.
#ifdef USE_4CG
  KmeansPara param[NUM_THREADS];
  for(i=0; i<NUM_THREADS; i++)
  {
     block_size = local_count/NUM_THREADS + (i<(local_count%NUM_THREADS));
     offset = i*(local_count/NUM_THREADS) + (i<(local_count%NUM_THREADS) ? i : (local_count%NUM_THREADS));
	   param[i].dims = dims;
 	   param[i].cluster_count = cluster_count;
	   param[i].cluster_center = cluster_center_new;
	   param[i].cluster_center_num = cluster_center_num + i*cluster_count;
	   param[i].data_size = block_size;
	   param[i].data = data + offset*dims;
	   param[i].cluster_center_out = cluster_center + i*param_size;
  }
#else
  KmeansPara param;
	param.dims = dims;
	param.data_size = local_count;
	param.cluster_count = cluster_count;
	param.cluster_center = cluster_center_new;
	param.cluster_center_num = cluster_center_num;
	param.data = data;
  param.rank = rank;
	param.cluster_center_out = cluster_center;

#endif

  float * cluster_distance_count = (float *) malloc(cluster_count * sizeof(float));
  float * non_cluster_distance_count = (float *) malloc(cluster_count * sizeof(float));

  EvaluationPara evaluation_para;
  evaluation_para.out_filename = distance_filename;
  evaluation_para.rank = rank;
  evaluation_para.dims = dims;
  evaluation_para.data_size = local_count;
  evaluation_para.all_data_size = data_size;
  evaluation_para.cluster_count = cluster_count;
  evaluation_para.cluster_center = cluster_center_new;
  evaluation_para.data = data;
  evaluation_para.data_group = data_group;
  evaluation_para.all_data = all_data;
  evaluation_para.data_start = start;
  evaluation_para.cluster_distance_count = cluster_distance_count;
  evaluation_para.non_cluster_distance_count = non_cluster_distance_count;
  evaluation_para.radius = radius;
#ifdef PRINT_ONE_ROUND_TIME
  //if(rank ==0)writeClusterDataToFile(6666,dims,cluster_count,cluster_center_new);
  if(rank == 0){
    printf("starting process ...\n");
    /*int tmp;
    for(tmp = 0; tmp < param_size; ++tmp)
    {
      printf("%f ", cluster_center_new[tmp]);
      if(tmp % 3 == 2)
      {
        printf("\n");
      }
    }*/
  }
  unsigned long run_time = 0,round_run_time = 0,mpi_one_round_time = 0;
  run_time = rpcc_usec();
#endif
	for(round =0 ;round < MAX_ROUND;round++)
	{
#ifdef PRINT_ONE_ROUND_TIME
    round_run_time = rpcc_usec();
#endif
#ifdef USE_4CG
    sw_memset_f(cluster_center,NUM_THREADS*param_size);
    for(i=0; i<NUM_THREADS; i++)
       pthread_create(&pthread_handler[i],NULL, do_slave_kmeans, (void*)(&param[i]));
    for(i=0; i<NUM_THREADS; i++)
       pthread_join(pthread_handler[i], NULL);
    sw_memcpy_f(cluster_center_new,cluster_center_old,param_size);
    sw_add_4cg_f(cluster_center,cluster_center+param_size,cluster_center+2*param_size,
        cluster_center+3*param_size,cluster_center_new,param_size);

    sw_add_4cg_i(cluster_center_num,cluster_center_num+cluster_count,cluster_center_num+2*cluster_count,
        cluster_center_num+3*cluster_count,cluster_center_num,cluster_count);
#else
    sw_memset_f(cluster_center,param_size);

    /*if(rank == 0)
    {
      int m;
      printf("sw_slave_kmeans_f start\n");
      for(m = 0; m < cluster_count * dims; ++m)
      {
        //printf("%f\n", cluster_center[m]);
        if(cluster_center[m] > 1e-4) printf("it's not 0\n");
      }
    }*/
    athread_spawn(sw_slave_kmeans_f,(void*)(&param));
		athread_join();

    /*if(rank == 0)
    {
      int m;
      int count = 0;
      for(m = 0; m < cluster_count; ++m)
      {
        count += cluster_center_num[m];
        if(m < 10)
          printf("%d %d \n", m, cluster_center_num[m]);
      }
      printf("cnt=%d\n", count);
    }*/
    /*if(rank == 0)
    {
      int tmp1, tmp2;
      float tmp3;
      for(tmp1 = 0; tmp1 < param.dims; ++tmp1)
      {
        tmp3 = 0;
        for(tmp2 = 0; tmp2 < param.cluster_count; ++tmp2)
        {
          tmp3 += param.cluster_center_out[tmp2 * param.dims + tmp1];
        }
        printf("%f ", tmp3);
      }
      printf("\n");
    }*/
#ifdef USE_MPI
    sw_memcpy_f(cluster_center_new,(float*)pack_buff1,param_size);
    sw_memcpy_f(cluster_center,cluster_center_new,param_size);
    sw_memcpy_f((float*)pack_buff1,cluster_center,param_size);
#else
    sw_memcpy_f(cluster_center_new,cluster_center_old,param_size);
    sw_memcpy_f(cluster_center,cluster_center_new,param_size);
#endif

#endif
#ifdef USE_MPI
#ifdef PRINT_ONE_ROUND_TIME
   mpi_one_round_time = rpcc_usec();
#endif
    //sw_memcpy_f(cluster_center_new,pack_buff0,param_size);
    //sw_memcpy_i(cluster_center_num,pack_buff0+param_size*sizeof(float),cluster_count);
    /*
    if(root_index > 1)
    {
      for(i=0;i<root_index;i++)
      {
        if(prime_comm[i] != MPI_COMM_NULL)
          caffe_mpi_reduce(pack_buff0,pack_buff1,buff_size,param_size,cluster_count,0,prime_comm[i]);
      }
      //all reduce between mid 
      caffe_mpi_supernode_allreduce(pack_buff1,pack_buff0,buff_size,param_size,cluster_count,root_index,root_ranks,MPI_COMM_WORLD);
      //cluster_center = cluster_center/numprocs
      sw_div_f(pack_buff0,cluster_center_new,cluster_center_old,pack_buff0+sizeof(float)*param_size,cluster_count,dims);
      //bcast in mid
      for(i=0;i<root_index;i++)
      {
        if(prime_comm[i] != MPI_COMM_NULL)
           caffe_mpi_bcast_f(cluster_center_new, param_size,0, prime_comm[i]);
      }
    }
    else
    {
      caffe_mpi_reduce(pack_buff0,pack_buff1,buff_size,param_size,cluster_count,0, MPI_COMM_WORLD);
      //cluster_center = cluster_center_tmp/numprocs
      sw_div_f(pack_buff1,cluster_center_new,cluster_center_old,pack_buff1+param_size*sizeof(float),cluster_count,dims);
      caffe_mpi_bcast_f(cluster_center_new, param_size,0, MPI_COMM_WORLD);
    }*/
    //if(rank == 0) printf("rank = %d reduce begin\n", rank);
    
    caffe_mpi_reduce(pack_buff0,pack_buff1,buff_size,param_size,cluster_count,0, MPI_COMM_WORLD);
    //cluster_center = cluster_center_tmp/numprocs
    //if(rank == 0) printf("rank = %d reduce end\n", rank);
    
    sw_div_f(pack_buff1,cluster_center_new,cluster_center_old,pack_buff1+param_size*sizeof(float),cluster_count,dims);
    caffe_mpi_bcast_f(cluster_center_new, param_size,0, MPI_COMM_WORLD);
    //MPI_Allreduce(pack_buff0,pack_buff1,(buff_size>>2),MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
    //sw_div_f(pack_buff1,cluster_center_new,cluster_center_old,pack_buff1+param_size*sizeof(float),cluster_count,dims);

#ifdef PRINT_ONE_ROUND_TIME
   mpi_one_round_time = rpcc_usec() - mpi_one_round_time;
#endif
#else
   sw_div_f(cluster_center_new,cluster_center_new,cluster_center_old,cluster_center_num,cluster_count,dims);
#endif
   if(rank == 0){
      is_continue = param_size;
      //simd_vfcmpeq
      for(i = 0;i < param_size;i++){
        if(fabs(cluster_center_new[i] - cluster_center_old[i])< 1e-4)
          is_continue --;
        else break;
      }
#ifdef PRINT_ONE_ROUND_TIME
   round_run_time = rpcc_usec() - round_run_time;
   if(rank == 0)
   printf("round = %d  run_time = %lu us  mpi_run_time = %lu us  root_index = %d\n",round,round_run_time,mpi_one_round_time,root_index);//  /1000000.0
   //if(save_status) writeClusterDataToFile(round,dims,cluster_count,commPara.cluster_center);
#endif
#ifdef USE_MPI
		 for(i=1;i<numprocs;i++)
         MPI_Isend(&is_continue,1, MPI_INT,i,exit_tag,MPI_COMM_WORLD, &request);
#endif   
     if(is_continue < 1)
     {
        cluster_time = rpcc_usec() - cluster_time;
        run_time = rpcc_usec() - run_time;
        float total_data_size = data_size*dims*sizeof(float);
        float dTime = run_time/1000000.0;
        char cInf[80],cInf1[80];
        float dTmp = total_data_size/dTime;
        format_result(total_data_size,cInf);
        format_result(dTmp,cInf1);
        printf("KMeans speed (pkt = %s): time= %.6fs speed = %s/s avg_run_time = %.6f\n",cInf, dTime,cInf1,dTime/(round+1));
        /*athread_spawn(get_data_group,(void*)(&evaluation_para)); // get the data_group
        athread_join();
        printf("get_data_group finished\n");

        athread_spawn(caculate_evaluation_function, (void*)(&evaluation_para));
        athread_join();
        printf("caculate_evaluation_function done  %f\n", cluster_distance_count[0]);

        float * cluster_distance_count_dst = (float *) malloc(cluster_count * sizeof(float));
        float * non_cluster_distance_count_dst = (float *)malloc(cluster_count * sizeof(float));

        MPI_Reduce(cluster_distance_count, cluster_distance_count_dst, cluster_count, MPI_FLOAT, MPI_SUM, 0, comm);
        MPI_Reduce(non_cluster_distance_count, non_cluster_distance_count_dst, cluster_count, MPI_FLOAT, MPI_SUM, 0, comm);
        writeClusterDataToFile(round,dims,cluster_count,cluster_center_new, local_count, data_group, cluster_distance_count, non_cluster_distance_count);*/
        break;
     }
     if(round == MAX_ROUND -1){
       //writeClusterDataToFile(round,dims,cluster_count,cluster_center_new, 0, NULL);
       printf("Not find!\n");
     }
     else {
       if(save_status) printf("I comment it\n");//writeClusterDataToFile(round,dims,cluster_count,cluster_center_new, 0, NULL);
     }
   }
#ifdef USE_MPI
   else
   {
			MPI_Irecv(&is_continue,1,MPI_INT,0,exit_tag,comm,&request);
			MPI_Wait(&request,&status);
			if(is_continue < 1)  break;
   }
#endif
  }
  int * all_data_group = (int*)malloc(data_size * sizeof(int));
  assert(all_data_group != NULL);
  float * cluster_distance_count_dst = (float *) malloc(cluster_count * sizeof(float));
  assert(cluster_distance_count_dst != NULL);
  float * non_cluster_distance_count_dst = (float *)malloc(cluster_count * sizeof(float));
  assert(non_cluster_distance_count_dst != NULL);
  if(is_continue < 1 && caculate_status > 0){
    /*if(rank == 0)
    {
      int *cluster_center_num1 = (int*)(pack_buff1 + param_size*sizeof(float));
      int m, n;
        for(m = 0; m < cluster_count; ++m)
        {
          //for(n = 0; n < dims; ++n)
          //{
          printf("%d %d \n", m, cluster_center_num1[m]);
          //}
          //printf("\n");
       }
    }*/
    athread_spawn(get_data_group,(void*)(&evaluation_para)); // get the data_group
    athread_join();

    printf("get_data_group finished\n");
    //int * all_data_group = (int*)malloc(data_size * sizeof(int));
    //assert(all_data_group != NULL);
    if(rank != 0)
    {
      MPI_Send(data_group, local_count, MPI_INT, 0, 0, comm);
    }
    else
    {
      int other_local_count, other_start;
      for(i = 0; i < local_count; ++i)
      {
        all_data_group[i] = data_group[i];
      }
      for(i = 1; i < numprocs; ++i)
      {
        other_local_count = data_size / numprocs + (i < data_size % numprocs);
        start = i * (data_size / numprocs) + ((i < data_size % numprocs)? i: data_size % numprocs);
        MPI_Recv(all_data_group + start, other_local_count, MPI_INT, i, 0, comm, MPI_STATUS_IGNORE);
      }

    }
    MPI_Bcast(all_data_group, data_size, MPI_INT, 0, comm);
    evaluation_para.all_data_group = all_data_group;

    if(caculate_status == 1)
    {
      writeClusterDataToFile(round,dims,cluster_count,cluster_center_new, data_size, NULL, NULL, NULL, all_time, round_run_time, cluster_time, NULL);
      if(sense_data > 0)
      {
        write_data_group(data_size, all_data_group);
      }
    }
    else
    {
      int fd_distance = open(distance_filename, O_RDONLY, S_IRWXU|S_IRWXG|S_IRWXO);
      unsigned long distance_size = (unsigned long) data_size * (unsigned long) data_size;
      distance_size *= (unsigned long) sizeof(float);
      printf("distance_size=%lu\n", distance_size);
      float * distance = (float *)malloc(distance_size);

      load_data(fd_distance, distance, distance_size, 0);
      close(fd_distance);
      printf("test\n");
      evaluation_para.distance = distance;
      athread_spawn(caculate_radius, (void*)(&evaluation_para));
      athread_join();

      free(distance);
      if(rank != 0)
      {
        MPI_Send(radius, local_count, MPI_FLOAT, 0, 0, comm);
      }
      else
      {
        int other_local_count, other_start;
        for(i = 0; i < local_count; ++i)
        {
          all_radius[i] = radius[i];
        }
        for(i = 1; i < numprocs; ++i)
        {
          other_local_count = data_size / numprocs + (i < data_size % numprocs);
          start = i * (data_size / numprocs) + ((i < data_size % numprocs)? i : data_size % numprocs);
          MPI_Recv(all_radius + start, other_local_count, MPI_FLOAT, i, 0, comm, MPI_STATUS_IGNORE);
        }
      }
      //MPI_Reduce(cluster_distance_count, cluster_distance_count_dst, cluster_count, MPI_FLOAT, MPI_SUM, 0, comm);
      //MPI_Reduce(non_cluster_distance_count, non_cluster_distance_count_dst, cluster_count, MPI_FLOAT, MPI_SUM, 0, comm);

      if(rank == 0)
      {
        //printf("local_count=%d\n", local_count);
        /*for(i = 0; i < local_count; ++i)
        {
          printf("%d %f\n", i, all_radius[i]);
        }*/
        all_time = rpcc_usec() - all_time;
        printf("all_time=%lu\n", all_time);
        writeClusterDataToFile(round,dims,cluster_count,cluster_center_new, data_size, all_data_group, NULL, NULL, all_time, round_run_time, cluster_time, all_radius);
      }
    }
  }
  else if(is_continue < 1 && caculate_status == 0)
  {
    if(rank == 0)
      writeClusterDataToFile(round, dims, cluster_count, cluster_center_new, data_size, NULL, NULL, NULL, all_time, round_run_time, cluster_time, NULL);
  }

	athread_halt();
  if(cluster_center)free(cluster_center);
  if(cluster_center_new)free(cluster_center_new);
  if(cluster_center_num)free(cluster_center_num);
  if(cluster_center_old)free(cluster_center_old);
	if(data)free(data);
  if(data_group) free(data_group);
  if(all_data) free(all_data);
  if(all_data_group) free(all_data_group);
  if(cluster_distance_count) free(cluster_distance_count);
  if(non_cluster_distance_count) free(non_cluster_distance_count);
  if(cluster_distance_count_dst) free(cluster_distance_count_dst);
  if(non_cluster_distance_count_dst) free(non_cluster_distance_count_dst);
#ifdef USE_MPI
  if(pack_buff0) free(pack_buff0);
  if(pack_buff1) free(pack_buff1);
  if(radius) free(radius);
  if(all_radius) free(all_radius);
  //release_net_topology();
	MPI_Finalize();
#endif
  return 0;
}
