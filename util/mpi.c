#include "util/mpi.h"

int caffe_mpi_supernode_allreduce_f( void *sendbuf, void *recvbuf, int count,
    int root_count,int* ranks, MPI_Comm comm  ){
  int i = 0,index = -1;
  int pof2 = 1,dest=0,tag=1;
  int mpi_rank;
  MPI_Request recv_req,send_req;
  MPI_Status  recv_statue;
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
        MPI_Wait(&recv_req,&recv_statue);
        sw_add_f((float*)recvbuf,(float*)sendbuf,(float*)sendbuf,count);
     }
     pof2 = pof2 << 1;
  }
  return 0;
}
int caffe_mpi_supernode_allreduce_d( void *sendbuf, void *recvbuf, int count, int root_count,int* ranks, MPI_Comm comm  ){
  int i = 0,index = -1;
  int pof2 = 1,dest=0,tag=1;
  int mpi_rank;
  MPI_Request recv_req,send_req;
  MPI_Status  recv_statue;
  MPI_Comm_rank(comm, &mpi_rank);
  for(i = 0;i < root_count;i++)
  {
     if(mpi_rank != ranks[i])
       continue;
     index = i;
     break;
  }
  if(index < 0) return 0;
  sw_memcpy_d((double*)sendbuf,(double*)recvbuf,count);
  while(pof2 < root_count){
     dest = index ^ pof2;
     if(dest < root_count){
        MPI_Irecv(sendbuf, count,MPI_DOUBLE, ranks[dest], tag,comm,&recv_req);
        MPI_Isend(recvbuf, count,MPI_DOUBLE, ranks[dest], tag,comm,&send_req);
        MPI_Wait(&recv_req,&recv_statue);
        sw_add_d((double*)recvbuf,(double*)sendbuf,(double*)sendbuf,count);
     }
     pof2 = pof2 << 1;
  }
  return 0;
}
int caffe_mpi_reduce_f( void *sendbuf, void *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm  ){
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

int caffe_mpi_reduce_d( void *sendbuf, void *recvbuf, int count,MPI_Op op, int root, MPI_Comm comm  ){
    int comm_size,rank;
    MPI_Status status;
    MPI_Request send_req,recv_req;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &rank);
    int mask = 0x1,source=0,tag = 10;
    int relrank = (rank - root + comm_size) % comm_size;

    sw_memcpy_d((double*)sendbuf,(double*)recvbuf,count);
    double * tmp_buff = (double*)malloc(count * sizeof(double));
    assert(tmp_buff != NULL);
    while(mask < comm_size){
      // Receive
      if ((mask & relrank) == 0) {
        source = (relrank | mask);
        if (source < comm_size) {
          source = (source + root) % comm_size;
	        MPI_Irecv(tmp_buff,count,MPI_DOUBLE,source,tag,comm,&recv_req);
          MPI_Wait(&recv_req,&status);
          sw_add_d((double*)tmp_buff,(double*)recvbuf,(double*)recvbuf,count);
        }
      }
      else {
         //I've received all that I'm going to.  Send my result to my parent 
         source = ((relrank & (~ mask)) + root) % comm_size;
	       MPI_Isend(recvbuf, count,MPI_DOUBLE, source, tag,comm,&send_req);
         break;
      }
      mask = mask << 1;
    }
    free(tmp_buff);
    return 0;
}

int caffe_mpi_bcast_f( void *buffer, int count, int root, MPI_Comm comm ) {
  return MPI_Bcast(buffer, count, MPI_FLOAT, root, comm);
 /* 
  int comm_size,rank;
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

int caffe_mpi_bcast_d( void *buffer, int count, int root,MPI_Comm comm ) {
  return MPI_Bcast(buffer, count, MPI_DOUBLE, root, comm);
}

