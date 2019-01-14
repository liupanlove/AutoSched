#ifndef UTIL_MPI_H_
#define UTIL_MPI_H_

#include <mpi.h>

int caffe_mpi_supernode_allreduce_f( void *sendbuf, void *recvbuf, int count, int root_count,int* ranks, MPI_Comm comm  );
int caffe_mpi_supernode_allreduce_d( void *sendbuf, void *recvbuf, int count, int root_count,int* ranks, MPI_Comm comm  );

int caffe_mpi_reduce_f( void *sendbuf, void *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm  );
int caffe_mpi_reduce_d( void *sendbuf, void *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm  );

int caffe_mpi_bcast_f( void *buffer, int count, int root, MPI_Comm comm );
int caffe_mpi_bcast_d( void *buffer, int count, int root, MPI_Comm comm );

#endif
