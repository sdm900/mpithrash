//
// Code developed by Dr Stuart Midgley sdm900@gmail.com
//
// MPITHRASH is an MPI code which produces an large unpredictable MPI
// load on the network and MPI subsystem.  It is designed to test the
// scalability of the high performance interconnect for a general work
// load.
//

//
// Building
// --------
// Compile the code for the host platform, eg.
//       cc -fast mpithrash.c -o mpithrash -lmpi
//

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

//
// maxmsgsize   is the maximum message size
// sendnmsgs    is the number of messages to send per process
// recvnmsgs    is the number of messages to receive simultaneously
//
#define maxmsgsize 65536
#define sendnmsgs 1000
#define recvnmsgs 20
#define iterations 10

#define min(a,b) (a<b?a:b)

int main(int argc, char** argv)
{
  int irk, isz, ier, dest;
  long i, j, k, msgsize, psenddps;
  double totalsentdps, senddps, time;
  int *sendcnt, *totsendcnt;
  double **a, **b;
  MPI_Request sendreq[sendnmsgs], recvreq[recvnmsgs];
  MPI_Status sendstatus[sendnmsgs], recvstatus[recvnmsgs];

  // Initialise MPI
  ier = MPI_Init(&argc, &argv);
  if (ier != 0) return 1;

  ier = MPI_Comm_rank(MPI_COMM_WORLD, &irk);
  ier = MPI_Comm_size(MPI_COMM_WORLD, &isz);

  // Ensure different random numbers on each process
  srand(isz+irk*4643);
  srand48(isz+irk*4643);
  
  // Allocate all buffers and initialise variables
  sendcnt = (int*) calloc(isz, sizeof(int));
  totsendcnt = (int*) calloc(isz, sizeof(int));
  a = (double**)malloc(sendnmsgs*sizeof(double*));
  for (i=0; i<sendnmsgs; i++) a[i] = (double*)calloc(maxmsgsize, sizeof(double));
  b = (double**)malloc(recvnmsgs*sizeof(double*));
  for (i=0; i<recvnmsgs; i++) b[i] = (double*)calloc(maxmsgsize, sizeof(double));

  senddps = 0.0;
  time = 0.0;

  // Loop over a number of iterations to get better timing results
  for (k=1; k<=iterations; k++)
    {
      psenddps = 0.0;
      for (i=0; i<isz; i++) sendcnt[i] = 0;
      for (i=0; i<sendnmsgs; i++) sendreq[i]=MPI_REQUEST_NULL;
      for (i=0; i<recvnmsgs; i++) recvreq[i]=MPI_REQUEST_NULL;

      // Send 1000 messages/process of random size to a random destination
      for (i=0; i<sendnmsgs; i++)
	{
	  dest = rand()%isz;
	  msgsize = lrand48()%maxmsgsize;
	  // Messages are sent synchonously so that they do not get
	  // transported over the network until the recieve is
	  // posted
	  ier = MPI_Issend(&a[i][0], msgsize, MPI_DOUBLE, dest, 100, MPI_COMM_WORLD, &sendreq[i]);
	  sendcnt[dest]++;
	  psenddps += msgsize;
	}

      // Global reduction so that each proces knows what messages
      // have been sent.  Each process can then post the correct
      // number of recieves
      senddps += (double) psenddps;
      ier = MPI_Allreduce(sendcnt, totsendcnt, isz, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

      // Start timing
      ier = MPI_Barrier(MPI_COMM_WORLD);
      time -= MPI_Wtime();

      // Post recieves in blocks of 20
      for (i=0; i<totsendcnt[irk]; i+=recvnmsgs)
	{
	  for (j=i; j<min(i+recvnmsgs, totsendcnt[irk]); j++)
	    ier = MPI_Irecv(&b[j-i][0], maxmsgsize, MPI_DOUBLE, MPI_ANY_SOURCE, 100, MPI_COMM_WORLD, &recvreq[j-i]);

	  ier = MPI_Waitall(recvnmsgs, recvreq, recvstatus);
	}

      // Finish up
      ier = MPI_Waitall(sendnmsgs, sendreq, sendstatus);
      ier = MPI_Barrier(MPI_COMM_WORLD);
      time+= MPI_Wtime();
    }

  // Display results
  ier = MPI_Allreduce(&senddps, &totalsentdps, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  if (irk == 0) printf("Transfered %10.3e double precision reals in %9.2es at an average of %9.2eMB/s per process for %3i processors\n",totalsentdps, time, sizeof(double)*totalsentdps/time/1024.0/1024.0/(double)isz,isz);

  ier = MPI_Finalize();
  
}
