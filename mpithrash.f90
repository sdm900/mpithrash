program mpithrash
  !
  ! Code developed by Dr Stuart Midgley sdm900@gmail.com
  !
  ! MPITHRASH is an MPI code which produces an large unpredictable MPI
  ! load on the network and MPI subsystem.  It is designed to test the
  ! scalability of the high performance interconnect for a general work
  ! load.
  !

  !
  ! Building
  ! --------
  ! Compile the code for the host platform, eg.
  !       f90 -fast mpithrash.f90 -o mpithrash -lmpi
  !

  implicit none

  include 'mpif.h'

  !
  ! dp           specifies the KIND value for double precision
  !              real numbers, typically 8, eg. REAL*8, REAL(8)
  !
  integer, parameter :: dp=kind(1.0d0)
  !
  ! maxmsgsize   is the maximum message size
  ! sendnmsgs    is the number of messages to send per process
  ! recvnmsgs    is the number of messages to receive simultaneously
  !
  integer, parameter :: maxmsgsize=65536, sendnmsgs=1000, recvnmsgs=20
  integer, parameter :: iterations=10


  integer :: irk, isz, ier, i, j, k, dest, msgsize, psenddps, szdbl
  real(dp) :: totalsentdps, senddps, time
  integer, dimension(:), allocatable :: seed, sendcnt, totsendcnt
  real, dimension(2) :: tmp
  real(dp), dimension(:,:), allocatable :: a, b
  integer, dimension(sendnmsgs) :: sendreq
  integer, dimension(recvnmsgs) :: recvreq
  integer, dimension(MPI_STATUS_SIZE, sendnmsgs) :: sendstatus
  integer, dimension(MPI_STATUS_SIZE, recvnmsgs) :: recvstatus


  ! Initialise MPI
  call MPI_Init(ier)
  if (ier /= 0) stop 'Can not initiate MPI'

  call MPI_Comm_rank(MPI_COMM_WORLD, irk, ier)
  call MPI_Comm_size(MPI_COMM_WORLD, isz, ier)

  ! Ensure different random numbers on each process
  call random_seed(size=i)
  allocate(seed(i))
  call random_seed(get=seed)
  seed = (modulo(seed,isz)+irk)*(seed/(isz+irk))
  call random_seed(put=seed)
  deallocate(seed)

  ! Allocate all buffers and initialise variables
  allocate(sendcnt(0:isz-1), totsendcnt(0:isz-1), a(maxmsgsize, sendnmsgs), b(maxmsgsize, recvnmsgs))
  a = 0.0d0
  senddps = 0.0d0
  time=0.0d0
  szdbl = size(transfer(senddps, (/ 'a' /)))
  
  ! Loop over a number of iterations to get better timing results
  do k=1, iterations
     psenddps = 0
     sendcnt = 0
     sendreq = MPI_REQUEST_NULL
     recvreq = MPI_REQUEST_NULL
     
     ! Send 1000 messages/process of random size to a random destination
     do i = 1, sendnmsgs
        call random_number(tmp)
        dest = int(tmp(1)*real(isz))
        msgsize = 1+int(tmp(2)*real(maxmsgsize))
        ! Messages are sent synchonously so that they do not get
        ! transported over the network until the recieve is
        ! posted
        call MPI_Issend(a(1, i), msgsize, MPI_DOUBLE_PRECISION, dest, 100, MPI_COMM_WORLD, sendreq(i), ier)
        sendcnt(dest) = sendcnt(dest)+1
        psenddps = psenddps + msgsize
     end do
     
     ! Global reduction so that each proces knows what messages
     ! have been sent.  Each process can then post the correct
     ! number of recieves
     senddps = senddps + dble(psenddps)
     call MPI_Allreduce(sendcnt, totsendcnt, isz, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ier)
     
     ! Start timing
     call MPI_Barrier(MPI_COMM_WORLD, ier)
     time = time - MPI_Wtime()
     
     ! Post recieves in blocks of 20
     do i = 1, totsendcnt(irk), recvnmsgs
        do j = i, min(i+recvnmsgs-1, totsendcnt(irk))
           call MPI_Irecv(b(1, j-i+1), maxmsgsize, MPI_DOUBLE_PRECISION, MPI_ANY_SOURCE, 100, MPI_COMM_WORLD, recvreq(j-i+1), ier)
        end do
        
        call MPI_Waitall(recvnmsgs, recvreq, recvstatus, ier)
     end do
     
     ! Finish up
     call MPI_Waitall(sendnmsgs, sendreq, sendstatus, ier)
     call MPI_Barrier(MPI_COMM_WORLD, ier)
     time = time + MPI_Wtime()
  end do

  ! Display results
  call MPI_Allreduce(senddps, totalsentdps, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ier)

  if (irk == 0) write(*,'(a,es10.3,a,es9.2,a,es9.2,a,i3,a)') &
       'Transfered ', &
       totalsentdps, &
       ' double precision reals in ', &
       time, &
       's at an average of ', &
       dble(szdbl)*totalsentdps/time/1024.0d0/1024.0d0/dble(isz), &
       'MB/s per process for ', &
       isz, &
       ' processors.'

  deallocate(sendcnt, totsendcnt, a, b)

  call MPI_Finalize(ier)
end program mpithrash
