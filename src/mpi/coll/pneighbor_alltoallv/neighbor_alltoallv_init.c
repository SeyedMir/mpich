/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2012 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

/*---------------------------------------------------------------------*/
/*     (C) Copyright 2017 Parallel Processing Research Laboratory      */
/*                   Queen's University at Kingston                    */
/*                Neighborhood Collective Communication                */
/*                    Seyed Hessamedin Mirsadeghi                      */
/*---------------------------------------------------------------------*/

#include "mpiimpl.h"

/* -- Begin Profiling Symbol Block for routine MPI_Neighbor_alltoallv_init */
#if defined(HAVE_PRAGMA_WEAK)
#pragma weak MPI_Neighbor_alltoallv_init = PMPI_Neighbor_alltoallv_init
#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#pragma _HP_SECONDARY_DEF PMPI_Neighbor_alltoallv_init  MPI_Neighbor_alltoallv_init
#elif defined(HAVE_PRAGMA_CRI_DUP)
#pragma _CRI duplicate MPI_Neighbor_alltoallv_init as PMPI_Neighbor_alltoallv_init
#elif defined(HAVE_WEAK_ATTRIBUTE)
int MPI_Neighbor_alltoallv_init(const void *sendbuf, const int sendcounts[], const int sdispls[],
								MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
								const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm,
								MPI_Request *request)
    __attribute__((weak,alias("PMPI_Neighbor_alltoallv_init")));
#endif
/* -- End Profiling Symbol Block */

/* Define MPICH_MPI_FROM_PMPI if weak symbols are not supported to build
   the MPI routines */
#ifndef MPICH_MPI_FROM_PMPI
#undef MPI_Neighbor_alltoallv_init
#define MPI_Neighbor_alltoallv_init PMPI_Neighbor_alltoallv_init

/* any non-MPI functions go here, especially non-static ones */

#undef FUNCNAME
#define FUNCNAME MPIR_Neighbor_alltoallv_init_impl
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Neighbor_alltoallv_init_impl(const void *sendbuf, const int sendcounts[],
									  const int sdispls[], MPI_Datatype sendtype,
									  void *recvbuf, const int recvcounts[],
									  const int rdispls[], MPI_Datatype recvtype,
									  MPIR_Comm *comm_ptr, MPI_Request *request)
{
    /*
     * We are ignoring the request handle passed to this
     * function for now until we fix the persistent
     * implementation later.
     */
    int mpi_errno = MPI_SUCCESS;

    MPIR_Topology *topo_ptr = NULL;
    topo_ptr = MPIR_Topology_get(comm_ptr);
    if(!topo_ptr)
    {
        fprintf(stderr, "ERROR: topo_ptr is not valid! Aborting MPI_Neighbor_alltoallv_start\n");
        goto fn_fail;
    }
    if(topo_ptr->kind != MPI_DIST_GRAPH)
    {
        fprintf(stderr,
                "ERROR: Topology is not MPI_DIST_GRAPH; Aborting MPI_Neighbor_alltoallv_start\n");
        goto fn_fail;
    }
    if(topo_ptr->topo.dist_graph.shm_nbh_coll_sched != NULL)
    {
        if(comm_ptr->rank == 0)
            printf("Warning: Overwriting a previous shm_nbh_coll_sched!\n");
        MPIR_Sched_free(topo_ptr->topo.dist_graph.shm_nbh_coll_sched);
        while(topo_ptr->topo.dist_graph.sched_mem_to_free_num_entries > 0)
        	MPL_free(topo_ptr->topo.dist_graph.sched_mem_to_free[--(topo_ptr->
                        topo.dist_graph.sched_mem_to_free_num_entries)]);
    }

    MPIR_Sched_t s = MPIR_SCHED_NULL;

    mpi_errno = MPIR_Sched_create(&s);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

    MPIR_Sched_make_persistent(&s);

    mpi_errno = MPIR_Ineighbor_alltoallv_sched(sendbuf, sendcounts, sdispls, sendtype,
                                               recvbuf, recvcounts, rdispls, recvtype,
                                               comm_ptr, s, 1);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

    topo_ptr->topo.dist_graph.shm_nbh_coll_sched = s;

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

#endif /* MPICH_MPI_FROM_PMPI */

#undef FUNCNAME
#define FUNCNAME MPI_Neighbor_alltoallv_init
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
/*@
MPI_Neighbor_alltoallv_init - Persistent version of MPI_Neighbor_alltoallv.

Input Parameters:
+ sendbuf - starting address of the send buffer (choice)
. sendcount - number of elements sent to each neighbor (non-negative integer)
. sendtype - data type of send buffer elements (handle)
. recvcount - number of elements received from each neighbor (non-negative integer)
. recvtype - data type of receive buffer elements (handle)
- comm - communicator (handle)

Output Parameters:
+ recvbuf - starting address of the receive buffer (choice)
- request - communication request (handle)

.N ThreadSafe

.N Fortran

.N Errors
@*/
int MPI_Neighbor_alltoallv_init(const void *sendbuf, const int sendcounts[], const int sdispls[],
                                MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
                                const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm,
                                MPI_Request *request)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_Comm *comm_ptr = NULL;
    MPIR_FUNC_TERSE_STATE_DECL(MPID_STATE_MPI_NEIGHBOR_ALLTOALLV_INIT);

    MPID_THREAD_CS_ENTER(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    MPIR_FUNC_TERSE_ENTER(MPID_STATE_MPI_NEIGHBOR_ALLTOALLV_INIT);

    /* Validate parameters, especially handles needing to be converted */
#ifdef HAVE_ERROR_CHECKING
    {
        MPID_BEGIN_ERROR_CHECKS
        {
            MPIR_ERRTEST_DATATYPE(sendtype, "sendtype", mpi_errno);
            MPIR_ERRTEST_DATATYPE(recvtype, "recvtype", mpi_errno);
            MPIR_ERRTEST_COMM(comm, mpi_errno);

            /* TODO more checks may be appropriate */
        }
        MPID_END_ERROR_CHECKS
    }
#endif /* HAVE_ERROR_CHECKING */

    /* Convert MPI object handles to object pointers */
    MPIR_Comm_get_ptr(comm, comm_ptr);

    /* Validate parameters and objects (post conversion) */
#ifdef HAVE_ERROR_CHECKING
    {
        MPID_BEGIN_ERROR_CHECKS;
        {
            if (HANDLE_GET_KIND(sendtype) != HANDLE_KIND_BUILTIN) {
                MPIR_Datatype *sendtype_ptr = NULL;
                MPIR_Datatype_get_ptr(sendtype, sendtype_ptr);
                MPIR_Datatype_valid_ptr(sendtype_ptr, mpi_errno);
                if (mpi_errno != MPI_SUCCESS) goto fn_fail;
                MPIR_Datatype_committed_ptr(sendtype_ptr, mpi_errno);
                if (mpi_errno != MPI_SUCCESS) goto fn_fail;
            }

            if (HANDLE_GET_KIND(recvtype) != HANDLE_KIND_BUILTIN) {
                MPIR_Datatype *recvtype_ptr = NULL;
                MPIR_Datatype_get_ptr(recvtype, recvtype_ptr);
                MPIR_Datatype_valid_ptr(recvtype_ptr, mpi_errno);
                if (mpi_errno != MPI_SUCCESS) goto fn_fail;
                MPIR_Datatype_committed_ptr(recvtype_ptr, mpi_errno);
                if (mpi_errno != MPI_SUCCESS) goto fn_fail;
            }

            MPIR_Comm_valid_ptr( comm_ptr, mpi_errno, FALSE );
            if (mpi_errno != MPI_SUCCESS) goto fn_fail;
            MPIR_ERRTEST_ARGNULL(request, "request", mpi_errno);
            /* TODO more checks may be appropriate (counts, in_place, buffer aliasing, etc) */
        }
        MPID_END_ERROR_CHECKS;
    }
#endif /* HAVE_ERROR_CHECKING */

    /* ... body of routine ...  */

    mpi_errno = MPIR_Neighbor_alltoallv_init_impl(sendbuf, sendcounts, sdispls,
    											  sendtype, recvbuf, recvcounts,
    											  rdispls, recvtype, comm_ptr, request);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

    /* ... end of body of routine ... */

fn_exit:
    MPIR_FUNC_TERSE_EXIT(MPID_STATE_MPI_NEIGHBOR_ALLTOALLV_INIT);
    MPID_THREAD_CS_EXIT(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    return mpi_errno;

fn_fail:
    /* --BEGIN ERROR HANDLING-- */
#ifdef HAVE_ERROR_CHECKING
    {
        mpi_errno =
            MPIR_Err_create_code(mpi_errno, MPIR_ERR_RECOVERABLE, FCNAME, __LINE__, MPI_ERR_OTHER,
                                 "**mpi_neighbor_alltoallv_init",
                                 "**mpi_neighbor_alltoallv_init %p %p %p %D %p %p %p %D %C %p",
                                 sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts,
                                 rdispls, recvtype, comm, request);
    }
#endif
    mpi_errno = MPIR_Err_return_comm(NULL, FCNAME, mpi_errno);
    goto fn_exit;
    /* --END ERROR HANDLING-- */
}
