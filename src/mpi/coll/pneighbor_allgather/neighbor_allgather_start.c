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

/* -- Begin Profiling Symbol Block for routine MPI_Neighbor_allgather_start */
#if defined(HAVE_PRAGMA_WEAK)
#pragma weak MPI_Neighbor_allgather_start = PMPI_Neighbor_allgather_start
#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#pragma _HP_SECONDARY_DEF PMPI_Neighbor_allgather_start  MPI_Neighbor_allgather_start
#elif defined(HAVE_PRAGMA_CRI_DUP)
#pragma _CRI duplicate MPI_Neighbor_allgather_start as PMPI_Neighbor_allgather_start
#elif defined(HAVE_WEAK_ATTRIBUTE)
int MPI_Neighbor_allgather_start(MPI_Comm comm, MPI_Request *request)
    __attribute__((weak,alias("PMPI_Neighbor_allgather_start")));
#endif
/* -- End Profiling Symbol Block */

/* Define MPICH_MPI_FROM_PMPI if weak symbols are not supported to build
   the MPI routines */
#ifndef MPICH_MPI_FROM_PMPI
#undef MPI_Neighbor_allgather_start
#define MPI_Neighbor_allgather_start PMPI_Neighbor_allgather_start

/* any non-MPI functions go here, especially non-static ones */

#endif /* MPICH_MPI_FROM_PMPI */


#undef FUNCNAME
#define FUNCNAME MPI_Neighbor_allgather_start
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
/*@
MPI_Neighbor_allgather_start - Persistent version of MPI_Neighbor_allgather.

Input Parameters:
- comm - communicator (handle)
- request - communication request (handle)

Output Parameters:
+ None


.N ThreadSafe

.N Fortran

.N Errors
@*/
int MPI_Neighbor_allgather_start(MPI_Comm comm, MPI_Request *request)
{
    /*
     * This is not the right way for implementing a
     * persistent neighborhood collective. We should
     * call the 'start' function on a request, not the
     * communicator. I'm doing it this way because I'm
     * not sure how to make a persistent request out of
     * a MPIR_Sched_t schedule.
     */
    int mpi_errno = MPI_SUCCESS;
    MPIR_Comm *comm_ptr = NULL;
    MPIR_FUNC_TERSE_STATE_DECL(MPID_STATE_MPI_NEIGHBOR_ALLGATHER_START);

    MPID_THREAD_CS_ENTER(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    MPIR_FUNC_TERSE_ENTER(MPID_STATE_MPI_INEIGHBOR_ALLGATHER_START);

    /* Convert MPI object handles to object pointers */
    MPIR_Comm_get_ptr(comm, comm_ptr);

    MPIR_Topology *topo_ptr = NULL;
    topo_ptr = MPIR_Topology_get(comm_ptr);
    if(!topo_ptr)
    {
        fprintf(stderr, "ERROR: topo_ptr is not valid! Aborting MPI_Neighbor_allgather_start\n");
        goto fn_fail;
    }
    if(topo_ptr->kind != MPI_DIST_GRAPH)
    {
        fprintf(stderr,
                "ERROR: Topology is not MPI_DIST_GRAPH; Aborting MPI_Neighbor_allgather_start\n");
        goto fn_fail;
    }
    if(topo_ptr->topo.dist_graph.shm_nbh_coll_sched == NULL)
    {
        fprintf(stderr,
                "ERROR: shm_nbh_coll_sched is NULL; Aborting MPI_Neighbor_allgather_start\n");
        goto fn_fail;
    }

    int tag = -1;
    MPIR_Request *reqp = NULL;
    MPIR_Sched_t s = MPIR_SCHED_NULL;

    *request = MPI_REQUEST_NULL;

    mpi_errno = MPIR_Sched_next_tag(comm_ptr, &tag);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

    s = topo_ptr->topo.dist_graph.shm_nbh_coll_sched;

    mpi_errno = MPIR_Sched_start(&s, comm_ptr, tag, &reqp);
    if (reqp)
        *request = reqp->handle;
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

fn_exit:
    MPIR_FUNC_TERSE_EXIT(MPID_STATE_MPI_INEIGHBOR_ALLGATHER_START);
    MPID_THREAD_CS_EXIT(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    return mpi_errno;
fn_fail:
    goto fn_exit;
}
