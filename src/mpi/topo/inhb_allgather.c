/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2012 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpiimpl.h"

/* -- Begin Profiling Symbol Block for routine MPI_Ineighbor_allgather */
#if defined(HAVE_PRAGMA_WEAK)
#pragma weak MPI_Ineighbor_allgather = PMPI_Ineighbor_allgather
#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#pragma _HP_SECONDARY_DEF PMPI_Ineighbor_allgather  MPI_Ineighbor_allgather
#elif defined(HAVE_PRAGMA_CRI_DUP)
#pragma _CRI duplicate MPI_Ineighbor_allgather as PMPI_Ineighbor_allgather
#elif defined(HAVE_WEAK_ATTRIBUTE)
int MPI_Ineighbor_allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                            void *recvbuf, int recvcount, MPI_Datatype recvtype,
                            MPI_Comm comm, MPI_Request *request)
                            __attribute__((weak,alias("PMPI_Ineighbor_allgather")));
#endif
/* -- End Profiling Symbol Block */

/* Define MPICH_MPI_FROM_PMPI if weak symbols are not supported to build
   the MPI routines */
#ifndef MPICH_MPI_FROM_PMPI
#undef MPI_Ineighbor_allgather
#define MPI_Ineighbor_allgather PMPI_Ineighbor_allgather

/* any non-MPI functions go here, especially non-static ones */

/***** Begin SHM ******/
//SHM
MPI_Aint find_incom_tmp_buf_size(int **incom_sched_mat, int num_rows, MPI_Aint size_per_rank)
{
    int i;
	MPI_Aint buff_size;
    buff_size = 0;
    for(i = 0; i < num_rows; i++)
    {
        if(incom_sched_mat[i][1] != -1) //Initially-ON incoming neighbors
        {
            buff_size += incom_sched_mat[i][2] * size_per_rank;
        }
    }
    return buff_size;
}

MPI_Aint find_incom_tmp_buf_offset(int **incom_sched_mat, int nbr_index, MPI_Aint size_per_rank)
{
    int i;
	MPI_Aint offset;
    offset = 0;
    for(i = 0; i < nbr_index; i++)
    {
        if(incom_sched_mat[i][1] != -1) //Initially-ON incoming neighbors
        {
            offset += incom_sched_mat[i][2] * size_per_rank;
        }
    }
    return offset;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Ineighbor_allgather_SHM
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Ineighbor_allgather_SHM(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount,
                                MPI_Datatype recvtype, MPIR_Comm *comm_ptr, MPIR_Sched_t s, int persistent_coll)
{
    /* At this point, we already have extracted all information
	 * needed for building the schedule. The information is
	 * attached to the topo.dist_graph field of the communicator
	 * as a SHM_nbr_coll_patt structure. Now, we have to build
	 * the schedule out of the information provided by this struct.
	 */
	//TODO: Update this implementation based on the new one used for alltoallv.
	
    double sched_time = -MPI_Wtime();
    
	int mpi_errno = MPI_SUCCESS;
    MPIR_Topology *topo_ptr = NULL;
    topo_ptr = MPIR_Topology_get(comm_ptr);
    Common_nbrhood_matrix *cmn_nbh_mat = topo_ptr->topo.dist_graph.shm_nbh_coll_patt->cmn_nbh_mat;
    int **incom_sched_mat = topo_ptr->topo.dist_graph.shm_nbh_coll_patt->incom_sched_mat;

    int i, j;
    int indegree, outdegree;
    int *srcs, *dests;
	
	MPI_Aint recvtype_extent, recvtype_true_extent, 
			 recvtype_max_extent, recvtype_true_lb;
    MPI_Aint sendtype_extent, sendtype_true_extent, 
			 sendtype_max_extent, sendtype_true_lb;
    MPI_Aint recvbuf_extent;
    MPI_Aint sendbuf_extent;
	MPI_Aint exchange_tmp_buf_extent;
	MPI_Aint incom_tmp_buf_extent;
    MPI_Aint exchange_recv_count, exchange_send_count;
	MPI_Aint incom_recv_count, out_send_count;

    void *exchange_tmp_buf = NULL;
    void *incom_tmp_buf = NULL;

    indegree = topo_ptr->topo.dist_graph.indegree;
    outdegree = topo_ptr->topo.dist_graph.outdegree;
    srcs = topo_ptr->topo.dist_graph.in;
    dests = topo_ptr->topo.dist_graph.out;

	//get extent and true extent of sendtype and recvtype
    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
    MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
    MPIR_Type_get_true_extent_impl(recvtype, &recvtype_true_lb, &recvtype_true_extent);
    MPIR_Type_get_true_extent_impl(sendtype, &sendtype_true_lb, &sendtype_true_extent);
    recvtype_max_extent = MPL_MAX(recvtype_true_extent, recvtype_extent);
    sendtype_max_extent = MPL_MAX(sendtype_true_extent, sendtype_extent);

    //get extent (true) of sendbuf and recvbuf
	recvbuf_extent = recvcount * recvtype_max_extent;
    sendbuf_extent = sendcount * sendtype_max_extent;
    
    MPIR_SCHED_CHKPMEM_DECL(2);
    MPIR_CHKPMEM_DECL(2);

    exchange_tmp_buf_extent = 2 * sendbuf_extent;//locally known for a simple allgather
    incom_tmp_buf_extent = find_incom_tmp_buf_size(incom_sched_mat, indegree, recvbuf_extent);
    if(persistent_coll)
    {
        MPIR_CHKPMEM_MALLOC(exchange_tmp_buf, void*, exchange_tmp_buf_extent, mpi_errno, "exchange_tmp_buf");
        MPIR_CHKPMEM_MALLOC(incom_tmp_buf, void*, incom_tmp_buf_extent, mpi_errno, "incom_tmp_buf");
        topo_ptr->topo.dist_graph.sched_mem_to_free[topo_ptr->topo.dist_graph.sched_mem_to_free_num_entries++] = exchange_tmp_buf;
        topo_ptr->topo.dist_graph.sched_mem_to_free[topo_ptr->topo.dist_graph.sched_mem_to_free_num_entries++] = incom_tmp_buf;
    }
    else //not persistent schedule
    {
        MPIR_SCHED_CHKPMEM_MALLOC(exchange_tmp_buf, void*, exchange_tmp_buf_extent, mpi_errno, "exchange_tmp_buf");
        MPIR_SCHED_CHKPMEM_MALLOC(incom_tmp_buf, void*, incom_tmp_buf_extent, mpi_errno, "incom_tmp_buf");
    }

    int t;
    for(t = 0; t < cmn_nbh_mat->t; t++) 
    {
		/** Building the schedule one time step at a time **/
        
		//Compiling cmn_nbh_mat first
        for(i = 0; i < cmn_nbh_mat->num_rows; i++)
        {
            if(cmn_nbh_mat->comb_matrix[i][t].opt != IDLE) //Then we'll have to do an exchange with a friend
            {
                exchange_recv_count = sendcount;
                exchange_send_count = sendcount;
                char *rb = ((char *)exchange_tmp_buf) + exchange_send_count * sendtype_extent;
                mpi_errno = MPIR_Sched_recv(rb, exchange_recv_count, sendtype, cmn_nbh_mat->comb_matrix[i][t].paired_frnd, comm_ptr, s);
                if (mpi_errno) MPIR_ERR_POP(mpi_errno);

                mpi_errno = MPIR_Sched_send(sendbuf, exchange_send_count, sendtype, cmn_nbh_mat->comb_matrix[i][t].paired_frnd, comm_ptr, s);
                if (mpi_errno) MPIR_ERR_POP(mpi_errno);

                //Schedule a copy from sendbuf to exchange_tmp_buf
                char *copy_from = ((char*)sendbuf);
                char *copy_to = ((char*)exchange_tmp_buf);
                MPIR_Sched_copy(copy_from, sendcount, sendtype,
                                copy_to, sendcount, sendtype, s);

                MPIR_SCHED_BARRIER(s);

                //Schedule to send the combined message to corresponding neighbors
                int k;
                for(k = i; k < cmn_nbh_mat->num_rows; k++)
                {
                    if(!cmn_nbh_mat->is_row_offloaded[k] &&
                        cmn_nbh_mat->comb_matrix[k][t].opt != IDLE)
                    {
                        mpi_errno = MPIR_Sched_send(exchange_tmp_buf, 
									exchange_send_count + exchange_recv_count,
                                    sendtype, dests[k], comm_ptr, s);
                        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
                    }
                }

                break;//we have a maximum of one paired friend at each time step t
            }
        }

        //Compiling the incom_sched_mat
        //Scheduling the necessary recv operations
        for(i = 0; i < indegree; i++)
        {
            if(!incom_sched_mat[i][0]) //On incoming neighbors
            {
                if(incom_sched_mat[i][1] == t)
                {
                    //Schedule a receive from the corresponding source.
                    incom_recv_count = incom_sched_mat[i][2] * recvcount;
                    char *rb = ((char*)incom_tmp_buf) + find_incom_tmp_buf_offset(incom_sched_mat, i, recvbuf_extent);
                    mpi_errno = MPIR_Sched_recv(rb, incom_recv_count, recvtype, incom_sched_mat[i][COMB_LIST_START_IDX], comm_ptr, s);
                    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
                }
            }
            else //Off incoming neighbors
            {
                //Nothing to do with no cumulative.
            }
        }

        MPIR_SCHED_BARRIER(s);
        
		//Scheduling message copies from incom_tmp_buf to final recv_buf
        for(i = 0; i < indegree; i++)
        {
            if(!incom_sched_mat[i][0]) //On incoming neighbors
            {
                if(incom_sched_mat[i][1] == t)
                {
                    MPI_Aint offset1 = find_incom_tmp_buf_offset(incom_sched_mat, i, recvbuf_extent);
                    for(j = 0; j < incom_sched_mat[i][2]; j++) //Each rank in the combining list
                    {
                        int nbr_idx = find_in_arr(srcs, indegree, incom_sched_mat[i][j + COMB_LIST_START_IDX]);
                        char *copy_from = ((char*)incom_tmp_buf) + offset1 + j * recvcount * recvtype_extent;
                        char *copy_to = ((char*)recvbuf) + nbr_idx * recvcount * recvtype_extent;
                        MPIR_Sched_copy(copy_from, recvcount, recvtype,
                                        copy_to, recvcount, recvtype, s);
                    }
                }
            }
        }

        MPIR_SCHED_BARRIER(s);
    }

    MPIR_SCHED_BARRIER(s);
    
	//Schedule a send for any residual outgoing neighbors
    for(i = 0; i < cmn_nbh_mat->num_rows; i++)
    {
        if(!cmn_nbh_mat->ignore_row[i]) //out neighbor still active
        {
            mpi_errno = MPIR_Sched_send(sendbuf, sendcount, sendtype,
                                        dests[i], comm_ptr, s);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);
        }
    }

	//Schedule a recv for any residual incoming neighbors
    /* NOTE: the incoming message can still be a combined one.
     * TODO: We might be able to use 't' to schedule these 
	 * residual recvs in a stepwise fashion to create some 
	 * overlap among the recv and copy operations.
     */
    for(i = 0; i < indegree; i++)
    {
        if(!incom_sched_mat[i][0] && incom_sched_mat[i][1] >= cmn_nbh_mat->t) //On incoming neighbors not covered before
        {
            //Schedule a receive from the corresponding source.
            incom_recv_count = incom_sched_mat[i][2] * recvcount;
            char *rb = ((char*)incom_tmp_buf) + find_incom_tmp_buf_offset(incom_sched_mat, i, recvbuf_extent);
            mpi_errno = MPIR_Sched_recv(rb, incom_recv_count, recvtype, incom_sched_mat[i][COMB_LIST_START_IDX], comm_ptr, s);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);
        }
        else //Off incoming neighbors
        {
			//Nothing to do with no cumulative.
        }
    }

    MPIR_SCHED_BARRIER(s);

    //Scheduling message copies from incom_tmp_buf to final recv_buf
    for(i = 0; i < indegree; i++)
    {
        if(!incom_sched_mat[i][0] && incom_sched_mat[i][1] >= cmn_nbh_mat->t) //On incoming neighbors not covered before
        {
            MPI_Aint offset1 = find_incom_tmp_buf_offset(incom_sched_mat, i, recvbuf_extent);
            for(j = 0; j < incom_sched_mat[i][2]; j++)
            {
                int nbr_idx = find_in_arr(srcs, indegree, incom_sched_mat[i][j + COMB_LIST_START_IDX]);
                char *copy_from = ((char*)incom_tmp_buf) + offset1 + j * recvcount * recvtype_extent;
                char *copy_to = ((char*)recvbuf) + nbr_idx * recvcount * recvtype_extent;
                MPIR_Sched_copy(copy_from, recvcount, recvtype,
                                copy_to, recvcount, recvtype, s);
            }
        }
    }

    MPIR_SCHED_BARRIER(s);

    sched_time += MPI_Wtime();

    if(persistent_coll) 
    {
		/* NOTE: In non-persistent case, time measurement overheads
		 * will fall within the critical path. Thus, we only do it 
		 * in the persistent case.
		 */
        double max_sched_time = 0;
		MPIR_Errflag_t errflag = MPIR_ERR_NONE;
        MPIR_Reduce_impl(&sched_time, &max_sched_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm_ptr, &errflag);
        if(comm_ptr->rank == 0)
        {
            printf("Time to build the SHM neighborhood schedule (max): %lf (s)\n", max_sched_time);
            fflush(stdout);
        }
    }

    if(persistent_coll)
    {
        MPIR_CHKPMEM_COMMIT();
    }
    else
    {
        MPIR_SCHED_CHKPMEM_COMMIT(s);
    }

fn_exit:
    return mpi_errno;
fn_fail:
    if(persistent_coll)
    {
        MPIR_CHKPMEM_REAP();
    }
    else
    {
        MPIR_SCHED_CHKPMEM_REAP(s);
    }
    goto fn_exit;
}
/***** End SHM *****/

#undef FUNCNAME
#define FUNCNAME MPIR_Ineighbor_allgather_default
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Ineighbor_allgather_default(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPIR_Comm *comm_ptr, MPIR_Sched_t s)
{
    int mpi_errno = MPI_SUCCESS;
    int indegree, outdegree, weighted;
    int k,l;
    int *srcs, *dsts;
    MPI_Aint recvtype_extent;
    MPIR_CHKLMEM_DECL(2);

    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);

    /* This is the largest offset we add to recvbuf */
    MPIR_Ensure_Aint_fits_in_pointer(MPIR_VOID_PTR_CAST_TO_MPI_AINT recvbuf +
                                     (comm_ptr->local_size * recvcount * recvtype_extent));

    mpi_errno = MPIR_Topo_canon_nhb_count(comm_ptr, &indegree, &outdegree, &weighted);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    MPIR_CHKLMEM_MALLOC(srcs, int *, indegree*sizeof(int), mpi_errno, "srcs");
    MPIR_CHKLMEM_MALLOC(dsts, int *, outdegree*sizeof(int), mpi_errno, "dsts");
    mpi_errno = MPIR_Topo_canon_nhb(comm_ptr,
                                    indegree, srcs, MPI_UNWEIGHTED,
                                    outdegree, dsts, MPI_UNWEIGHTED);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

    for (k = 0; k < outdegree; ++k) {
        mpi_errno = MPIR_Sched_send(sendbuf, sendcount, sendtype, dsts[k], comm_ptr, s);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    }

    for (l = 0; l < indegree; ++l) {
        char *rb = ((char *)recvbuf) + l * recvcount * recvtype_extent;
        mpi_errno = MPIR_Sched_recv(rb, recvcount, recvtype, srcs[l], comm_ptr, s);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    }

    MPIR_SCHED_BARRIER(s);

fn_exit:
    MPIR_CHKLMEM_FREEALL();
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Ineighbor_allgather_impl
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Ineighbor_allgather_impl(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPIR_Comm *comm_ptr, MPI_Request *request)
{
    int mpi_errno = MPI_SUCCESS;
    int tag = -1;
    MPIR_Request *reqp = NULL;
    MPIR_Sched_t s = MPIR_SCHED_NULL;

    *request = MPI_REQUEST_NULL;

    mpi_errno = MPIR_Sched_next_tag(comm_ptr, &tag);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    mpi_errno = MPIR_Sched_create(&s);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    MPIR_Assert(comm_ptr->coll_fns != NULL);
    MPIR_Assert(comm_ptr->coll_fns->Ineighbor_allgather != NULL);
	//SHM
    if(nbr_impl == 1) //Use SHM implementation--This is not the right way of doing it.
    	mpi_errno = MPIR_Ineighbor_allgather_SHM(sendbuf, sendcount, sendtype,
    											 recvbuf, recvcount, recvtype,
    											 comm_ptr, s, 0); //0 to denote a non-persistent call
	else
		mpi_errno = comm_ptr->coll_fns->Ineighbor_allgather(sendbuf, sendcount, sendtype,
                                                        recvbuf, recvcount, recvtype,
                                                        comm_ptr, s);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

    mpi_errno = MPIR_Sched_start(&s, comm_ptr, tag, &reqp);
    if (reqp)
        *request = reqp->handle;
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}

#endif /* MPICH_MPI_FROM_PMPI */

#undef FUNCNAME
#define FUNCNAME MPI_Ineighbor_allgather
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
/*@
MPI_Ineighbor_allgather - Nonblocking version of MPI_Neighbor_allgather.

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
int MPI_Ineighbor_allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_Comm *comm_ptr = NULL;
    MPIR_FUNC_TERSE_STATE_DECL(MPID_STATE_MPI_INEIGHBOR_ALLGATHER);

    MPID_THREAD_CS_ENTER(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    MPIR_FUNC_TERSE_ENTER(MPID_STATE_MPI_INEIGHBOR_ALLGATHER);

    /* Validate parameters, especially handles needing to be converted */
#   ifdef HAVE_ERROR_CHECKING
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
#   endif /* HAVE_ERROR_CHECKING */

    /* Convert MPI object handles to object pointers */
    MPIR_Comm_get_ptr(comm, comm_ptr);

    /* Validate parameters and objects (post conversion) */
#   ifdef HAVE_ERROR_CHECKING
    {
        MPID_BEGIN_ERROR_CHECKS
        {
            if (HANDLE_GET_KIND(sendtype) != HANDLE_KIND_BUILTIN) {
                MPIR_Datatype *sendtype_ptr = NULL;
                MPID_Datatype_get_ptr(sendtype, sendtype_ptr);
                MPIR_Datatype_valid_ptr(sendtype_ptr, mpi_errno);
                if (mpi_errno != MPI_SUCCESS) goto fn_fail;
                MPID_Datatype_committed_ptr(sendtype_ptr, mpi_errno);
                if (mpi_errno != MPI_SUCCESS) goto fn_fail;
            }

            if (HANDLE_GET_KIND(recvtype) != HANDLE_KIND_BUILTIN) {
                MPIR_Datatype *recvtype_ptr = NULL;
                MPID_Datatype_get_ptr(recvtype, recvtype_ptr);
                MPIR_Datatype_valid_ptr(recvtype_ptr, mpi_errno);
                if (mpi_errno != MPI_SUCCESS) goto fn_fail;
                MPID_Datatype_committed_ptr(recvtype_ptr, mpi_errno);
                if (mpi_errno != MPI_SUCCESS) goto fn_fail;
            }

            MPIR_Comm_valid_ptr( comm_ptr, mpi_errno, FALSE );
            if (mpi_errno != MPI_SUCCESS) goto fn_fail;
            MPIR_ERRTEST_ARGNULL(request, "request", mpi_errno);
            /* TODO more checks may be appropriate (counts, in_place, buffer aliasing, etc) */
        }
        MPID_END_ERROR_CHECKS
    }
#   endif /* HAVE_ERROR_CHECKING */

    /* ... body of routine ...  */

    mpi_errno = MPID_Ineighbor_allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm_ptr, request);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

    /* ... end of body of routine ... */

fn_exit:
    MPIR_FUNC_TERSE_EXIT(MPID_STATE_MPI_INEIGHBOR_ALLGATHER);
    MPID_THREAD_CS_EXIT(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    return mpi_errno;

fn_fail:
    /* --BEGIN ERROR HANDLING-- */
#   ifdef HAVE_ERROR_CHECKING
    {
        mpi_errno = MPIR_Err_create_code(
            mpi_errno, MPIR_ERR_RECOVERABLE, FCNAME, __LINE__, MPI_ERR_OTHER,
            "**mpi_ineighbor_allgather", "**mpi_ineighbor_allgather %p %d %D %p %d %D %C %p", sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request);
    }
#   endif
    mpi_errno = MPIR_Err_return_comm(NULL, FCNAME, mpi_errno);
    goto fn_exit;
    /* --END ERROR HANDLING-- */
}
