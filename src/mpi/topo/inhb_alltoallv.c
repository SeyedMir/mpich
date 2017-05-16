/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2012 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpiimpl.h"

//SHM
#ifdef SHM_SCHED_DEBUG
/***** FOR DEBUGGING ONLY *****/
int debug_glob_all_reqs_idx = 0;
MPI_Request debug_glob_all_reqs[MAX_DEBUG_SCHED_REQS];
#endif

/* -- Begin Profiling Symbol Block for routine MPI_Ineighbor_alltoallv */
#if defined(HAVE_PRAGMA_WEAK)
#pragma weak MPI_Ineighbor_alltoallv = PMPI_Ineighbor_alltoallv
#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#pragma _HP_SECONDARY_DEF PMPI_Ineighbor_alltoallv  MPI_Ineighbor_alltoallv
#elif defined(HAVE_PRAGMA_CRI_DUP)
#pragma _CRI duplicate MPI_Ineighbor_alltoallv as PMPI_Ineighbor_alltoallv
#elif defined(HAVE_WEAK_ATTRIBUTE)
int MPI_Ineighbor_alltoallv(const void *sendbuf, const int sendcounts[], const int sdispls[],
                            MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
                            const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm,
                            MPI_Request *request)
                            __attribute__((weak,alias("PMPI_Ineighbor_alltoallv")));
#endif
/* -- End Profiling Symbol Block */

/* Define MPICH_MPI_FROM_PMPI if weak symbols are not supported to build
   the MPI routines */
#ifndef MPICH_MPI_FROM_PMPI
#undef MPI_Ineighbor_alltoallv
#define MPI_Ineighbor_alltoallv PMPI_Ineighbor_alltoallv

/* any non-MPI functions go here, especially non-static ones */

/***** Begin SHM ******/
/*---------------------------------------------------------------------*/
/*     (C) Copyright 2017 Parallel Processing Research Laboratory      */
/*                   Queen's University at Kingston                    */
/*                Neighborhood Collective Communication                */
/*                    Seyed Hessamedin Mirsadeghi                      */
/*---------------------------------------------------------------------*/
#undef FUNCNAME
#define FUNCNAME MPIR_Ineighbor_alltoallv_SHM
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Ineighbor_alltoallv_SHM(const void *sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype,
                                 void *recvbuf, const int recvcounts[], const int rdispls[], MPI_Datatype recvtype,
                                 MPIR_Comm *comm_ptr, MPIR_Sched_t s, int persistent_coll)
{
    double sched_time = -MPI_Wtime();

#ifdef SHM_DEBUG
    char content[256]; /* for in-file prints */
#endif

    int mpi_errno = MPI_SUCCESS;
    int i, j;
    int indegree, outdegree;
    int *srcs, *dests;
    MPIR_Topology *topo_ptr = NULL;
    topo_ptr = MPIR_Topology_get(comm_ptr);
    indegree = topo_ptr->topo.dist_graph.indegree;
    outdegree = topo_ptr->topo.dist_graph.outdegree;
    srcs = topo_ptr->topo.dist_graph.in;
    dests = topo_ptr->topo.dist_graph.out;

    Common_nbrhood_matrix *cmn_nbh_mat = topo_ptr->topo.dist_graph.shm_nbh_coll_patt->cmn_nbh_mat;
    int **incom_sched_mat = topo_ptr->topo.dist_graph.shm_nbh_coll_patt->incom_sched_mat;

    MPI_Aint recvtype_extent, recvtype_true_extent,
			 recvtype_max_extent, recvtype_true_lb;
    MPI_Aint sendtype_extent, sendtype_true_extent,
			 sendtype_max_extent, sendtype_true_lb;
    MPI_Aint recvbuf_extent;
    MPI_Aint sendbuf_extent;

    /* get extent and true extent of sendtype and recvtype */
    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);
    MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
    MPIR_Type_get_true_extent_impl(recvtype, &recvtype_true_lb, &recvtype_true_extent);
    MPIR_Type_get_true_extent_impl(sendtype, &sendtype_true_lb, &sendtype_true_extent);
    recvtype_max_extent = MPL_MAX(recvtype_true_extent, recvtype_extent);
    sendtype_max_extent = MPL_MAX(sendtype_true_extent, sendtype_extent);

    /* find the total sum for sendcounts and recvcounts */
    int sendcount_sum = 0, recvcount_sum = 0;
    sendcount_sum = find_array_sum(sendcounts, outdegree);
    recvcount_sum = find_array_sum(recvcounts, indegree);

    /* get extent (true) of sendbuf and recvbuf */
    recvbuf_extent = recvcount_sum * recvtype_max_extent;
    sendbuf_extent = sendcount_sum * sendtype_max_extent;

    MPIR_CHKPMEM_DECL(1); /* for incom_tmp_buf */
    MPIR_SCHED_CHKPMEM_DECL(1);
    void *sched_bufs_from_funcs[SCHED_MEM_TO_FREE_MAX_SIZE] = { NULL };
    int sched_bufs_from_funcs_count = 0;

    /* allocate temporary buffer to receive data from incoming neighbors */
    MPI_Aint incom_tmp_buf_extent;
    void *incom_tmp_buf = NULL;
    incom_tmp_buf_extent = recvbuf_extent;
    if(persistent_coll)
    {
        MPIR_CHKPMEM_MALLOC(incom_tmp_buf, void*, incom_tmp_buf_extent, mpi_errno, "incom_tmp_buf");
        topo_ptr->topo.dist_graph.sched_mem_to_free[topo_ptr->topo.dist_graph.sched_mem_to_free_num_entries++] = incom_tmp_buf;
    }
    else /* not persistent schedule */
    {
        MPIR_SCHED_CHKPMEM_MALLOC(incom_tmp_buf, void*, incom_tmp_buf_extent, mpi_errno, "incom_tmp_buf");
    }

	int all_reqs_idx = 0;
	MPI_Request all_reqs[2];

    int t;
    int *own_off_counts = NULL;
	int *friend_off_counts = NULL;
    void *exchange_sendbuf = NULL;
    void *exchange_recvbuf = NULL;
    void *combined_sendbuf = NULL;
    for(t = 0; t < cmn_nbh_mat->t; t++)
    {
    	/** Building the schedule one time step at a time **/

        /* Compiling cmn_nbh_mat first */
        if(cmn_nbh_mat->num_onloaded[t] > 0)
        {
			/***** GET FRIEND_OFF_COUNTS *****/
			mpi_errno = a2aV_get_friend_off_counts(t, cmn_nbh_mat, comm_ptr,
											  &(all_reqs[all_reqs_idx++]),
											  &friend_off_counts);

			/***** SEND OWN_OFF_COUNTS TO FRIEND *****/
			mpi_errno = a2aV_send_own_off_counts(t, sendcounts, cmn_nbh_mat, comm_ptr,
											&(all_reqs[all_reqs_idx++]), &own_off_counts);

			mpi_errno = MPIR_Waitall_impl(all_reqs_idx, all_reqs, MPI_STATUS_IGNORE);
			if (mpi_errno) MPIR_ERR_POP(mpi_errno);
			all_reqs_idx = 0; /* set index back to zero for future use */

#ifdef SHM_SCHED_DEBUG
			print_vect(comm_ptr->rank, cmn_nbh_mat->num_onloaded[t], friend_off_counts, "friend_off_counts: ");
			print_vect(comm_ptr->rank, cmn_nbh_mat->num_offloaded[t], own_off_counts, "own_off_counts: ");
#endif

			/***** EXCHANGE DATA WITH FRIEND *****/
			mpi_errno = a2aV_exchange_data_with_friend(t, persistent_coll, sendbuf, sendcounts, sdispls,
									  sendtype, sendtype_extent, sendtype_max_extent,
									  own_off_counts, friend_off_counts, cmn_nbh_mat,
									  comm_ptr, s, &exchange_sendbuf, &exchange_recvbuf);
			if(!persistent_coll)
			{
				sched_bufs_from_funcs[sched_bufs_from_funcs_count++] = exchange_sendbuf;
				sched_bufs_from_funcs[sched_bufs_from_funcs_count++] = exchange_recvbuf;
			}

			MPIR_SCHED_BARRIER(s);

			/* allocate and populate all combined messages */
			a2aV_make_all_combined_msgs(t, persistent_coll, exchange_recvbuf, friend_off_counts,
								   sendbuf, sendcounts, sdispls, dests,
								   sendtype, sendtype_extent, sendtype_max_extent,
								   cmn_nbh_mat, comm_ptr, s, &combined_sendbuf);
			if(!persistent_coll)
				sched_bufs_from_funcs[sched_bufs_from_funcs_count++] = combined_sendbuf;

			MPIR_SCHED_BARRIER(s);

			/* Schedule for the outgoing combined messages
             * (CANNOT be mixed with combined buffer building)*/
			a2aV_send_to_onloaded_nbrs(t, combined_sendbuf, friend_off_counts,
									   sendcounts, sdispls, dests,
									   sendtype, sendtype_extent,
									   cmn_nbh_mat, comm_ptr, s);
			/* We can (should) free own and friend's off-counts arrays
			 * because they are not part of the schedule buffers. */
			MPL_free(own_off_counts);
			MPL_free(friend_off_counts);
        }

        /* Compiling the incom_sched_mat */
        /* Scheduling necessary recv operations */
        char *rb = (char*)incom_tmp_buf;
        for(i = 0; i < indegree; i++)
        {
            if(!incom_sched_mat[i][0]) /* On incoming neighbors */
            {
                if(incom_sched_mat[i][1] == t)
                {
                    /* Schedule a receive from the corresponding source. */
                    int incom_recv_count = a2aV_find_incom_recv_count(incom_sched_mat[i], srcs,
                    										   recvcounts, indegree);
                    mpi_errno = MPIR_Sched_recv(rb, incom_recv_count, recvtype,
                    							incom_sched_mat[i][COMB_LIST_START_IDX],
                    							comm_ptr, s);
                    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
#ifdef SHM_SCHED_DEBUG
                    /***** FOR DEBUGGING ONLY *****/
                    MPIR_Request *req_ptr = NULL;
                    int context_offset = (comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM) ?
                                                      MPIR_CONTEXT_INTRA_COLL : MPIR_CONTEXT_INTER_COLL;
                    mpi_errno = MPID_Irecv(rb, incom_recv_count, recvtype,
                                           incom_sched_mat[i][COMB_LIST_START_IDX],
                                           1000, comm_ptr, context_offset, &req_ptr);
                    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
                    debug_glob_all_reqs[debug_glob_all_reqs_idx++] = req_ptr->handle;
#endif
                    rb += incom_recv_count * recvtype_extent;
                }
            }
            else /* Off incoming neighbors */
            {
                /* Nothing to do with no cumulative. */
            }
        }

        MPIR_SCHED_BARRIER(s);

#ifdef SHM_SCHED_DEBUG
        /***** FOR DEBUGGING ONLY *****/
        mpi_errno = MPIR_Waitall_impl(debug_glob_all_reqs_idx, debug_glob_all_reqs, MPI_STATUS_IGNORE);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
        debug_glob_all_reqs_idx = 0; /* set index back to zero for future use */
        print_vect(comm_ptr->rank, indegree, (int*)incom_tmp_buf, "incom_tmp_buf: ");
#endif
        /* Scheduling message copies from incom_tmp_buf to final recv_buf */
        char *copy_from = (char*)incom_tmp_buf;
        char *copy_to;
        for(i = 0; i < indegree; i++)
        {
            if(!incom_sched_mat[i][0]) /* On incoming neighbors */
            {
                if(incom_sched_mat[i][1] == t)
                {
                    for(j = 0; j < incom_sched_mat[i][2]; j++)
                    {
                        int nbr_idx = find_in_arr(srcs, indegree, incom_sched_mat[i][j + COMB_LIST_START_IDX]);
#ifdef SHM_SCHED_DEBUG
                        sprintf(content, "t = %d, incom_nbr = %d, nbr_idx = %d, "\
                                "recvcounts[nbr_idx] = %d, rdispls[nbr_idx] = %d\n",
                                t, incom_sched_mat[i][j + COMB_LIST_START_IDX], nbr_idx,
                                recvcounts[nbr_idx], rdispls[nbr_idx]);
                        print_in_file(comm_ptr->rank, content);
#endif
                        copy_to = ((char*)recvbuf) + rdispls[nbr_idx] * recvtype_extent;
                        MPIR_Sched_copy(copy_from, recvcounts[nbr_idx], recvtype,
                                        copy_to, recvcounts[nbr_idx], recvtype, s);
#ifdef SHM_SCHED_DEBUG
                        /***** FOR DEBUGGING ONLY *****/
                        MPIR_Localcopy(copy_from, recvcounts[nbr_idx], recvtype,
                                       copy_to, recvcounts[nbr_idx], recvtype);
#endif
                        copy_from += recvcounts[nbr_idx] * recvtype_extent;
                    }
                }
            }
        }

        MPIR_SCHED_BARRIER(s);
    }

#ifdef SHM_SCHED_DEBUG
    /***** FOR DEBUGGING ONLY *****/
	print_vect(comm_ptr->rank, indegree, (int*)recvbuf, "recvbuf before residual send/recvs: ");
#endif

	/* Schedule a send for any residual outgoing neighbors */
    MPIR_SCHED_BARRIER(s);
    for(i = 0; i < cmn_nbh_mat->num_rows; i++)
    {
        if(!cmn_nbh_mat->ignore_row[i]) /* out neighbor still active */
        {
            mpi_errno = MPIR_Sched_send(((char*)sendbuf) + sdispls[i] * sendtype_extent,
            							sendcounts[i], sendtype, dests[i], comm_ptr, s);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);
#ifdef SHM_SCHED_DEBUG
            /***** FOR DEBUGGING ONLY *****/
			int context_offset = (comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM) ?
									  MPIR_CONTEXT_INTRA_COLL : MPIR_CONTEXT_INTER_COLL;
			MPIR_Request *req_ptr = NULL;
			mpi_errno = MPID_Isend(((char*)sendbuf) + sdispls[i] * sendtype_extent,
								   sendcounts[i], sendtype, dests[i],
								   1000, comm_ptr, context_offset, &req_ptr);
			if (mpi_errno) MPIR_ERR_POP(mpi_errno);
			debug_glob_all_reqs[debug_glob_all_reqs_idx++] = req_ptr->handle;
#endif
        }
    }

    /* Schedule a recv for any residual incoming neighbors */
	/* NOTE: the incoming message can still be a combined one.
	 * TODO: We might be able to use 't' to schedule these
	 * residual recvs in a stepwise fashion to create some
	 * overlap among the recv and copy operations.
	 */
    char *rb = (char*)incom_tmp_buf;
    for(i = 0; i < indegree; i++)
    {
        /* ON incoming neighbors not covered before */
        if(!incom_sched_mat[i][0] && incom_sched_mat[i][1] >= cmn_nbh_mat->t)
        {
            /* Schedule a receive from the corresponding source. */
        	int incom_recv_count = a2aV_find_incom_recv_count(incom_sched_mat[i], srcs,
        												   recvcounts, indegree);
            mpi_errno = MPIR_Sched_recv(rb, incom_recv_count, recvtype,
            							incom_sched_mat[i][COMB_LIST_START_IDX], comm_ptr, s);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);
#ifdef SHM_SCHED_DEBUG
            /***** FOR DEBUGGING ONLY *****/
			MPIR_Request *req_ptr = NULL;
			int context_offset = (comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM) ?
											  MPIR_CONTEXT_INTRA_COLL : MPIR_CONTEXT_INTER_COLL;
			mpi_errno = MPID_Irecv(rb, incom_recv_count, recvtype,
								   incom_sched_mat[i][COMB_LIST_START_IDX],
								   1000, comm_ptr, context_offset, &req_ptr);
			if (mpi_errno) MPIR_ERR_POP(mpi_errno);
			debug_glob_all_reqs[debug_glob_all_reqs_idx++] = req_ptr->handle;
#endif
            rb += incom_recv_count * recvtype_extent;
        }
        else /* OFF incoming neighbors */
        {
            /* Nothing to do with no cumulative. */
        }
    }

    MPIR_SCHED_BARRIER(s);

#ifdef SHM_SCHED_DEBUG
    /***** FOR DEBUGGING ONLY *****/
	mpi_errno = MPIR_Waitall_impl(debug_glob_all_reqs_idx, debug_glob_all_reqs, MPI_STATUS_IGNORE);
	if (mpi_errno) MPIR_ERR_POP(mpi_errno);
	debug_glob_all_reqs_idx = 0; /* set index back to zero for future use */
	print_vect(comm_ptr->rank, indegree, (int*)incom_tmp_buf,
			   "incom_tmp_buf after residual recvs: ");
#endif

    /* Scheduling message copies from incom_tmp_buf to final recv_buf */
    char *copy_from = (char*)incom_tmp_buf;
	char *copy_to;
    for(i = 0; i < indegree; i++)
    {
        /* ON incoming neighbors not covered before */
        if(!incom_sched_mat[i][0] && incom_sched_mat[i][1] >= cmn_nbh_mat->t)
        {
            for(j = 0; j < incom_sched_mat[i][2]; j++)
            {
                int nbr_idx = find_in_arr(srcs, indegree, incom_sched_mat[i][j + COMB_LIST_START_IDX]);
                copy_to = ((char*)recvbuf) + rdispls[nbr_idx] * recvtype_extent;
                MPIR_Sched_copy(copy_from, recvcounts[nbr_idx], recvtype,
                                copy_to, recvcounts[nbr_idx], recvtype, s);
#ifdef SHM_SCHED_DEBUG
                /***** FOR DEBUGGING ONLY *****/
                MPIR_Localcopy(copy_from, recvcounts[nbr_idx], recvtype,
							   copy_to, recvcounts[nbr_idx], recvtype);
#endif
                copy_from += recvcounts[nbr_idx] * recvtype_extent;
            }
        }
    }

    MPIR_SCHED_BARRIER(s);

#ifdef SHM_SCHED_DEBUG
    /***** FOR DEBUGGING ONLY *****/
    print_vect(comm_ptr->rank, indegree, (int*)recvbuf, "Final recvbuf: ");
#endif

    if(persistent_coll)
        MPIR_CHKPMEM_COMMIT();
    else
    {
        MPIR_SCHED_CHKPMEM_COMMIT(s);
		while (sched_bufs_from_funcs_count > 0)
		{
			/* Schedule to free all bufs allocated in functions */
			mpi_errno = MPIR_Sched_cb(&MPIR_Sched_cb_free_buf,
									  (sched_bufs_from_funcs[--sched_bufs_from_funcs_count]),
									  s);
			if (mpi_errno) MPIR_ERR_POP(mpi_errno);
		}
    }

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

#ifdef SHM_DEBUG
    sprintf(content, "Done with building the schedule.\n");
    print_in_file(comm_ptr->rank, content);
#endif

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

//SHM
#undef FUNCNAME
#define FUNCNAME make_all_combined_msgs
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int a2aV_make_all_combined_msgs(int t, int persistent_coll, void *exchange_recvbuf, int *friend_off_counts,
                          const void *sendbuf, const int *sendcounts, const int *sdispls, const int *dests,
                          MPI_Datatype sendtype, MPI_Aint sendtype_extent,
                          MPI_Aint sendtype_max_extent, Common_nbrhood_matrix *cmn_nbh_mat,
                          MPIR_Comm *comm_ptr, MPIR_Sched_t s, void **combined_sendbuf_ptr)
{
#ifdef SHM_SCHED_DEBUG
    char content[256];
#endif
	MPIR_CHKPMEM_DECL(1);
	MPIR_SCHED_CHKPMEM_DECL(1);
	int mpi_errno = MPI_SUCCESS;
    int i, dest_idx, f_displ = 0, fc_idx = 0;
    MPIR_Topology *topo_ptr = NULL;
	topo_ptr = MPIR_Topology_get(comm_ptr);
    int num_onloaded = cmn_nbh_mat->num_onloaded[t];
    int onload_start = cmn_nbh_mat->sorted_cmn_nbrs[t].onload_start;
	int onload_end = cmn_nbh_mat->sorted_cmn_nbrs[t].onload_end;
	/* allocate combined_sendbuf */
	int combined_sendbuf_count = 0;
	MPI_Aint combined_sendbuf_extent = 0;
    for(i = onload_start; i <= onload_end; i++)
    {
    	dest_idx = cmn_nbh_mat->sorted_cmn_nbrs[t].cmn_nbrs[i].index;
    	combined_sendbuf_count += sendcounts[dest_idx] + friend_off_counts[fc_idx];
    	fc_idx++;
    }
#ifdef SHM_SCHED_DEBUG
    sprintf(content, "combined_send_count = %d (for all onloaded at t = %d)\n", combined_sendbuf_count, t);
    print_in_file(comm_ptr->rank, content);
#endif

    combined_sendbuf_extent = combined_sendbuf_count * sendtype_max_extent;

    void *combined_sendbuf = NULL;
	if(persistent_coll)
	{
		MPIR_CHKPMEM_MALLOC(combined_sendbuf, void*, combined_sendbuf_extent, mpi_errno, "combined_sendbuf");
		topo_ptr->topo.dist_graph.sched_mem_to_free[topo_ptr->topo.dist_graph.sched_mem_to_free_num_entries++] = combined_sendbuf;
	}
	else /* not persistent schedule */
	{
		MPIR_SCHED_CHKPMEM_MALLOC(combined_sendbuf, void*, combined_sendbuf_extent, mpi_errno, "combined_sendbuf");
	}
	*(char**)combined_sendbuf_ptr = combined_sendbuf; /* to return to the caller */
#ifdef SHM_SCHED_DEBUG
	/***** FOR DEBUGGING ONLY *****/
	print_vect(comm_ptr->rank, find_array_sum(friend_off_counts, num_onloaded), exchange_recvbuf, "exchange_recvbuf in make_all_combined: ");
#endif
	/**** pack the combined message from sendbuf and exchange_recvbuf into combined_buf ****/
	char *copy_from;
	char *copy_to = ((char*)combined_sendbuf);
	f_displ = fc_idx = 0;
	for(i = onload_start; i <= onload_end; i++)
    {
		dest_idx = cmn_nbh_mat->sorted_cmn_nbrs[t].cmn_nbrs[i].index;
		/* copy own data first */
		copy_from = ((char*)sendbuf) + sdispls[dest_idx] * sendtype_extent;
		MPIR_Sched_copy(copy_from, sendcounts[dest_idx], sendtype,
						copy_to, sendcounts[dest_idx], sendtype, s);
#ifdef SHM_SCHED_DEBUG
		/***** FOR DEBUGGING ONLY *****/
		MPIR_Localcopy(copy_from, sendcounts[dest_idx], sendtype,
                       copy_to, sendcounts[dest_idx], sendtype);
#endif
		copy_to += sendcounts[dest_idx] * sendtype_extent;
		/* append friend's data */
		copy_from = ((char*)exchange_recvbuf) + f_displ * sendtype_extent;
		MPIR_Sched_copy(copy_from, friend_off_counts[fc_idx], sendtype,
						copy_to, friend_off_counts[fc_idx], sendtype, s);
#ifdef SHM_SCHED_DEBUG
		/***** FOR DEBUGGING ONLY *****/
		MPIR_Localcopy(copy_from, friend_off_counts[fc_idx], sendtype,
                        copy_to, friend_off_counts[fc_idx], sendtype);
#endif
		copy_to += friend_off_counts[fc_idx] * sendtype_extent;
		f_displ += friend_off_counts[fc_idx]; /* increase displacement in exchange_recvbuf
                                                 for next iteration */
		fc_idx++;
		/**** End of packing ****/
    }
#ifdef SHM_SCHED_DEBUG
	/***** FOR DEBUGGING ONLY *****/
    print_vect(comm_ptr->rank, combined_sendbuf_count, (int*)combined_sendbuf, "combined_sendbuf in make_all_combined: ");
#endif

    if(persistent_coll)
    	MPIR_CHKPMEM_COMMIT();
    /*DO NOT call MPIR_SCHED_CHKPMEM_COMMIT(s) here for the
     * non-persistent case. It will schedule to free the
     * schedule buffers, which should only be done at the end
     * of the schedule building function.
     */

fn_exit:
	return mpi_errno;
fn_fail:
	if(persistent_coll)
	{
		MPIR_CHKPMEM_REAP();
		topo_ptr->topo.dist_graph.sched_mem_to_free_num_entries--;
	}
	else
	{
		MPIR_SCHED_CHKPMEM_REAP(s);
	}
	goto fn_exit;
}

//SHM
#undef FUNCNAME
#define FUNCNAME send_to_onloaded_nbrs
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int a2aV_send_to_onloaded_nbrs(int t, void *combined_sendbuf, int *friend_off_counts,
						  	   const int *sendcounts, const int *sdispls, const int *dests,
						  	   MPI_Datatype sendtype, MPI_Aint sendtype_extent,
						  	   Common_nbrhood_matrix *cmn_nbh_mat, MPIR_Comm *comm_ptr,
						  	   MPIR_Sched_t s)
{
	int mpi_errno = MPI_SUCCESS;
    int i, dest_idx, fc_idx = 0;
    MPIR_Topology *topo_ptr = NULL;
	topo_ptr = MPIR_Topology_get(comm_ptr);
    int num_onloaded = cmn_nbh_mat->num_onloaded[t];
    int onload_start = cmn_nbh_mat->sorted_cmn_nbrs[t].onload_start;
	int onload_end = cmn_nbh_mat->sorted_cmn_nbrs[t].onload_end;

	char *send_from = (char*)combined_sendbuf;
	int combined_sendcount;
	fc_idx = 0;
    for(i = onload_start; i <= onload_end; i++)
    {
    	dest_idx = cmn_nbh_mat->sorted_cmn_nbrs[t].cmn_nbrs[i].index;
    	combined_sendcount = sendcounts[dest_idx] + friend_off_counts[fc_idx];
		mpi_errno = MPIR_Sched_send(send_from, combined_sendcount,
									sendtype, dests[dest_idx], comm_ptr, s);
		if (mpi_errno) MPIR_ERR_POP(mpi_errno);
#ifdef SHM_SCHED_DEBUG
		/***** FOR DEBUGGING ONLY *****/
        int context_offset = (comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM) ?
                                  MPIR_CONTEXT_INTRA_COLL : MPIR_CONTEXT_INTER_COLL;
        MPIR_Request *req_ptr = NULL;
        mpi_errno = MPID_Isend(send_from, combined_sendcount,
                               sendtype, dests[dest_idx],
                               1000, comm_ptr, context_offset, &req_ptr);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
        debug_glob_all_reqs[debug_glob_all_reqs_idx++] = req_ptr->handle;

		char tmp_string[256];
		sprintf(tmp_string, "buffer sent to %d: ", dests[dest_idx]);
		print_vect(comm_ptr->rank, combined_sendcount, (int*)send_from, tmp_string);
#endif
		send_from += combined_sendcount * sendtype_extent;
		fc_idx++;
    }

fn_exit:
	return mpi_errno;
fn_fail:
	goto fn_exit;
}

//SHM
#undef FUNCNAME
#define FUNCNAME send_own_off_counts
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int a2aV_send_own_off_counts(int t, const int *sendcounts, Common_nbrhood_matrix *cmn_nbh_mat,
						MPIR_Comm *comm_ptr, MPI_Request *mpi_req_ptr, int **own_off_counts_ptr)
{
    /***** PACK AND SEND OWN_OFF_COUNTS TO FRIEND *****/
	MPIR_CHKPMEM_DECL(1);
    int mpi_errno = MPI_SUCCESS;
    int i, dest_idx, idx = 0;
    int num_offloaded = cmn_nbh_mat->num_offloaded[t];
    int offload_start = cmn_nbh_mat->sorted_cmn_nbrs[t].offload_start;
    int offload_end = cmn_nbh_mat->sorted_cmn_nbrs[t].offload_end;
    /* find friend rank from any of onloaded/offloaded neighbor entries in comb_matrix (at t) */
    int tmp_idx = cmn_nbh_mat->sorted_cmn_nbrs[t].cmn_nbrs[offload_start].index;
    int friend = cmn_nbh_mat->comb_matrix[tmp_idx][t].paired_frnd;
    /* allocate own ranks_counts. size is 2*num_offloaded integers */
    int *own_off_counts;
    MPIR_CHKPMEM_MALLOC(own_off_counts, int*, num_offloaded * sizeof(int), mpi_errno, "own_off_counts");
    *own_off_counts_ptr = own_off_counts; /* to return to the caller */
    /* populate it from dsts and sendcounts arrays */
    for(i = offload_start; i <= offload_end; i++)
    {
    	dest_idx = cmn_nbh_mat->sorted_cmn_nbrs[t].cmn_nbrs[i].index;
		own_off_counts[idx++] = sendcounts[dest_idx];
    }
    /* send own_off_counts to the friend */
    int context_offset = (comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM) ?
    					  MPIR_CONTEXT_INTRA_COLL : MPIR_CONTEXT_INTER_COLL;
    MPIR_Request *req_ptr = NULL;
    mpi_errno = MPID_Isend(own_off_counts, num_offloaded, MPI_INT, friend,
						   1000, comm_ptr, context_offset, &req_ptr);
	if (mpi_errno) MPIR_ERR_POP(mpi_errno);
	*mpi_req_ptr = req_ptr->handle;

	MPIR_CHKPMEM_COMMIT();

fn_exit:
    return mpi_errno;
fn_fail:
	MPIR_CHKPMEM_REAP();
    goto fn_exit;
}

//SHM
#undef FUNCNAME
#define FUNCNAME get_friend_off_counts
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int a2aV_get_friend_off_counts(int t, Common_nbrhood_matrix *cmn_nbh_mat, MPIR_Comm *comm_ptr,
						  MPI_Request *mpi_req_ptr, int **friend_off_counts_ptr)
{
    /***** GET friend_off_counts *****/
	MPIR_CHKPMEM_DECL(1);
    int mpi_errno = MPI_SUCCESS;
    int i;
    int num_onloaded = cmn_nbh_mat->num_onloaded[t];
    int onload_start = cmn_nbh_mat->sorted_cmn_nbrs[t].onload_start;
	int onload_end = cmn_nbh_mat->sorted_cmn_nbrs[t].onload_end;
	/* find friend rank from any of onloaded/offloaded neighbor entries in comb_matrix (at t) */
	int tmp_idx = cmn_nbh_mat->sorted_cmn_nbrs[t].cmn_nbrs[onload_start].index;
	int friend = cmn_nbh_mat->comb_matrix[tmp_idx][t].paired_frnd;
    /* allocate ranks+counts. size is 2*num_onloaded integers */
    int *friend_off_counts;
    MPIR_CHKPMEM_MALLOC(friend_off_counts, int*, num_onloaded * sizeof(int), mpi_errno, "friend_off_counts");
    *friend_off_counts_ptr = friend_off_counts; /* to return to the caller */
    /* recv from friend into friend_off_counts */
    int context_offset = (comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM) ?
						  MPIR_CONTEXT_INTRA_COLL : MPIR_CONTEXT_INTER_COLL;
	MPIR_Request *req_ptr = NULL;
	mpi_errno = MPID_Irecv(friend_off_counts, num_onloaded, MPI_INT, friend,
						   1000, comm_ptr, context_offset, &req_ptr);
	if (mpi_errno) MPIR_ERR_POP(mpi_errno);
	*mpi_req_ptr = req_ptr->handle;

	MPIR_CHKPMEM_COMMIT();

fn_exit:
    return mpi_errno;
fn_fail:
    MPIR_CHKPMEM_REAP();
    goto fn_exit;
}

//SHM
#undef FUNCNAME
#define FUNCNAME send_own_data
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int a2aV_exchange_data_with_friend(int t, int persistent_coll, const void *sendbuf, const int *sendcounts, const int *sdispls,
                  MPI_Datatype sendtype, MPI_Aint sendtype_extent, MPI_Aint sendtype_max_extent,
                  int *own_off_counts, int *friend_off_counts, Common_nbrhood_matrix *cmn_nbh_mat, MPIR_Comm *comm_ptr,
                  MPIR_Sched_t s, void **exchange_sendbuf_ptr, void **exchange_recvbuf_ptr)
{
	MPIR_CHKPMEM_DECL(2);
	MPIR_SCHED_CHKPMEM_DECL(2);
    int mpi_errno = MPI_SUCCESS;
    int i, dest_idx;
    MPIR_Topology *topo_ptr = NULL;
	topo_ptr = MPIR_Topology_get(comm_ptr);
    int num_offloaded = cmn_nbh_mat->num_offloaded[t];
    int offload_start = cmn_nbh_mat->sorted_cmn_nbrs[t].offload_start;
	int offload_end = cmn_nbh_mat->sorted_cmn_nbrs[t].offload_end;
	int num_onloaded = cmn_nbh_mat->num_onloaded[t];
	int onload_start = cmn_nbh_mat->sorted_cmn_nbrs[t].onload_start;
	int onload_end = cmn_nbh_mat->sorted_cmn_nbrs[t].onload_end;
	/* find friend rank from any of onloaded/offloaded neighbor entries in comb_matrix (at t) */
	int tmp_idx = cmn_nbh_mat->sorted_cmn_nbrs[t].cmn_nbrs[offload_start].index;
	int friend = cmn_nbh_mat->comb_matrix[tmp_idx][t].paired_frnd;
    /* find exchange_sendbuf_extent */
    int own_off_counts_sum = find_array_sum(own_off_counts, num_offloaded);
    MPI_Aint exchange_sendbuf_extent = own_off_counts_sum * sendtype_max_extent;
    /* find exchange_recvbuf_extent */
	int friend_off_counts_sum = find_array_sum(friend_off_counts, num_onloaded);
	MPI_Aint exchange_recvbuf_extent = friend_off_counts_sum * sendtype_max_extent;
    /* allocate exchange_sendbuf and exchange_recvbuf */
    void *exchange_sendbuf = NULL;
    void *exchange_recvbuf = NULL;
    if(persistent_coll)
    {
        MPIR_CHKPMEM_MALLOC(exchange_sendbuf, void*, exchange_sendbuf_extent, mpi_errno, "exchange_sendbuf");
        topo_ptr->topo.dist_graph.sched_mem_to_free[topo_ptr->topo.dist_graph.sched_mem_to_free_num_entries++] = exchange_sendbuf;
        MPIR_CHKPMEM_MALLOC(exchange_recvbuf, void*, exchange_recvbuf_extent, mpi_errno, "exchange_recvbuf");
		topo_ptr->topo.dist_graph.sched_mem_to_free[topo_ptr->topo.dist_graph.sched_mem_to_free_num_entries++] = exchange_recvbuf;
    }
    else /* not persistent schedule */
    {
        MPIR_SCHED_CHKPMEM_MALLOC(exchange_sendbuf, void*, exchange_sendbuf_extent, mpi_errno, "exchange_sendbuf");
        MPIR_SCHED_CHKPMEM_MALLOC(exchange_recvbuf, void*, exchange_recvbuf_extent, mpi_errno, "exchange_recvbuf");
    }
    *(char**)exchange_sendbuf_ptr = exchange_sendbuf; /* to return to the caller */
    *(char**)exchange_recvbuf_ptr = exchange_recvbuf; /* to return to the caller */
    /* populate exchange_send_buf from sendbuf */
    char *copy_from;
    char *copy_to = (char*)exchange_sendbuf;
    for(i = offload_start; i <= offload_end; i++)
    {
    	dest_idx = cmn_nbh_mat->sorted_cmn_nbrs[t].cmn_nbrs[i].index;
		copy_from = ((char*)sendbuf) + sdispls[dest_idx] * sendtype_extent;
		MPIR_Sched_copy(copy_from, sendcounts[dest_idx], sendtype,
						copy_to, sendcounts[dest_idx], sendtype, s);
		if(mpi_errno) MPIR_ERR_POP(mpi_errno);
#ifdef SHM_SCHED_DEBUG
		/***** FOR DEBUGGING ONLY *****/
		MPIR_Localcopy(copy_from, sendcounts[dest_idx], sendtype,
					   copy_to, sendcounts[dest_idx], sendtype);
#endif
		copy_to += sendcounts[dest_idx] * sendtype_extent;
    }

    MPIR_SCHED_BARRIER(s);
#ifdef SHM_SCHED_DEBUG
    /***** FOR DEBUGGING ONLY *****/
    int all_reqs_idx = 0;
    MPI_Request all_reqs[2];
    int context_offset = (comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM) ?
                              MPIR_CONTEXT_INTRA_COLL : MPIR_CONTEXT_INTER_COLL;
    MPIR_Request *req_ptr = NULL;
    mpi_errno = MPID_Isend(exchange_sendbuf, own_off_counts_sum, sendtype, friend,
                           1000, comm_ptr, context_offset, &req_ptr);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    all_reqs[all_reqs_idx++] = req_ptr->handle;

    mpi_errno = MPID_Irecv(exchange_recvbuf, friend_off_counts_sum, sendtype, friend,
                           1000, comm_ptr, context_offset, &req_ptr);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    all_reqs[all_reqs_idx++] = req_ptr->handle;

    mpi_errno = MPIR_Waitall_impl(all_reqs_idx, all_reqs, MPI_STATUS_IGNORE);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    all_reqs_idx = 0; /* set index back to zero for future use */

    print_vect(comm_ptr->rank, own_off_counts_sum, (int*)exchange_sendbuf, "exchange_sendbuf: ");
    print_vect(comm_ptr->rank, friend_off_counts_sum, (int*)exchange_recvbuf, "exchange_recvbuf: ");
#endif
    /* send exchange_sendbuf to friend
     * - type is obviously sendtype.
     * - count is own_off_counts_count_sum.
     */
    mpi_errno = MPIR_Sched_send(exchange_sendbuf, own_off_counts_sum, sendtype, friend, comm_ptr, s);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    /* recv into exchange_recvbuf from friend
     * - type is sendtype because the incoming data was
     *   supposed to be sent to neighbors common to this rank.
     * - count is friend_off_counts_count_sum.
     */
	mpi_errno = MPIR_Sched_recv(exchange_recvbuf, friend_off_counts_sum, sendtype, friend, comm_ptr, s);
	if (mpi_errno) MPIR_ERR_POP(mpi_errno);

    if(persistent_coll)
    	MPIR_CHKPMEM_COMMIT();
    /*DO NOT call MPIR_SCHED_CHKPMEM_COMMIT(s) here for the
	 * non-persistent case. It will schedule to free the
	 * schedule buffers, which should only be done at the end
	 * of the schedule building function.
	 */

fn_exit:
    return mpi_errno;
fn_fail:
    if(persistent_coll)
    {
        MPIR_CHKPMEM_REAP();
        topo_ptr->topo.dist_graph.sched_mem_to_free_num_entries -= 2;
    }
    else
    {
        MPIR_SCHED_CHKPMEM_REAP(s);
    }
    goto fn_exit;
}

//SHM
int a2aV_find_incom_recv_count(int *incom_sched_vec, int *srcs, const int *recvcounts, int indegree)
{
	int i, nbr_idx, sum = 0;
	for(i = 0; i < incom_sched_vec[2]; i++)
	{
		nbr_idx = find_in_arr(srcs, indegree, incom_sched_vec[i + COMB_LIST_START_IDX]);
		sum += recvcounts[nbr_idx];
	}
	return sum;
}
/***** End SHM ******/

#undef FUNCNAME
#define FUNCNAME MPIR_Ineighbor_alltoallv_default
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Ineighbor_alltoallv_default(const void *sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int rdispls[], MPI_Datatype recvtype, MPIR_Comm *comm_ptr, MPIR_Sched_t s)
{
    int mpi_errno = MPI_SUCCESS;
    int indegree, outdegree, weighted;
    int i, k,l;
    int *srcs, *dsts;
    int comm_size;
    MPI_Aint sendtype_extent, recvtype_extent;
    MPIR_CHKLMEM_DECL(2);

    comm_size = comm_ptr->local_size;

    MPID_Datatype_get_extent_macro(sendtype, sendtype_extent);
    MPID_Datatype_get_extent_macro(recvtype, recvtype_extent);

    for (i = 0; i < comm_size; ++i) {
        MPIR_Ensure_Aint_fits_in_pointer(MPIR_VOID_PTR_CAST_TO_MPI_AINT sendbuf +
                                         (sdispls[i] * sendtype_extent));
        MPIR_Ensure_Aint_fits_in_pointer(MPIR_VOID_PTR_CAST_TO_MPI_AINT recvbuf +
                                         (rdispls[i] * recvtype_extent));
    }

    mpi_errno = MPIR_Topo_canon_nhb_count(comm_ptr, &indegree, &outdegree, &weighted);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    MPIR_CHKLMEM_MALLOC(srcs, int *, indegree*sizeof(int), mpi_errno, "srcs");
    MPIR_CHKLMEM_MALLOC(dsts, int *, outdegree*sizeof(int), mpi_errno, "dsts");
    mpi_errno = MPIR_Topo_canon_nhb(comm_ptr,
                                    indegree, srcs, MPI_UNWEIGHTED,
                                    outdegree, dsts, MPI_UNWEIGHTED);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

    for (k = 0; k < outdegree; ++k) {
        char *sb = ((char *)sendbuf) + sdispls[k] * sendtype_extent;
        mpi_errno = MPIR_Sched_send(sb, sendcounts[k], sendtype, dsts[k], comm_ptr, s);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    }

    for (l = 0; l < indegree; ++l) {
        char *rb = ((char *)recvbuf) + rdispls[l] * recvtype_extent;
        mpi_errno = MPIR_Sched_recv(rb, recvcounts[l], recvtype, srcs[l], comm_ptr, s);
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
#define FUNCNAME MPIR_Ineighbor_alltoallv_impl
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Ineighbor_alltoallv_impl(const void *sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int rdispls[], MPI_Datatype recvtype, MPIR_Comm *comm_ptr, MPI_Request *request)
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
    MPIR_Assert(comm_ptr->coll_fns->Ineighbor_alltoallv != NULL);
    //SHM
	if(nbr_impl == 1) /* Use SHM implementation--This is not the right way of doing it */
		mpi_errno = MPIR_Ineighbor_alltoallv_SHM(sendbuf, sendcounts, sdispls, sendtype,
												 recvbuf, recvcounts, rdispls, recvtype,
												 comm_ptr, s, 0); /* 0 to denote a non-persistent call */
	else
		mpi_errno = comm_ptr->coll_fns->Ineighbor_alltoallv(sendbuf, sendcounts, sdispls, sendtype,
                                                        recvbuf, recvcounts, rdispls, recvtype,
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
#define FUNCNAME MPI_Ineighbor_alltoallv
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
/*@
MPI_Ineighbor_alltoallv - Nonblocking version of MPI_Neighbor_alltoallv.

Input Parameters:
+ sendbuf - starting address of the send buffer (choice)
. sendcounts - non-negative integer array (of length outdegree) specifying the number of elements to send to each neighbor
. sdispls - integer array (of length outdegree).  Entry j specifies the displacement (relative to sendbuf) from which to send the outgoing data to neighbor j
. sendtype - data type of send buffer elements (handle)
. recvcounts - non-negative integer array (of length indegree) specifying the number of elements that are received from each neighbor
. rdispls - integer array (of length indegree).  Entry i specifies the displacement (relative to recvbuf) at which to place the incoming data from neighbor i.
. recvtype - data type of receive buffer elements (handle)
- comm - communicator with topology structure (handle)

Output Parameters:
+ recvbuf - starting address of the receive buffer (choice)
- request - communication request (handle)

.N ThreadSafe

.N Fortran

.N Errors
@*/
int MPI_Ineighbor_alltoallv(const void *sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_Comm *comm_ptr = NULL;
    MPIR_FUNC_TERSE_STATE_DECL(MPID_STATE_MPI_INEIGHBOR_ALLTOALLV);

    MPID_THREAD_CS_ENTER(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    MPIR_FUNC_TERSE_ENTER(MPID_STATE_MPI_INEIGHBOR_ALLTOALLV);

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
            MPIR_ERRTEST_ARGNULL(request, "request", mpi_errno);
            /* TODO more checks may be appropriate (counts, in_place, buffer aliasing, etc) */
        }
        MPID_END_ERROR_CHECKS
    }
#   endif /* HAVE_ERROR_CHECKING */

    /* ... body of routine ...  */

    mpi_errno = MPID_Ineighbor_alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm_ptr, request);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);

    /* ... end of body of routine ... */

fn_exit:
    MPIR_FUNC_TERSE_EXIT(MPID_STATE_MPI_INEIGHBOR_ALLTOALLV);
    MPID_THREAD_CS_EXIT(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    return mpi_errno;

fn_fail:
    /* --BEGIN ERROR HANDLING-- */
#   ifdef HAVE_ERROR_CHECKING
    {
        mpi_errno = MPIR_Err_create_code(
            mpi_errno, MPIR_ERR_RECOVERABLE, FCNAME, __LINE__, MPI_ERR_OTHER,
            "**mpi_ineighbor_alltoallv", "**mpi_ineighbor_alltoallv %p %p %p %D %p %p %p %D %C %p", sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm, request);
    }
#   endif
    mpi_errno = MPIR_Err_return_comm(NULL, FCNAME, mpi_errno);
    goto fn_exit;
    /* --END ERROR HANDLING-- */
}
