/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *
 *  (C) 2009 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpiimpl.h"

/*---------------------------------------------------------------------*/
/*     (C) Copyright 2017 Parallel Processing Research Laboratory      */
/*                   Queen's University at Kingston                    */
/*                Neighborhood Collective Communication                */
/*                    Seyed Hessamedin Mirsadeghi                      */
/*---------------------------------------------------------------------*/
/*
=== BEGIN_MPI_T_CVAR_INFO_BLOCK ===

cvars:
    - name        : MPIR_CVAR_NEIGHBOR_COLL_MSG_COMB_FRNDSHP_THRSHLD
      category    : COLLECTIVE
      type        : int
      default     : 4
      class       : device
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : |-
        Value of the friendship threshold used in the message
        combining algorithm for neighborhood collectives.

=== END_MPI_T_CVAR_INFO_BLOCK ===
*/

#include <math.h>
#include "heap.h"

int free_nbh_mat(Common_nbrhood_matrix *cmn_nbh_mat)
{
    if(!cmn_nbh_mat)
        return 1;
    int outdegree = cmn_nbh_mat->num_rows;
    int i, j;
    for(i = 0; i < outdegree; i++)
    {
        MPL_free(cmn_nbh_mat->matrix[i]);
        MPL_free(cmn_nbh_mat->outnbrs_innbrs_bitmap[i]);
        MPL_free(cmn_nbh_mat->comb_matrix[i]);
    }
    for(i = 0; i < cmn_nbh_mat->t; i++)
    {
        if(cmn_nbh_mat->sorted_cmn_nbrs[i].num_cmn_nbrs > 0)
            if(cmn_nbh_mat->sorted_cmn_nbrs[i].cmn_nbrs)
                MPL_free(cmn_nbh_mat->sorted_cmn_nbrs[i].cmn_nbrs);
    }
    MPL_free(cmn_nbh_mat->matrix);
    MPL_free(cmn_nbh_mat->outnbrs_innbrs_bitmap);
    MPL_free(cmn_nbh_mat->comb_matrix);
    MPL_free(cmn_nbh_mat->row_sizes);
    MPL_free(cmn_nbh_mat->ignore_row);
    MPL_free(cmn_nbh_mat->is_row_offloaded);
    MPL_free(cmn_nbh_mat->my_innbrs_bitmap);
    MPL_free(cmn_nbh_mat->comb_matrix_num_entries_in_row);

    MPL_free(cmn_nbh_mat);
    return 0;
}

int add_frnd_to_comb_matrix(Common_nbrhood_matrix *cmn_nbh_mat, int row_idx, int frnd, Operation opt)
{
    int num_entries = cmn_nbh_mat->comb_matrix_num_entries_in_row[row_idx];
    int t = cmn_nbh_mat->t;
    if(num_entries >= MAX_COMB_DEGREE)
    {
        fprintf(stderr, "ERROR: No more space to add new friend to the comb_matrix!"
                " num_entries = %d\n", num_entries);
        return -1;
    }
    if(t >= MAX_COMB_DEGREE)
    {
            fprintf(stderr, "ERROR: No more space to add a new scheduling step to the comb_matrix!"
                   " t = %d\n", t);
            return -1;
    }

    cmn_nbh_mat->comb_matrix[row_idx][t].paired_frnd = frnd;
    cmn_nbh_mat->comb_matrix[row_idx][t].opt = opt;
    num_entries++;
    cmn_nbh_mat->comb_matrix_num_entries_in_row[row_idx] = num_entries;
    return 0;
}

int print_in_file(int rank, char *name)
{
    char outfile[256];
    sprintf(outfile, "./rank_%d_SHM.txt", rank);
    FILE *fp = fopen(outfile, "a");
    if(fp == NULL)
        fprintf(stderr, "Could not open file!\n");

    int i;
    fprintf(fp, name);
    fprintf(fp, "\n");
    fclose(fp);
    return 0;
}

int find_in_arr(const int *array, int size, int value)
{
    int i;
    for(i = 0; i < size; i++)
        if(array[i] == value)
            return i;
    return -1;
}

int find_array_sum(const int *array, int size)
{
    int sum = 0;
    int i;
    for(i = 0; i < size; i++)
    {
        sum += array[i];
    }
    return sum;
}

int print_vect(int rank, int size, const int *vec, char *name)
{
    char outfile[256];
    sprintf(outfile, "./rank_%d_SHM.txt", rank);
    FILE *fp = fopen(outfile, "a");
    if(fp == NULL)
        fprintf(stderr, "Could not open file!\n");

    int i;
    fprintf(fp, name);
    fprintf(fp, "\n");

    if(vec == NULL)
    {
        fprintf(fp, "The vector pointer is NULL");
    }
    else
    {
        for(i = 0; i < size; i++)
            fprintf(fp, "%d ", vec[i]);
    }

    fprintf(fp, "\n\n");
    fclose(fp);
    return 0;
}

int print_mat(int rank, int rows, int *cols, int **mat, int width, char *name)
{
    char outfile[256];
    sprintf(outfile, "./rank_%d_SHM.txt", rank);
    FILE *fp = fopen(outfile, "a");
    if(fp == NULL)
        fprintf(stderr, "Could not open file!\n");

    int i, j;
    fprintf(fp, name);
    fprintf(fp, "\n");
    for(i = 0; i < rows; i++)
    {
        for(j = 0; j < cols[i]; j++)
        {
            fprintf(fp, "%*d ", width, mat[i][j]);
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "\n");
    fclose(fp);
    return 0;
}

int print_nbh_mat(int rank, Common_nbrhood_matrix *cmn_nbh_mat, int width, char *name)
{
    char outfile[256];
    sprintf(outfile, "./rank_%d_SHM.txt", rank);
    FILE *fp = fopen(outfile, "a");
    if(fp == NULL)
        fprintf(stderr, "Could not open file!\n");

    int i, j;
    int rows = cmn_nbh_mat->num_rows;
    int *cols = cmn_nbh_mat->row_sizes;
    int **mat = cmn_nbh_mat->matrix;
    Comb_element **comb_mat = cmn_nbh_mat->comb_matrix;
    int *comb_mat_cols = cmn_nbh_mat->comb_matrix_num_entries_in_row;
    fprintf(fp, name);
    fprintf(fp, "\n");
    for(i = 0; i < rows; i++)
    {
        for(j = 0; j < cols[i]; j++)
        {
            fprintf(fp, "%*d ", width, mat[i][j]);
        }

        if(cmn_nbh_mat->is_row_offloaded[i])
        {
            fprintf(fp, "OFF ");
        }
        else
        {
            fprintf(fp, "ON  ");
        }

        if(cmn_nbh_mat->ignore_row[i])
        {
            fprintf(fp, "P ");
        }
        else
        {
            fprintf(fp, "A ");
        }

        /* print the list of SEND/RECV combinings to/from paired friends */
        for(j = 0; j < cmn_nbh_mat->t; j++)
        {
            if(comb_mat[i][j].opt == SEND)
                fprintf(fp, "S");
            else if(comb_mat[i][j].opt == RECV)
                fprintf(fp, "R");
            else
                fprintf(fp, "I");
            fprintf(fp, "%*d", width, comb_mat[i][j].paired_frnd);
        }
        fprintf(fp, "\n");
    }

    fprintf(fp, "\n");
    fclose(fp);
    return 0;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Get_inNbrs_of_outNbrs
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Get_inNbrs_of_outNbrs(MPIR_Comm *comm_ptr, Common_nbrhood_matrix **cmn_nbh_mat_ptr)
{
    /* This function returns the matrix that gives the
     * incoming neighbors of each of my outgoing neighbors.
     * Number of rows will be equal to my outdegree, and
     * the number of elements in row i will be equal to
     * the indegree of my ith outgoing neighbor.
     */

    int mpi_errno = MPI_SUCCESS;
    int indegree, outdegree, comm_size;
    int i, j, out_idx, in_idx, all_reqs_idx, reqs_max_size, context_offset;
    comm_size = comm_ptr->local_size;
    MPIR_Topology *topo_ptr = NULL;
    topo_ptr = MPIR_Topology_get(comm_ptr);
    if(topo_ptr == NULL)
    {
        fprintf(stderr, "ERROR: Communicator topology pointer is NULL!\n");
        goto fn_fail;
    }
    indegree = topo_ptr->topo.dist_graph.indegree;
    outdegree = topo_ptr->topo.dist_graph.outdegree;

    MPIR_CHKPMEM_DECL(9);

    /* Getting the indegree of my outgoing neighbors.
     * To this end, everybody should send its indegree
     * to each of its incoming neighbors, and also receive
     * the indegree from each of its outgoing neighbors.
    */
    context_offset = (comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM) ?
                      MPIR_CONTEXT_INTRA_COLL : MPIR_CONTEXT_INTER_COLL;
    reqs_max_size = indegree + outdegree;
    all_reqs_idx = 0;
    MPI_Request *all_reqs = MPL_malloc(reqs_max_size * sizeof(MPI_Request), MPL_MEM_OBJECT);
    MPIR_Request *req_ptr = NULL;

    /* Recv buffer */
    int *outnbrs_indegree;
    MPIR_CHKPMEM_MALLOC(outnbrs_indegree, int*, outdegree * sizeof(int),
                        mpi_errno, "outnbrs_indegree", MPL_MEM_OTHER);

	/* Send the indegree to each incoming neighbor */
    for(in_idx = 0; in_idx < indegree; in_idx++) /* for each of my incoming neighbors */
	{
		mpi_errno = MPID_Isend(&indegree, 1, MPI_INT, topo_ptr->topo.dist_graph.in[in_idx],
		                       1000, comm_ptr, context_offset, &req_ptr);
		if (mpi_errno) MPIR_ERR_POP(mpi_errno);
		all_reqs[all_reqs_idx++] = req_ptr->handle;
	}
	for(out_idx = 0; out_idx < outdegree; out_idx++) /* for each of my outgoing neighbors */
	{
		mpi_errno = MPID_Irecv(&outnbrs_indegree[out_idx], 1, MPI_INT,
		                       topo_ptr->topo.dist_graph.out[out_idx],
		                       1000, comm_ptr, context_offset, &req_ptr);
		if (mpi_errno) MPIR_ERR_POP(mpi_errno);
		all_reqs[all_reqs_idx++] = req_ptr->handle;
	}

	mpi_errno = MPIR_Waitall(all_reqs_idx, all_reqs, MPI_STATUS_IGNORE);
	if (mpi_errno) MPIR_ERR_POP(mpi_errno);
	all_reqs_idx = 0; /* set index back to zero for future use */

#ifdef DEBUG
	print_vect(comm_ptr->rank, outdegree, outnbrs_indegree, "My outnbrs_indegree is");
#endif
	/** Done with getting the indegree of my outgoing neighbors **/

	/* Going to get the incoming neighbors of my outgoing neighbors.
	 * I should send the list of my incoming neighbors to each of my
	 * incoming neighbors. Plus, I should receive a list (of incoming
	 * neighbors) from each of my outgoing neighbors.
	 */
	/* Build recv buffers first, which will become the result matrix */
	Common_nbrhood_matrix *cmn_nbh_mat;
	MPIR_CHKPMEM_MALLOC(cmn_nbh_mat, Common_nbrhood_matrix*, sizeof(Common_nbrhood_matrix),
                        mpi_errno, "cmn_nbh_mat", MPL_MEM_OTHER);
	MPIR_CHKPMEM_CALLOC(cmn_nbh_mat->ignore_row, int*, outdegree * sizeof(int),
                        mpi_errno, "cmn_nbh_mat->ignore_row", MPL_MEM_OTHER);
	MPIR_CHKPMEM_CALLOC(cmn_nbh_mat->is_row_offloaded, int*, outdegree * sizeof(int),
                        mpi_errno, "cmn_nbh_mat->is_row_offloaded", MPL_MEM_OTHER);
	MPIR_CHKPMEM_MALLOC(cmn_nbh_mat->my_innbrs_bitmap, int*, indegree * sizeof(int),
                        mpi_errno, "cmn_nbh_mat->my_innbrs_bitmap", MPL_MEM_OTHER);
	MPIR_CHKPMEM_MALLOC(cmn_nbh_mat->outnbrs_innbrs_bitmap, int**, outdegree * sizeof(int*),
                        mpi_errno, "cmn_nbh_mat->outnbrs_innbrs_bitmap", MPL_MEM_OTHER);
	MPIR_CHKPMEM_MALLOC(cmn_nbh_mat->matrix, int**, outdegree * sizeof(int*),
                        mpi_errno, "cmn_nbh_mat->matrix", MPL_MEM_OTHER);
	MPIR_CHKPMEM_MALLOC(cmn_nbh_mat->comb_matrix, Comb_element**, outdegree * sizeof(Comb_element*),
                        mpi_errno, "cmn_nbh_mat->comb_matrix", MPL_MEM_OTHER);
	MPIR_CHKPMEM_CALLOC(cmn_nbh_mat->comb_matrix_num_entries_in_row, int*, outdegree * sizeof(int),
                        mpi_errno, "cmn_nbh_mat->comb_matrix_num_entries_in_row", MPL_MEM_OTHER);
	cmn_nbh_mat->num_rows = outdegree;
    cmn_nbh_mat->indegree = indegree;
    cmn_nbh_mat->num_elements = 0;
    cmn_nbh_mat->t = 0; /* Keeping track of pairing steps order */
    cmn_nbh_mat->row_sizes = outnbrs_indegree;

	for(in_idx = 0; in_idx < indegree; in_idx++)
	{
		mpi_errno = MPID_Isend(topo_ptr->topo.dist_graph.in, indegree, MPI_INT,
	                           topo_ptr->topo.dist_graph.in[in_idx],
	                           2000, comm_ptr, context_offset, &req_ptr);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
        all_reqs[all_reqs_idx++] = req_ptr->handle;
		cmn_nbh_mat->my_innbrs_bitmap[in_idx] = 1; /* Set all incoming neighbors to active */
	}
	for(out_idx = 0; out_idx < outdegree; out_idx++)
	{
		cmn_nbh_mat->comb_matrix[out_idx] = MPL_malloc(MAX_COMB_DEGREE * sizeof(Comb_element),
                                                       MPL_MEM_OTHER);
	    for(j = 0; j < MAX_COMB_DEGREE; j++)
	    {
	        cmn_nbh_mat->comb_matrix[out_idx][j].opt = IDLE;
	        cmn_nbh_mat->comb_matrix[out_idx][j].paired_frnd = -1;
	    }
	    cmn_nbh_mat->outnbrs_innbrs_bitmap[out_idx] = MPL_malloc(outnbrs_indegree[out_idx] *
                                                                 sizeof(int), MPL_MEM_OTHER);
	    for(j = 0; j < outnbrs_indegree[out_idx]; j++)
	    {
            /* Set all incoming neighbors of all outgoing neighbors to active */
	        cmn_nbh_mat->outnbrs_innbrs_bitmap[out_idx][j] = 1;
	    }

	    cmn_nbh_mat->matrix[out_idx] = MPL_malloc(outnbrs_indegree[out_idx] * sizeof(int),
                                                  MPL_MEM_OTHER);
		mpi_errno = MPID_Irecv(cmn_nbh_mat->matrix[out_idx], outnbrs_indegree[out_idx],
	                           MPI_INT, topo_ptr->topo.dist_graph.out[out_idx],
                               2000, comm_ptr, context_offset, &req_ptr);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
        all_reqs[all_reqs_idx++] = req_ptr->handle;
		cmn_nbh_mat->num_elements += cmn_nbh_mat->row_sizes[out_idx];
	}
	for(i = 0; i < MAX_COMB_DEGREE; i++)
	{
	    cmn_nbh_mat->num_onloaded[i] = 0;
	    cmn_nbh_mat->num_offloaded[i] = 0;
		cmn_nbh_mat->sorted_cmn_nbrs[i].cmn_nbrs = NULL;
		cmn_nbh_mat->sorted_cmn_nbrs[i].num_cmn_nbrs = 0;
	}

	mpi_errno = MPIR_Waitall(all_reqs_idx, all_reqs, MPI_STATUS_IGNORE);
	if (mpi_errno) MPIR_ERR_POP(mpi_errno);
	all_reqs_idx = 0; /* set index back to zero for future use */

	*cmn_nbh_mat_ptr = cmn_nbh_mat;

	MPIR_CHKPMEM_COMMIT();

    fn_exit:
        MPL_free(all_reqs);
        return mpi_errno;
    fn_fail:
        MPIR_CHKPMEM_REAP();
        goto fn_exit;
}

int find_willing_to_pair_idx(int *frnds_pot_peers, int num_frnds, int self_rank, int start_idx)
{
    int f1_idx = -1;
    int i;
    for(i = start_idx; i < num_frnds; i++)
    {
        if(frnds_pot_peers[i] == self_rank)
        {
            f1_idx = i;
            break;
        }
    }
    return f1_idx;
}

int compare_cmn_nbr(const void * a, const void * b)
{
    if(((Common_neighbor*)a)->rank <  ((Common_neighbor*)b)->rank) return -1;
    if(((Common_neighbor*)a)->rank ==  ((Common_neighbor*)b)->rank) return 0;
    if(((Common_neighbor*)a)->rank >  ((Common_neighbor*)b)->rank) return 1;
}

Common_neighbor* find_cmn_nbrs(Common_nbrhood_matrix *cmn_nbh_mat,
                               int paired_frnd, int num_cmn_nbrs, int *dests)
{
    int i, j, k;
    k = 0;
    Common_neighbor *cmn_nbrs = MPL_malloc(num_cmn_nbrs * sizeof(Common_neighbor), MPL_MEM_OTHER);
    for(i = 0; i < cmn_nbh_mat->num_rows; i++)
    {
        if(cmn_nbh_mat->ignore_row[i])
            continue;
        for(j = 0; j < cmn_nbh_mat->row_sizes[i]; j++)
        {
            if(cmn_nbh_mat->matrix[i][j] == paired_frnd) /* It is a common neighbor */
            {
                cmn_nbrs[k].index = i;
                cmn_nbrs[k].rank = dests[i];
                k++;
            }
        }
    }
    return cmn_nbrs;
}

int mask_frnd_in_nbh_matrix(Common_nbrhood_matrix *cmn_nbh_mat, int friend)
{
    int i, j;
    for(i = 0; i < cmn_nbh_mat->num_rows; i++)
    {
        if(cmn_nbh_mat->ignore_row[i])
            continue;
        for(j = 0; j < cmn_nbh_mat->row_sizes[i]; j++)
        {
            if(cmn_nbh_mat->matrix[i][j] == friend)
                cmn_nbh_mat->matrix[i][j] = -1;
        }
    }
    return 0;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Update_common_nbrhood_mat
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Update_common_nbrhood_mat(Common_nbrhood_matrix* cmn_nbh_mat,
                                   MPIR_Comm *comm_ptr, int num_frnds)
{
    /*
     * This function updates the common neighborhood matrix
     * entries based on the remaining active incoming neighbors
     * of each outgoing neighbor. More specifically, it set
     * the ijth element to -1 if the ijth element in the updated
     * outnbrs_innbrs_bitmap matrix is 0.
     */

    int mpi_errno = MPI_SUCCESS;
    int indegree, outdegree, comm_size;
    int i, j, out_idx, in_idx, all_reqs_idx, reqs_max_size, context_offset;
    comm_size = comm_ptr->local_size;
    MPIR_Topology *topo_ptr = NULL;
    topo_ptr = MPIR_Topology_get(comm_ptr);
    if(topo_ptr == NULL)
    {
        fprintf(stderr, "ERROR: Communicator topology pointer is NULL!\n");
        return -1;
    }
    indegree = topo_ptr->topo.dist_graph.indegree;
    outdegree = topo_ptr->topo.dist_graph.outdegree;

    /* Getting the incoming neighbors bitmap of my outgoing
     * neighbors. To this end, everybody should send its own
     * innbrs bitmap to each of its incoming neighbors, and
     * also receive the bitmap from each of its outgoing neighbors.
    */
    context_offset = (comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM) ?
                      MPIR_CONTEXT_INTRA_COLL : MPIR_CONTEXT_INTER_COLL;
    reqs_max_size = indegree + outdegree;
    all_reqs_idx = 0;
    MPI_Request *all_reqs = MPL_malloc(reqs_max_size * sizeof(MPI_Request), MPL_MEM_OBJECT);
    MPIR_Request *req_ptr = NULL;

    /* Sending my own innbrs bitmap to each not-ignored incoming neighbor */
    for(in_idx = 0; in_idx < indegree; in_idx++)
    {
        if(!cmn_nbh_mat->my_innbrs_bitmap[in_idx])
            continue;
        mpi_errno = MPID_Isend(cmn_nbh_mat->my_innbrs_bitmap, indegree, MPI_INT,
                               topo_ptr->topo.dist_graph.in[in_idx], 1100,
                               comm_ptr, context_offset, &req_ptr);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
        all_reqs[all_reqs_idx++] = req_ptr->handle;
    }
    /* Receiving the innbrs bitmap of each of my not-ignored outgoing neighbors */
    for(out_idx = 0; out_idx < outdegree && num_frnds != 0; out_idx++)
    {
        if(cmn_nbh_mat->ignore_row[out_idx])
            continue;
        mpi_errno = MPID_Irecv(cmn_nbh_mat->outnbrs_innbrs_bitmap[out_idx],
                               cmn_nbh_mat->row_sizes[out_idx], MPI_INT,
                               topo_ptr->topo.dist_graph.out[out_idx],
                               1100, comm_ptr, context_offset, &req_ptr);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
        all_reqs[all_reqs_idx++] = req_ptr->handle;
    }
    mpi_errno = MPIR_Waitall(all_reqs_idx, all_reqs, MPI_STATUS_IGNORE);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    all_reqs_idx = 0; /* set index back to zero for future use */
    /** Done with getting the innbrs bitmaps of my outgoing neighbors **/

    /* Update the cmn_nbh_mat based on the updated bitmap matrix */
    for(i = 0; i < outdegree; i++)
    {
        if(cmn_nbh_mat->ignore_row[i])
            continue;
        for(j = 0; j < cmn_nbh_mat->row_sizes[i]; j++)
        {
            if(!cmn_nbh_mat->outnbrs_innbrs_bitmap[i][j])
            {
                cmn_nbh_mat->matrix[i][j] = -1;
            }
        }
    }

    fn_exit:
        MPL_free(all_reqs);
        return mpi_errno;
    fn_fail:
        goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIR_Build_nbh_coll_patt
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_Build_nbh_coll_patt(MPIR_Comm *comm_ptr)
{
	/* This is the entry point to the algorithm that
	 * will build the message-combining pattern. It
	 * is called in MPI_Dist_graph_create_adjacent().
	 */

    int mpi_errno = MPI_SUCCESS;
    int i, j, all_reqs_idx, reqs_max_size, context_offset;
    int self_rank = comm_ptr->rank;
    int comm_size = comm_ptr->local_size;

    MPIR_Topology *topo_ptr = NULL;
    topo_ptr = MPIR_Topology_get(comm_ptr);
    if(topo_ptr == NULL)
    {
        fprintf(stderr, "ERROR: Communicator topology pointer is NULL!\n");
        goto fn_fail;
    }
    int indegree = topo_ptr->topo.dist_graph.indegree;
    int outdegree = topo_ptr->topo.dist_graph.outdegree;
    int *srcs = topo_ptr->topo.dist_graph.in;
    int *dests = topo_ptr->topo.dist_graph.out;

    MPIR_CHKPMEM_DECL(2);

    context_offset = (comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM) ?
                      MPIR_CONTEXT_INTRA_COLL : MPIR_CONTEXT_INTER_COLL;
    reqs_max_size = indegree + outdegree;
    all_reqs_idx = 0;
    MPI_Request *all_reqs = MPL_malloc(reqs_max_size * sizeof(MPI_Request), MPL_MEM_OBJECT);
    MPIR_Request *req_ptr = NULL;

    /* Extract common neighborhood matrix */
    Common_nbrhood_matrix *cmn_nbh_mat;
    mpi_errno = MPIR_Get_inNbrs_of_outNbrs(comm_ptr, &cmn_nbh_mat);
    if(mpi_errno) MPIR_ERR_POP(mpi_errno);


	/***** Start of the main loop of the algorithm *****/
    int num_frnds = 0;
    int loop_count = 0;
    do
    {
#ifdef DEBUG
        char content[256];
        sprintf(content, "---------- Main loop iteration %d ----------", loop_count);
        print_in_file(self_rank, content);
#endif
        if(self_rank == 0)
            printf("\n---------- Main loop iteration %d ----------\n", loop_count);

        /*** MAJOR STEP: Find a friend to pair with ***/
        int paired_frnd, num_cmn_nbrs;
        paired_frnd = MPIR_pair_frnd(comm_ptr, cmn_nbh_mat, &num_cmn_nbrs, &num_frnds);

        if(self_rank == 0)
            printf("Rank %d: number of friends = %d", self_rank, num_frnds);

#ifdef DEBUG
        sprintf(content, "number of friends = %d", num_frnds);
        print_in_file(self_rank, content);
#endif

        int ON = 1;
        int OFF = 0;
        if(paired_frnd != -1)
        {
            /* Divide common neighbors between the paired friends.
             * Dividing is essentially turning offloaded neighbors
             * to the "off" state in the neighborhood matrix.
             *   1- Find the common neighbors' ranks.
             *   2- Sort the common neighbors based on their ranks.
             *   3- Assign the first half to the peer with lower rank,
             *      and the other half to the peer with higher rank.
             */
            Common_neighbor *cmn_nbrs = find_cmn_nbrs(cmn_nbh_mat, paired_frnd,
                                                      num_cmn_nbrs, dests);
            qsort(cmn_nbrs, num_cmn_nbrs, sizeof(Common_neighbor), compare_cmn_nbr);
            int start_keep_idx, end_keep_idx;
            int start_off_idx, end_off_idx;

            if(self_rank < paired_frnd)
            {
                /* Keep the first half of the sorted common neighbors */
                start_keep_idx = 0;
                end_keep_idx = (int) ceil(num_cmn_nbrs / 2.0) - 1;
                /* Offload the other half */
                start_off_idx = end_keep_idx + 1;
                end_off_idx = num_cmn_nbrs - 1;
            }
            else
            {
                /* Keep the second half of the sorted common neighbors */
                start_keep_idx = (int) ceil(num_cmn_nbrs / 2.0);
                end_keep_idx = num_cmn_nbrs - 1;
                /* Offload the other half */
                start_off_idx = 0;
                end_off_idx = start_keep_idx - 1;
            }

            /* Do the actual ignoring/offloading */
            for(i = start_off_idx; i <= end_off_idx; i++)
            {
                /* Marking the offloaded neighbors and also
                 * ignoring them for future phases.
                 */
                cmn_nbh_mat->is_row_offloaded[cmn_nbrs[i].index] = 1;
                cmn_nbh_mat->ignore_row[cmn_nbrs[i].index] = 1;
                add_frnd_to_comb_matrix(cmn_nbh_mat, cmn_nbrs[i].index, paired_frnd, SEND);
                mpi_errno = MPID_Isend(&OFF, 1, MPI_INT, dests[cmn_nbrs[i].index],
                                       1000, comm_ptr, context_offset, &req_ptr);
                if (mpi_errno) MPIR_ERR_POP(mpi_errno);
                all_reqs[all_reqs_idx++] = req_ptr->handle;
            }
            /* No cumulative combining, so ignore the onloaded neighbors too. */
            for(i = start_keep_idx; i <= end_keep_idx; i++)
			{
				cmn_nbh_mat->ignore_row[cmn_nbrs[i].index] = 1;
				mpi_errno = MPID_Isend(&OFF, 1, MPI_INT, dests[cmn_nbrs[i].index],
									   1000, comm_ptr, context_offset, &req_ptr);
				if (mpi_errno) MPIR_ERR_POP(mpi_errno);
				all_reqs[all_reqs_idx++] = req_ptr->handle;
			}


            /* After offloading/onloading, mark the kept neighbors
             * with the rank of the friend with which it paired.
             * We need this to remember to add the current friends's
             * data to the combined message that will later on be sent
             * to each of these neighbors.
             */
            for(i = start_keep_idx; i <= end_keep_idx; i++)
                add_frnd_to_comb_matrix(cmn_nbh_mat, cmn_nbrs[i].index, paired_frnd, RECV);

            /* Mark this friend so as NOT to pair with it again */
            mask_frnd_in_nbh_matrix(cmn_nbh_mat, paired_frnd);

            cmn_nbh_mat->sorted_cmn_nbrs[cmn_nbh_mat->t].cmn_nbrs = cmn_nbrs;
            cmn_nbh_mat->sorted_cmn_nbrs[cmn_nbh_mat->t].num_cmn_nbrs = num_cmn_nbrs;
            cmn_nbh_mat->sorted_cmn_nbrs[cmn_nbh_mat->t].offload_start = start_off_idx;
            cmn_nbh_mat->sorted_cmn_nbrs[cmn_nbh_mat->t].offload_end = end_off_idx;
            cmn_nbh_mat->sorted_cmn_nbrs[cmn_nbh_mat->t].onload_start = start_keep_idx;
            cmn_nbh_mat->sorted_cmn_nbrs[cmn_nbh_mat->t].onload_end = end_keep_idx;
            cmn_nbh_mat->num_onloaded[cmn_nbh_mat->t] = end_keep_idx - start_keep_idx + 1;
            cmn_nbh_mat->num_offloaded[cmn_nbh_mat->t] = end_off_idx - start_off_idx + 1;
        }
        /* End of if found a friend to pair with */

        /* Notify all not-yet-ignored outgoing neighbors */
        for(i = 0; i < outdegree; i++)
        {
            if(cmn_nbh_mat->ignore_row[i] == 0)
            {
                if(num_frnds == 0)
                {
                    /* We send OFF because this rank will be quitting
                     * the main while loop with num_frnd == 0, and so
                     * others should not be expecting to receive any
                     * more notifications from it.
                     */
                    mpi_errno = MPID_Isend(&OFF, 1, MPI_INT, dests[i], 1000,
                                           comm_ptr, context_offset, &req_ptr);
                    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
                    all_reqs[all_reqs_idx++] = req_ptr->handle;
                }
                else
                {
                    mpi_errno = MPID_Isend(&ON, 1, MPI_INT, dests[i], 1000,
                                           comm_ptr, context_offset, &req_ptr);
                    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
                    all_reqs[all_reqs_idx++] = req_ptr->handle;
                }
            }
        }
        for(i = 0; i < indegree; i++) /* for each of my still-active incoming neighbors */
        {
            if(cmn_nbh_mat->my_innbrs_bitmap[i] == 1)
            {
                mpi_errno = MPID_Irecv(&(cmn_nbh_mat->my_innbrs_bitmap[i]), 1, MPI_INT,
                                       srcs[i], 1000, comm_ptr, context_offset, &req_ptr);
                if (mpi_errno) MPIR_ERR_POP(mpi_errno);
                all_reqs[all_reqs_idx++] = req_ptr->handle;
            }
        }
#ifdef DEBUG
        print_vect(self_rank, indegree, cmn_nbh_mat->my_innbrs_bitmap,
                   "cmn_nbh_mat->my_innbrs_bitmap:");
        print_vect(self_rank, indegree, srcs, "srcs:");
#endif

        mpi_errno = MPIR_Waitall(all_reqs_idx, all_reqs, MPI_STATUS_IGNORE);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
        all_reqs_idx = 0; /* set index back to zero for future use */

        INFO(printf("ITR %d: Rank %d done with notifying all not-yet-ignored outgoing neighbors\n",
                    loop_count, self_rank);fflush(stdout););
#ifdef DEBUG
        print_vect(self_rank, indegree, cmn_nbh_mat->my_innbrs_bitmap,
                   "UPDATED cmn_nbh_mat->my_innbrs_bitmap:");
        fflush(stdout);
        print_nbh_mat(comm_ptr->rank, cmn_nbh_mat, -3,
                      "----------------Current common neighborhood matrix state----------------");
        fflush(stdout);
#endif

        /* Update cmn_nbh_mat with respect to the new innbrs bitmaps */
        MPIR_Update_common_nbrhood_mat(cmn_nbh_mat, comm_ptr, num_frnds);

        INFO(printf("ITR %d: Rank %d done with MPIR_Update_common_nbrhood_mat\n",
                    loop_count, self_rank);fflush(stdout););

        cmn_nbh_mat->t++;
        loop_count++;
    } while(num_frnds > 0 && loop_count < MAX_LOOP_COUNT);
	/***** End of the main loop of the algorithm *****/

	if(loop_count >= MAX_LOOP_COUNT && num_frnds != 0)
		printf("Rank %d: Warning! loop_count = %d reached MAX_LOOP_COUNT, while num_frnds = %d\n",
               self_rank, loop_count, num_frnds);

    if(num_frnds == 0 && loop_count == 1)
        printf("No friend was initially found by rank %d with the threshold set to %d.\n",
		self_rank, MPIR_CVAR_NEIGHBOR_COLL_MSG_COMB_FRNDSHP_THRSHLD);

    /* Once a rank gets out of the while loop above due
     * to num_frnds == 0, it should still issue the recv
     * operations corresponding to its still-active incoming
     * neighbors because those neighbors will be sending
     * notifications to this rank until they get out of the
     * loop too.
     */
    int have_atleast_one_active_in_nbr;
    do
    {
        have_atleast_one_active_in_nbr = 0;
        for(i = 0; i < indegree; i++)
        {
            /* receive from the ith incoming neighbor */
            if(cmn_nbh_mat->my_innbrs_bitmap[i])
            {
                mpi_errno = MPID_Irecv(&(cmn_nbh_mat->my_innbrs_bitmap[i]), 1, MPI_INT,
                                       srcs[i], 1000, comm_ptr, context_offset, &req_ptr);
                if (mpi_errno) MPIR_ERR_POP(mpi_errno);
                all_reqs[all_reqs_idx++] = req_ptr->handle;
                have_atleast_one_active_in_nbr = 1;
            }
        }
        mpi_errno = MPIR_Waitall(all_reqs_idx, all_reqs, MPI_STATUS_IGNORE);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
        all_reqs_idx = 0; /* set index back to zero for future use */

        /* Just for matching send/recvs for those who are still num_frnds > 0 */
        MPIR_Update_common_nbrhood_mat(cmn_nbh_mat, comm_ptr, num_frnds);
    }while(have_atleast_one_active_in_nbr);

#ifdef DEBUG
    print_nbh_mat(comm_ptr->rank, cmn_nbh_mat, -3,
                  "-----------------Final common neighborhood matrix state-----------------");
#endif
    /**************** Done with building the message-combining pattern ****************/

	/***** Extract/communicate some information that is required for building the schedule *****/
    int **sched_msg, *sched_msg_sizes;
    sched_msg = NULL;
    sched_msg_sizes = NULL;
    sched_msg = MPL_malloc(outdegree * sizeof(int*), MPL_MEM_OTHER);
    sched_msg_sizes = MPL_malloc(outdegree * sizeof(int), MPL_MEM_OTHER);

    for(i = 0; i < outdegree; i++)
    {
        if(!cmn_nbh_mat->is_row_offloaded[i])
		{
            sched_msg_sizes[i] = COMB_LIST_START_IDX + 1 +
                                 cmn_nbh_mat->comb_matrix_num_entries_in_row[i];
		}
        else
		{
			/* The last element for offloaded rows represents the rank to
			 * which the neighbor is offloaded (it has 'S' as the operation).
			 * We do not want to communicate this last element.
			 */
            sched_msg_sizes[i] = COMB_LIST_START_IDX +
                                 cmn_nbh_mat->comb_matrix_num_entries_in_row[i];
		}
        sched_msg[i] = MPL_malloc(sched_msg_sizes[i] * sizeof(int), MPL_MEM_OTHER);

        /* The order of elements in the each sched_msg array: (Should really change this!)
         * 0: is neighbor off or not
         * 1: t
         * 2: combination list size
         * 3: combination list itself which includes the rank of sending process too
         */
        sched_msg[i][0] = cmn_nbh_mat->is_row_offloaded[i];
        if(!cmn_nbh_mat->is_row_offloaded[i])
        {
            sched_msg[i][2] = 1 + cmn_nbh_mat->comb_matrix_num_entries_in_row[i];
            sched_msg[i][1] = cmn_nbh_mat->t;
        }
        else /* Again we do not consider the last element for the offloaded rows */
        {
            sched_msg[i][2] = cmn_nbh_mat->comb_matrix_num_entries_in_row[i];
            /* The value of 't' does not not mean anything for offloaded rows,
             * but we set it to -1 to recognize offloaded neighbors later on
             * while building the schedule as we modify the ON/OFF field there.
             * I know, terrible design! but I need to get results quickly.
             */
            sched_msg[i][1] = -1;
        }

        /* We can ignore the offloaded neighbors in the following (non-cumulative) */
        sched_msg[i][COMB_LIST_START_IDX] = self_rank;
        int comb_list_idx = COMB_LIST_START_IDX + 1;
        for(j = 0; j < cmn_nbh_mat->t; j++)
        {
            if(cmn_nbh_mat->comb_matrix[i][j].opt == RECV)
            {
                sched_msg[i][comb_list_idx] = cmn_nbh_mat->comb_matrix[i][j].paired_frnd;
                comb_list_idx++;
                sched_msg[i][1] = j;
            }
            else if(cmn_nbh_mat->comb_matrix[i][j].opt == SEND)
                break;
        }
    }

#ifdef DEBUG
    print_mat(self_rank, outdegree, sched_msg_sizes, sched_msg, -3,
            "The scheduling matrix to send out:\n"
            "Off t   Sz ");
#endif

    /* Communicating the extracted scheduling information */
    int **sched_recv_buffs;
    sched_recv_buffs = MPL_malloc(indegree * sizeof(int*), MPL_MEM_OTHER);
    for(i = 0; i < indegree; i++)
    	sched_recv_buffs[i] = MPL_malloc(MAX_COMB_DEGREE * sizeof(int), MPL_MEM_OTHER);
    for(i = 0; i < outdegree; i++)
    {
        mpi_errno = MPID_Isend(sched_msg[i], sched_msg_sizes[i], MPI_INT,
                               dests[i], 2000, comm_ptr, context_offset, &req_ptr);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
        all_reqs[all_reqs_idx++] = req_ptr->handle;
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    }
    for(i = 0; i < indegree; i++)
    {
        mpi_errno = MPID_Irecv(sched_recv_buffs[i], MAX_COMB_DEGREE, MPI_INT,
                               srcs[i], 2000, comm_ptr, context_offset, &req_ptr);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
        all_reqs[all_reqs_idx++] = req_ptr->handle;
    }
    mpi_errno = MPIR_Waitall(all_reqs_idx, all_reqs, MPI_STATUS_IGNORE);
    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
    all_reqs_idx = 0; /* set index back to zero for future use */

#ifdef DEBUG
    int *sched_recv_buff_sizes; /* We're using this for the print_mat matrix only */
    sched_recv_buff_sizes = MPL_malloc(indegree * sizeof(int), MPL_MEM_OTHER);
    for(i = 0; i < indegree; i++)
        sched_recv_buff_sizes[i] = COMB_LIST_START_IDX + sched_recv_buffs[i][2];
    print_mat(self_rank, indegree, sched_recv_buff_sizes, sched_recv_buffs, -3,
              "The scheduling matrix received from incoming neighbors:\n"
              "Off t   Sz ");
    MPL_free(sched_recv_buff_sizes);
#endif

    /* Attaching the received sched_recv_buff and cmn_nbh_mat to the topology of communicator */
    MPIR_CHKPMEM_MALLOC(topo_ptr->topo.dist_graph.nbh_coll_patt,
                        nbh_coll_patt*, sizeof(nbh_coll_patt), mpi_errno,
                        "topo_ptr->topo.dist_graph.nbh_coll_patt", MPL_MEM_OTHER);
    topo_ptr->topo.dist_graph.nbh_coll_patt->cmn_nbh_mat = cmn_nbh_mat;
    topo_ptr->topo.dist_graph.nbh_coll_patt->incom_sched_mat = sched_recv_buffs;

    MPIR_CHKPMEM_COMMIT();

fn_exit:
    if(sched_msg)
    {
        for(i = 0; i < outdegree; i++)
        {
            MPL_free(sched_msg[i]);
        }
    }
    MPL_free(sched_msg);
    MPL_free(sched_msg_sizes);
    MPL_free(all_reqs);
    return mpi_errno;
fn_fail:
    MPIR_CHKPMEM_REAP();
    goto fn_exit;
}

#undef FUNCNAME
#define FUNCNAME MPIR_pair_frnd
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
int MPIR_pair_frnd(MPIR_Comm *comm_ptr, Common_nbrhood_matrix *cmn_nbh_mat,
                       int *num_cmn_nbrs, int *num_frnds_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    int i, j, all_reqs_idx, reqs_max_size, context_offset;
    int self_rank = comm_ptr->rank;
    int comm_size = comm_ptr->local_size;
    int nbr_frndshp_thr = MPIR_CVAR_NEIGHBOR_COLL_MSG_COMB_FRNDSHP_THRSHLD;
    int num_frnds = 0; /* Number of friends that remain active after pairing attempt */
    int my_pot_peer = -1;
    int my_pot_peer_idx = -1; /* this is the index in the frnds_pot_peers array */
    int paired = 0;
    int terminal = 0;
    int itr = 0;
    int f1 = -1;
    int f1_idx = -1;

    MPIR_CHKLMEM_DECL(1);

#ifdef DEBUG
    print_nbh_mat(comm_ptr->rank, cmn_nbh_mat, -3, "My outnbrs_innbrs are");
#endif

    /* Build the global friendship array that counts
     * the number of neighbors that I have in common
     * with each other rank in the communicator. This
     * can be optimized later to use a better data structure.
     */
    int *glob_frndshp_arr;
    glob_frndshp_arr = MPL_malloc(comm_size * sizeof(int), MPL_MEM_OTHER);
    for(i = 0; i < comm_size; i++)
        glob_frndshp_arr[i] = 0;
    for(i = 0; i < cmn_nbh_mat->num_rows; i++)
    {
        if(cmn_nbh_mat->ignore_row[i])
            continue; /* Ignore disabled rows (from previous pairing phases) */

    	for(j = 0; j < cmn_nbh_mat->row_sizes[i]; j++)
    	{
    	    if(cmn_nbh_mat->matrix[i][j] == -1 || cmn_nbh_mat->matrix[i][j] == self_rank)
                continue; /* Ignore self rank and disabled friends */
    		glob_frndshp_arr[cmn_nbh_mat->matrix[i][j]]++;
    		if(glob_frndshp_arr[cmn_nbh_mat->matrix[i][j]] == nbr_frndshp_thr)
                num_frnds++;
    	}
    }
    *num_frnds_ptr = num_frnds;

    if(num_frnds == 0)
    {
        INFO2(printf("Rank %d failed to pair with anyone; num_frnds is 0!\n", self_rank););
        MPL_free(glob_frndshp_arr);
        *num_cmn_nbrs = 0;
        return -1;
    }

    context_offset = (comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM) ?
                      MPIR_CONTEXT_INTRA_COLL : MPIR_CONTEXT_INTER_COLL;
    reqs_max_size = 2 * num_frnds;
    all_reqs_idx = 0;
    MPI_Request *all_reqs = MPL_malloc(reqs_max_size * sizeof(MPI_Request), MPL_MEM_OTHER);
    MPIR_Request *req_ptr = NULL;

    /* Build the maxHeap of friends from the global friendship array */
    heap *frndshp_maxHeap;
    MPIR_CHKLMEM_MALLOC(frndshp_maxHeap, heap*, sizeof(heap),
                        mpi_errno, "frndshp_maxHeap", MPL_MEM_OTHER);
    heap_init(frndshp_maxHeap, cmn_nbh_mat->num_elements);
    for(i = 0; i < comm_size; i++)
    {
        if(glob_frndshp_arr[i] < nbr_frndshp_thr || i == self_rank)
            continue;
        heap_element *e = MPL_malloc(sizeof(heap_element), MPL_MEM_OTHER);
        e->key = glob_frndshp_arr[i]; /* key is the number of common friends */
        e->value = i; /* value is the rank */
        e->paired = 0;
        heap_insert(frndshp_maxHeap, e);
    }

    int *frnds_ranks, *frnds_pot_peers, *frnds_paired;
    frnds_ranks = MPL_malloc(num_frnds * sizeof(int), MPL_MEM_OTHER);
	frnds_pot_peers = MPL_malloc(num_frnds * sizeof(int), MPL_MEM_OTHER);
	frnds_paired = MPL_malloc(num_frnds * sizeof(int), MPL_MEM_OTHER);
	for(i = 0; i < num_frnds; i++)
    {
    	frnds_ranks[i] = frndshp_maxHeap->heap_arr[i+1]->value;
        frnds_pot_peers[i] = -1;
    	frnds_paired[i] = 0;
    }

    /* Pairing friends in an iterative fashion */
	while(!paired && !terminal)
	{
	    /* REMEMBER: my_pot_peer_idx represents the index in frnds_ranks,
		 * frnds_pot_peers, and frnds_paired arrays; it is not necessarily
		 * in correspondence with the index of the elements in the heap array.
		 * At the beginning of each iteration and before updating, my_pot_peer_idx
		 * is only valid for the frnds_paired array because the other two arrays
		 * have been updated by removing the paired friends from them.
		 */
#ifdef DEBUG
	    char content[256];
        sprintf(content, "------Iteration %d started------", itr);
	    print_in_file(self_rank, content);
#endif

	    if(itr == 0)
	    {
	        if(!heap_is_empty(frndshp_maxHeap))
	        {
	            my_pot_peer = heap_peek_max_value(frndshp_maxHeap);
	            my_pot_peer_idx = find_in_arr(frnds_ranks, num_frnds, my_pot_peer);
	        }
	        else
            {
                fprintf(stderr, "ERROR: Rank %d found the heap initially empty!\n", self_rank);
                my_pot_peer = my_pot_peer_idx = -1;
                terminal = 1;
            }
	    }
	    else
	    {
	        /* We consider two cases:
			 * a) my_pot_peer is still among my friends AND
	         *    my potential peer has failed too AND
			 *    self_rank < my_pot_peer.
             * b) my potential peer just gave up OR
			 *    my potential peer has NOT failed (already paired) OR
			 *    my potential peer has failed BUT self_rank >= my_pot_peer.
             */

	        /* update my_pot_peer_idx with respect to the updated frnds_ranks arrays */
	        my_pot_peer_idx = find_in_arr(frnds_ranks, num_frnds, my_pot_peer);

	        if(my_pot_peer_idx != -1		  &&
			   !frnds_paired[my_pot_peer_idx] &&
			   self_rank < my_pot_peer )
            {
				/* Case (a). Just stick to the previously chosen friend; do nothing */
            }
            else
            {
                /* Case b: Look for another potential peer */
                /* Look for the first friend that has previously
                 * chosen me. Let us call this friend f1.
                 */
                f1_idx = find_willing_to_pair_idx(frnds_pot_peers, num_frnds, self_rank, 0);
				/* We need an f1 whose rank is less than self_rank */
                while(f1_idx != -1 && frnds_ranks[f1_idx] > self_rank)
                    f1_idx = find_willing_to_pair_idx(frnds_pot_peers, num_frnds,
                                                      self_rank, f1_idx+1);
                if(f1_idx != -1) /* Found a valid f1 */
                {
                    my_pot_peer = frnds_ranks[f1_idx];
                    my_pot_peer_idx = f1_idx;
                }
                else /* Did not find a valid f1; look for someone else */
                {
                    /* Throw away the previously chosen friend. But how do
                     * we know if the previous chosen friend would also
                     * throw me away or not? This could lead to a deadlock
					 * when the friends communicate with each other. To solve
					 * this issue, we keep track of the friends with whom we
                     * should communicate in the frnds_ranks array, and do
                     * not use the heap array for this purpose. NOTE: The
                     * removal below is why the heap array could be different
                     * from frnds_ranks and other related arrays; we are removing
					 * a friend that is NOT paired.
                     */
                    int idx = heap_find_value(frndshp_maxHeap, my_pot_peer);
                    if(idx != -1)
                        heap_remove_index(frndshp_maxHeap, idx);
                    if(heap_is_empty(frndshp_maxHeap))
                    {
                        INFO2(printf("Rank %d failed to pair with anyone;"
                                     "heap is empty (iteration %d)\n", self_rank, itr););
                        my_pot_peer = my_pot_peer_idx = -1;
                        terminal = 1;
                        /* We should not break when someone becomes terminal;
                         * we should let it go to the end of the current
                         * iteration to contribute in the communications and
                         * avoid deadlock. Setting terminal to 1 would cause
                         * the terminal process to exit from the loop in the
                         * next iteration.
                         */
                    }
                    else
                    {
                        my_pot_peer = heap_peek_max_value(frndshp_maxHeap);
                        my_pot_peer_idx = find_in_arr(frnds_ranks, num_frnds, my_pot_peer);
                    }
                }
            }
	    }

		/* First round of communications to let the friends
         * inform each other about their chosen potential peers.
         */
        for(i = 0; i < num_frnds; i++)
        {
		    mpi_errno = MPID_Isend(&my_pot_peer, 1, MPI_INT, frnds_ranks[i],
		                           1000, comm_ptr, context_offset, &req_ptr);
		    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
		    all_reqs[all_reqs_idx++] = req_ptr->handle;

		    mpi_errno = MPID_Irecv(&frnds_pot_peers[i], 1, MPI_INT, frnds_ranks[i],
		                           1000, comm_ptr, context_offset, &req_ptr);
		    if (mpi_errno) MPIR_ERR_POP(mpi_errno);
		    all_reqs[all_reqs_idx++] = req_ptr->handle;
        }
#ifdef DEBUG
		print_vect(self_rank, num_frnds, frnds_ranks, "frnds_ranks:");
#endif
		mpi_errno = MPIR_Waitall(all_reqs_idx, all_reqs, MPI_STATUS_IGNORE);
		if (mpi_errno) MPIR_ERR_POP(mpi_errno);
		all_reqs_idx = 0; /* set index back to zero for future use */

#ifdef DEBUG
        print_vect(self_rank, num_frnds, frnds_pot_peers, "Updated frnds_pot_peers:");
#endif
        INFO(printf("Rank %d passed the first comm round; itr = %d\n", self_rank, itr));
#ifdef DEBUG
        sprintf(content, "MY_POT_PEER =  %d", my_pot_peer);
        print_in_file(self_rank, content);
#endif

        /* Check to see whether I could mutually pair with my potential peer */
        if(my_pot_peer_idx != -1 && frnds_pot_peers[my_pot_peer_idx] == self_rank)
        {
            paired = 1;
            if(self_rank == 0)
            {
                printf("Rank %d: paired successfully with %d "
                        "in iteration %d with %d common neighbors.\n",
                    self_rank, my_pot_peer, itr, glob_frndshp_arr[my_pot_peer]);
            }
        }

        /* Second round of communications with my friends to let them
		 * know whether I could mutually pair with my potential peer.
		 */
        for(i = 0; i < num_frnds; i++)
        {
            mpi_errno = MPID_Isend(&paired, 1, MPI_INT, frnds_ranks[i],
                                   2000, comm_ptr, context_offset, &req_ptr);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);
            all_reqs[all_reqs_idx++] = req_ptr->handle;

            mpi_errno = MPID_Irecv(&frnds_paired[i], 1, MPI_INT, frnds_ranks[i],
                                   2000, comm_ptr, context_offset, &req_ptr);
            if (mpi_errno) MPIR_ERR_POP(mpi_errno);
            all_reqs[all_reqs_idx++] = req_ptr->handle;
        }
        mpi_errno = MPIR_Waitall(all_reqs_idx, all_reqs, MPI_STATUS_IGNORE);
        if (mpi_errno) MPIR_ERR_POP(mpi_errno);
        all_reqs_idx = 0; /* set index back to zero for future use */

#ifdef DEBUG
        print_vect(self_rank, num_frnds, frnds_paired, "frnds_paired:");
#endif

        INFO(printf("Rank %d passed the second comm round; itr = %d\n", self_rank, itr));

#ifdef DEBUG
        int *vals = MPL_malloc(frndshp_maxHeap->count * sizeof(int), MPL_MEM_OTHER);
        if(!heap_get_values_array(frndshp_maxHeap, vals))
            print_vect(self_rank, frndshp_maxHeap->count, vals,
                       "heap array values (ranks) before removing paired:");
        MPL_free(vals);
#endif

        /* Remove from the heap those friends that
		 * have already been paired or terminated.
		 */
        for(i = 0; i < num_frnds; i++)
        {
            if(frnds_paired[i] || frnds_pot_peers[i] == -1)
            {
                int idx = heap_find_value(frndshp_maxHeap, frnds_ranks[i]);
                if(idx != -1)
                    heap_remove_index(frndshp_maxHeap, idx);
            }
        }

#ifdef DEBUG
        vals = MPL_malloc(frndshp_maxHeap->count * sizeof(int), MPL_MEM_OTHER);
        if(!heap_get_values_array(frndshp_maxHeap, vals))
            print_vect(self_rank, frndshp_maxHeap->count, vals,
                       "heap array values (ranks) after removing paired:");
        MPL_free(vals);
#endif

        /* Update the frnd_ranks and frnds_pot_peers arrays so
		 * to include only non-paired and non-terminal friends.
		 */
        int new_num_frnds = 0;
        for(i = 0; i < num_frnds; i++)
        {
            if(!frnds_paired[i]  && frnds_pot_peers[i] != -1)
            {
                frnds_ranks[new_num_frnds] = frnds_ranks[i];
                frnds_pot_peers[new_num_frnds] = frnds_pot_peers[i];
                new_num_frnds++;
            }
        }
        num_frnds = new_num_frnds;
        itr++;
	}
    /* End of the main pairing loop */

fn_exit:
    *num_cmn_nbrs = (my_pot_peer == -1) ? 0 : glob_frndshp_arr[my_pot_peer];
    heap_free_array(frndshp_maxHeap);
    MPL_free(glob_frndshp_arr);
    MPL_free(frnds_ranks);
    MPL_free(frnds_pot_peers);
    MPL_free(frnds_paired);
    MPL_free(all_reqs);
    MPIR_CHKLMEM_FREEALL();
    return my_pot_peer;
fn_fail:
    fprintf(stderr, "Rank %d failed in MPIR_pair_frnd!\n", self_rank);
    *num_cmn_nbrs = 0;
    MPIR_CHKLMEM_FREEALL();
    return -1;
}

/* -- Begin Profiling Symbol Block for routine MPI_Dist_graph_create_adjacent */
#if defined(HAVE_PRAGMA_WEAK)
#pragma weak MPI_Dist_graph_create_adjacent = PMPI_Dist_graph_create_adjacent
#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#pragma _HP_SECONDARY_DEF PMPI_Dist_graph_create_adjacent  MPI_Dist_graph_create_adjacent
#elif defined(HAVE_PRAGMA_CRI_DUP)
#pragma _CRI duplicate MPI_Dist_graph_create_adjacent as PMPI_Dist_graph_create_adjacent
#elif defined(HAVE_WEAK_ATTRIBUTE)
int MPI_Dist_graph_create_adjacent(MPI_Comm comm_old, int indegree, const int sources[],
                                   const int sourceweights[], int outdegree,
                                   const int destinations[], const int destweights[],
                                   MPI_Info info, int reorder, MPI_Comm * comm_dist_graph)
    __attribute__ ((weak, alias("PMPI_Dist_graph_create_adjacent")));
#endif
/* -- End Profiling Symbol Block */

/* Define MPICH_MPI_FROM_PMPI if weak symbols are not supported to build
   the MPI routines */
#ifndef MPICH_MPI_FROM_PMPI
#undef MPI_Dist_graph_create_adjacent
#define MPI_Dist_graph_create_adjacent PMPI_Dist_graph_create_adjacent
/* any utility functions should go here, usually prefixed with PMPI_LOCAL to
 * correctly handle weak symbols and the profiling interface */
#endif

#undef FUNCNAME
#define FUNCNAME MPI_Dist_graph_create_adjacent
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
/*@
MPI_Dist_graph_create_adjacent - returns a handle to a new communicator to
which the distributed graph topology information is attached.

Input Parameters:
+ comm_old - input communicator (handle)
. indegree - size of sources and sourceweights arrays (non-negative integer)
. sources - ranks of processes for which the calling process is a
            destination (array of non-negative integers)
. sourceweights - weights of the edges into the calling
                  process (array of non-negative integers or MPI_UNWEIGHTED)
. outdegree - size of destinations and destweights arrays (non-negative integer)
. destinations - ranks of processes for which the calling process is a
                 source (array of non-negative integers)
. destweights - weights of the edges out of the calling process
                (array of non-negative integers or MPI_UNWEIGHTED)
. info - hints on optimization and interpretation of weights (handle)
- reorder - the ranks may be reordered (true) or not (false) (logical)

Output Parameters:
. comm_dist_graph - communicator with distributed graph topology (handle)

.N ThreadSafe

.N Fortran

.N Errors
.N MPI_SUCCESS
.N MPI_ERR_ARG
.N MPI_ERR_OTHER
@*/
int MPI_Dist_graph_create_adjacent(MPI_Comm comm_old,
                                   int indegree, const int sources[],
                                   const int sourceweights[],
                                   int outdegree, const int destinations[],
                                   const int destweights[],
                                   MPI_Info info, int reorder, MPI_Comm * comm_dist_graph)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_Comm *comm_ptr = NULL;
    MPIR_Comm *comm_dist_graph_ptr = NULL;
    MPIR_Topology *topo_ptr = NULL;
    MPII_Dist_graph_topology *dist_graph_ptr = NULL;
    MPIR_CHKPMEM_DECL(5);
    MPIR_FUNC_TERSE_STATE_DECL(MPID_STATE_MPI_DIST_GRAPH_CREATE_ADJACENT);

    MPIR_ERRTEST_INITIALIZED_ORDIE();

    MPID_THREAD_CS_ENTER(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    MPIR_FUNC_TERSE_ENTER(MPID_STATE_MPI_DIST_GRAPH_CREATE_ADJACENT);

    /* Validate parameters, especially handles needing to be converted */
#ifdef HAVE_ERROR_CHECKING
    {
        MPID_BEGIN_ERROR_CHECKS;
        {
            MPIR_ERRTEST_COMM(comm_old, mpi_errno);
            MPIR_ERRTEST_INFO_OR_NULL(info, mpi_errno);
        }
        MPID_END_ERROR_CHECKS;
    }
#endif

    /* Convert MPI object handles to object pointers */
    MPIR_Comm_get_ptr(comm_old, comm_ptr);

    /* Validate parameters and objects (post conversion) */
#ifdef HAVE_ERROR_CHECKING
    {
        MPID_BEGIN_ERROR_CHECKS;
        {
            /* Validate comm_ptr */
            MPIR_Comm_valid_ptr(comm_ptr, mpi_errno, FALSE);
            if (mpi_errno != MPI_SUCCESS)
                goto fn_fail;
            /* If comm_ptr is not valid, it will be reset to null */
            if (comm_ptr) {
                MPIR_ERRTEST_COMM_INTRA(comm_ptr, mpi_errno);
            }

            MPIR_ERRTEST_ARGNEG(indegree, "indegree", mpi_errno);
            MPIR_ERRTEST_ARGNEG(outdegree, "outdegree", mpi_errno);

            if (indegree > 0) {
                MPIR_ERRTEST_ARGNULL(sources, "sources", mpi_errno);
                if (sourceweights == MPI_UNWEIGHTED && destweights != MPI_UNWEIGHTED) {
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_TOPOLOGY, "**unweightedboth");
                    goto fn_fail;
                }
                /* TODO check ranges for array elements too (**argarrayneg / **rankarray) */
            }
            if (outdegree > 0) {
                MPIR_ERRTEST_ARGNULL(destinations, "destinations", mpi_errno);
                if (destweights == MPI_UNWEIGHTED && sourceweights != MPI_UNWEIGHTED) {
                    MPIR_ERR_SET(mpi_errno, MPI_ERR_TOPOLOGY, "**unweightedboth");
                    goto fn_fail;
                }
            }
            MPIR_ERRTEST_ARGNULL(comm_dist_graph, "comm_dist_graph", mpi_errno);
        }
        MPID_END_ERROR_CHECKS;
    }
#endif /* HAVE_ERROR_CHECKING */

    /* ... body of routine ...  */

    /* Implementation based on Torsten Hoefler's reference implementation
     * attached to MPI-2.2 ticket #33. */
    *comm_dist_graph = MPI_COMM_NULL;

    /* following the spirit of the old topo interface, attributes do not
     * propagate to the new communicator (see MPI-2.1 pp. 243 line 11) */
    mpi_errno = MPII_Comm_copy(comm_ptr, comm_ptr->local_size, &comm_dist_graph_ptr);
    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

    /* Create the topology structure */
    MPIR_CHKPMEM_MALLOC(topo_ptr, MPIR_Topology *, sizeof(MPIR_Topology), mpi_errno, "topo_ptr",
                        MPL_MEM_COMM);
    topo_ptr->kind = MPI_DIST_GRAPH;
    dist_graph_ptr = &topo_ptr->topo.dist_graph;
    dist_graph_ptr->indegree = indegree;
    dist_graph_ptr->in = NULL;
    dist_graph_ptr->in_weights = NULL;
    dist_graph_ptr->outdegree = outdegree;
    dist_graph_ptr->out = NULL;
    dist_graph_ptr->out_weights = NULL;
    dist_graph_ptr->is_weighted = (sourceweights != MPI_UNWEIGHTED);
    dist_graph_ptr->nbh_coll_patt = NULL;
    dist_graph_ptr->nbh_coll_sched = NULL;
    dist_graph_ptr->sched_mem_to_free_num_entries = 0;
    int i;
    for(i = 0; i <  SCHED_MEM_TO_FREE_MAX_SIZE; i++)
        dist_graph_ptr->sched_mem_to_free[i] = NULL;

    MPIR_CHKPMEM_MALLOC(dist_graph_ptr->in, int *, indegree * sizeof(int), mpi_errno,
                        "dist_graph_ptr->in", MPL_MEM_COMM);
    MPIR_CHKPMEM_MALLOC(dist_graph_ptr->out, int *, outdegree * sizeof(int), mpi_errno,
                        "dist_graph_ptr->out", MPL_MEM_COMM);
    MPIR_Memcpy(dist_graph_ptr->in, sources, indegree * sizeof(int));
    MPIR_Memcpy(dist_graph_ptr->out, destinations, outdegree * sizeof(int));

    if (dist_graph_ptr->is_weighted) {
        MPIR_CHKPMEM_MALLOC(dist_graph_ptr->in_weights, int *, indegree * sizeof(int), mpi_errno,
                            "dist_graph_ptr->in_weights", MPL_MEM_COMM);
        MPIR_CHKPMEM_MALLOC(dist_graph_ptr->out_weights, int *, outdegree * sizeof(int), mpi_errno,
                            "dist_graph_ptr->out_weights", MPL_MEM_COMM);
        MPIR_Memcpy(dist_graph_ptr->in_weights, sourceweights, indegree * sizeof(int));
        MPIR_Memcpy(dist_graph_ptr->out_weights, destweights, outdegree * sizeof(int));
    }

    mpi_errno = MPIR_Topology_put(comm_dist_graph_ptr, topo_ptr);
    if (mpi_errno)
        MPIR_ERR_POP(mpi_errno);

    MPIR_OBJ_PUBLISH_HANDLE(*comm_dist_graph, comm_dist_graph_ptr->handle);
    MPIR_CHKPMEM_COMMIT();

    if(MPIR_Ineighbor_allgather_intra_algo_choice == MPIR_INEIGHBOR_ALLGATHER_INTRA_ALGO_COMB)
    {
        double build_nbh_coll_patt_time = -MPI_Wtime();
        MPIR_Build_nbh_coll_patt(comm_dist_graph_ptr);
        build_nbh_coll_patt_time += MPI_Wtime();

        double max_build_nbh_coll_patt_time = 0;
		MPIR_Errflag_t errflag = MPIR_ERR_NONE;
        MPIR_Reduce_impl(&build_nbh_coll_patt_time, &max_build_nbh_coll_patt_time,
                         1, MPI_DOUBLE, MPI_MAX, 0, comm_ptr, &errflag);
        if(comm_ptr->rank == 0)
            printf("\nTime to build the neighborhood pattern (max): %lf (s)\n",
		   max_build_nbh_coll_patt_time);
    }

    /* ... end of body of routine ... */
  fn_exit:
    MPIR_FUNC_TERSE_EXIT(MPID_STATE_MPI_DIST_GRAPH_CREATE_ADJACENT);
    MPID_THREAD_CS_EXIT(GLOBAL, MPIR_THREAD_GLOBAL_ALLFUNC_MUTEX);
    return mpi_errno;

    /* --BEGIN ERROR HANDLING-- */
  fn_fail:
    MPIR_CHKPMEM_REAP();
#ifdef HAVE_ERROR_CHECKING
    mpi_errno =
        MPIR_Err_create_code(mpi_errno, MPIR_ERR_RECOVERABLE, FCNAME, __LINE__, MPI_ERR_OTHER,
                             "**mpi_dist_graph_create_adjacent",
                             "**mpi_dist_graph_create_adjacent %C %d %p %p %d %p %p %I %d %p",
                             comm_old, indegree, sources, sourceweights, outdegree, destinations,
                             destweights, info, reorder, comm_dist_graph);
#endif
    mpi_errno = MPIR_Err_return_comm(comm_ptr, FCNAME, mpi_errno);
    goto fn_exit;
    /* --END ERROR HANDLING-- */
}
