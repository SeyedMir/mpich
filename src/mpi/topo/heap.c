/*---------------------------------------------------------------------*/
/*     (C) Copyright 2017 Parallel Processing Research Laboratory      */
/*                   Queen's University at Kingston                    */
/*                Neighborhood Collective Communication                */
/*                    Seyed Hessamedin Mirsadeghi                      */
/*---------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include "mpiimpl.h"
#include "heap.h"

//We use 1-indexed array for heap implementation.
//Element at index 0 is just undefined/ignored.

#define LCHILD(x) 2 * x
#define RCHILD(x) 2 * x + 1
#define PARENT(x) x / 2

#define INFO(stmt)

int heap_init(heap *h, int arr_size)
{
	if(h == NULL)
	{
		fprintf(stderr, "ERROR: heap handle is NULL in heap_init!\n");
		return -1;
	}

	h->arr_size = arr_size + 1; //+1 because we use 1-indexed array
	h->count = 0;
	h->heap_arr = MPL_malloc(h->arr_size * sizeof(heap_element*), MPL_MEM_OTHER);

	return 0;
}

int heap_free_array(heap *h)
{
    if(h == NULL)
    {
        fprintf(stderr, "ERROR: heap handle is NULL in heap_free!\n");
        return -1;
    }

    int i;
    for(i = 1; i <= h->count; i++)
    {
        MPL_free(h->heap_arr[i]);
    }
    MPL_free(h->heap_arr);
    h->heap_arr = NULL;
    return 0;
}

int heap_is_empty(heap *h)
{
    if(h == NULL)
    {
        fprintf(stderr, "ERROR: heap handle is NULL in heap_insert!\n");
        return -1;
    }
    if(h->count == 0)
        return 1;
    else
        return 0;
}

int heap_insert(heap *h,  heap_element *e)
{
	if(h == NULL)
	{
		fprintf(stderr, "ERROR: heap handle is NULL in heap_insert!\n");
		return -1;
	}

	if(h->count + 1 == h->arr_size)
	{
		fprintf(stderr, "ERROR: Cannot insert element; heap is full!\n");
		return -1;
	}

	h->count = h->count + 1;
	h->heap_arr[h->count] = e;

	//Shifting up if necessary
	int i = h->count;
	while(PARENT(i) >= 1)
	{
		if(h->heap_arr[i]->key <= h->heap_arr[PARENT(i)]->key)
			break; //we're good.
		else
		{
			//swap i with its parent
			h->heap_arr[i] = h->heap_arr[PARENT(i)];
			h->heap_arr[PARENT(i)] = e; //the to-be-inserted new element (e) is always a copy of the ith element
			i = PARENT(i);
		}
	}

	return 0;
}

int heap_remove_max(heap *h)
{
	if(h == NULL)
	{
		fprintf(stderr, "ERROR: heap handle is NULL in heap_max_remove!\n");
		return 1;
	}
	if(h->count == 0)
	{
		INFO(fprintf(stderr, "Cannot remove max; Heap is empty\n"));
		return 1;
	}

	heap_element* max_element = h->heap_arr[1];
	h->heap_arr[1] = h->heap_arr[h->count];
	h->count = h->count - 1;

	//Shifting down if necessary
	int i = 1;
	while(LCHILD(i) <= h->count)
	{
		//find the largest child first
		int largest = LCHILD(i);
		if(RCHILD(i) <= h->count)
		    if(h->heap_arr[RCHILD(i)]->key > h->heap_arr[largest]->key)
		        largest = RCHILD(i);

		if(h->heap_arr[i]->key >= h->heap_arr[largest]->key)
			break; //we're good.
		else
		{
			//swap i with largest
			h->heap_arr[i] = h->heap_arr[largest];
			h->heap_arr[largest] = h->heap_arr[h->count + 1];//count + 1 is a copy of the ith element
			i = largest;
		}
	}

	MPL_free(max_element);
	return 0;
}

int heap_remove_index(heap *h, int index)
{
    if(h == NULL)
    {
        fprintf(stderr, "ERROR: heap handle is NULL in heap_max_remove!\n");
        return 1;
    }
    if(h->count == 0)
    {
        INFO(fprintf(stderr, "Cannot remove max; Heap is empty\n"));
        return 1;
    }

    if(index < 1 || index > h->count)
    {
        fprintf(stderr, "Invalid index (%d) passed to heap_remove_index! heap count = %d\n", index, h->count);
        return 1;
    }

    heap_element* removed_element = h->heap_arr[index];
    h->heap_arr[index] = h->heap_arr[h->count];
    h->count = h->count - 1;

    if(index == h->count + 1) //We are removing the last element
    {
        MPL_free(removed_element);
        return 0;
    }

    int i = index;
    //Shifting up if necessary
    while(PARENT(i) >= 1)
    {
        if(h->heap_arr[i]->key <= h->heap_arr[PARENT(i)]->key)
            break; //we're good.
        else
        {
            //swap i with its parent
            h->heap_arr[i] = h->heap_arr[PARENT(i)];
            h->heap_arr[PARENT(i)] = h->heap_arr[h->count + 1];//count + 1 is a copy of the ith element
            i = PARENT(i);
        }
    }

    //Shifting down if necessary
    while(LCHILD(i) <= h->count)
    {
        //find the largest child first
        int largest = LCHILD(i);
        if(RCHILD(i) <= h->count)
            if(h->heap_arr[RCHILD(i)]->key > h->heap_arr[largest]->key)
                largest = RCHILD(i);

        if(h->heap_arr[i]->key >= h->heap_arr[largest]->key)
            break; //we're good.
        else
        {
            //swap i with largest
            h->heap_arr[i] = h->heap_arr[largest];
            h->heap_arr[largest] = h->heap_arr[h->count + 1];//count + 1 is a copy of the ith element
            i = largest;
        }
    }

    MPL_free(removed_element);
    return 0;
}

int heap_find_value(heap *h, int value)
{
    int i;
    for(i = 1; i <= h->count; i++)
    {
        if(h->heap_arr[i]->value == value)
            return i;
    }
    return -1;
}

int heap_get_keys_array(heap *h, int *keys)
{
    if(h == NULL)
    {
        fprintf(stderr, "ERROR: heap handle is NULL in heap_get_keys_array!\n");
        return -1;
    }
    if(heap_is_empty(h))
    {
        INFO(printf("Cannot get keys array; Heap is empty\n"));
        return -1;
    }

    int i;
    for(i = 1; i <= h->count; i++)
    {
        keys[i-1] = h->heap_arr[i]->key;
    }
    return 0;
}

int heap_get_values_array(heap *h, int *values)
{
    if(h == NULL)
    {
        fprintf(stderr, "ERROR: heap handle is NULL in heap_get_values_array!\n");
        return -1;
    }
    if(heap_is_empty(h))
    {
        INFO(printf("Cannot get values array; Heap is empty\n"));
        return -1;
    }

    int i;
    for(i = 1; i <= h->count; i++)
    {
        values[i-1] = h->heap_arr[i]->value;
    }
    return 0;
}

int heap_peek_max_key(heap *h)
{
	if(h == NULL)
	{
		fprintf(stderr, "ERROR: heap handle is NULL in heap_peek_max_key!\n");
		return -1;
	}
	if(h->count == 0)
	{
		INFO(printf("Cannot peek max key; Heap is empty\n"));
		return -1;
	}

	return h->heap_arr[1]->key;
}

int heap_peek_max_value(heap *h)
{
	if(h == NULL)
	{
		fprintf(stderr, "ERROR: heap handle is NULL in heap_peek_max_value!\n");
		return -1;
	}
	if(h->count == 0)
	{
		INFO(printf("Cannot peek max value; Heap is empty\n"));
		return -1;
	}

	return h->heap_arr[1]->value;
}

int heap_peek_max_paired(heap *h)
{
    if(h == NULL)
    {
        fprintf(stderr, "ERROR: heap handle is NULL in heap_peek_max_paired!\n");
        return -1;
    }
    if(h->count == 0)
    {
        INFO(printf("Cannot peek max paired; Heap is empty\n"));
        return -1;
    }

    return h->heap_arr[1]->paired;
}

int heap_peek_key_at_index(heap *h, int index)
{
    if(h == NULL)
    {
        fprintf(stderr, "ERROR: heap handle is NULL in heap_peek_key_at_index\n");
        return -1;
    }
    if(h->count == 0)
    {
        INFO(printf("Cannot peek key; Heap is empty\n"));
        return -1;
    }
    if(index < 1 || index > h->count)
    {
        fprintf(stderr, "Invalid index (%d) passed to heap_peek_key_at_index! heap count = %d\n", index, h->count);
        return -1;
    }

    return h->heap_arr[index]->key;
}

int heap_peek_value_at_index(heap *h, int index)
{
    if(h == NULL)
    {
        fprintf(stderr, "ERROR: heap handle is NULL in heap_peek_value_at_index\n");
        return -1;
    }
    if(h->count == 0)
    {
        INFO(printf("Cannot peek value; Heap is empty\n"));
        return -1;
    }
    if(index < 1 || index > h->count)
    {
        fprintf(stderr, "Invalid index (%d) passed to heap_peek_value_at_index! heap count = %d\n", index, h->count);
        return -1;
    }

    return h->heap_arr[index]->value;
}
