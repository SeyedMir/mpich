/*---------------------------------------------------------------------*/
/*     (C) Copyright 2017 Parallel Processing Research Laboratory      */
/*                   Queen's University at Kingston                    */
/*                Neighborhood Collective Communication                */
/*                    Seyed Hessamedin Mirsadeghi                      */
/*---------------------------------------------------------------------*/
//SHM
typedef struct heap_element {
	int key;
	int value;
	int paired;
}heap_element;

typedef struct shm_heap {
	int count;
	int arr_size;
	heap_element **heap_arr;
}shm_heap;

int heap_init(shm_heap *h, int arr_size);
int heap_insert(shm_heap *h,  heap_element *e);
int heap_remove_max(shm_heap *h);
int heap_remove_index(shm_heap *h, int index);
int heap_find_value(shm_heap *h, int value);
int heap_peek_max_key(shm_heap *h);
int heap_peek_max_value(shm_heap *h);
int heap_get_keys_array(shm_heap *h, int *keys);
int heap_get_values_array(shm_heap *h, int *values);
int heap_is_empty(shm_heap *h);
int heap_peek_key_at_index(shm_heap *h, int index);
int heap_peek_value_at_index(shm_heap *h, int index);
int heap_free_array(shm_heap *h);
//int heap_remove_elem(shm_heap *h, heap_element *e);
