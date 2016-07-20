#cython: boundscheck=False, wraparound=False

# Functions that will be called to support reading
# directly from a memory buffer (instead of file)

cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc.stdint cimport uintptr_t
import ctypes


ctypedef np.uint32_t UINT64
ctypedef np.uint32_t UINT32
ctypedef np.int32_t INT32
ctypedef np.uint8_t UINT8


cdef extern from "openjpeg.h":
    #struct opj_stream_t:
    #    pass

    ctypedef UINT32 OPJ_SIZE_T
    ctypedef UINT32 OPJ_BOOL
    ctypedef UINT64 OPJ_UINT64
    ctypedef INT32 OPJ_OFF_T
    ctypedef void* opj_stream_t
    
    # define the callbacks
    ctypedef OPJ_SIZE_T (* opj_stream_read_fn) (void * p_buffer, OPJ_SIZE_T p_nb_bytes, void * p_user_data)
    ctypedef OPJ_SIZE_T (* opj_stream_write_fn) (void * p_buffer, OPJ_SIZE_T p_nb_bytes, void * p_user_data)
    ctypedef OPJ_OFF_T (* opj_stream_skip_fn) (OPJ_OFF_T p_nb_bytes, void * p_user_data)
    ctypedef OPJ_BOOL (* opj_stream_seek_fn) (OPJ_OFF_T p_nb_bytes, void * p_user_data)
    ctypedef void (* opj_stream_free_user_data_fn) (void * p_user_data)

    # define the wrapper functions
    opj_stream_t* opj_stream_create(OPJ_SIZE_T p_buffer_size, OPJ_BOOL p_is_input) nogil
    void opj_stream_destroy(opj_stream_t* p_stream) nogil
    void opj_stream_set_read_function(opj_stream_t* p_stream, opj_stream_read_fn p_function) nogil
    void opj_stream_set_write_function(opj_stream_t* p_stream, opj_stream_write_fn p_function) nogil
    void opj_stream_set_skip_function(opj_stream_t* p_stream, opj_stream_skip_fn p_function) nogil
    void opj_stream_set_seek_function(opj_stream_t* p_stream, opj_stream_seek_fn p_function) nogil
    void opj_stream_set_user_data (opj_stream_t* p_stream, void * p_data, opj_stream_free_user_data_fn p_function) nogil
    void opj_stream_set_user_data_length(opj_stream_t* p_stream, OPJ_UINT64 data_length) nogil

cdef struct MemBuffer_t:
    UINT8 *data
    UINT32 pos
    UINT32 length
    
cdef void opj_destroy_mem_buffer(MemBuffer_t *buffer) nogil:
    buffer.data = NULL
    buffer.pos = 0
    buffer.length = 0
    free(<void *> buffer)
    
cdef OPJ_SIZE_T opj_read_from_mem_buffer(void *p_out_buffer, OPJ_SIZE_T p_nb_bytes, MemBuffer_t *p_mem_buffer) nogil:
    cdef:
        UINT32 actual_bytes_nb = p_nb_bytes
        UINT8 *out_buffer = <UINT8 *>p_out_buffer
        UINT8 *in_buffer = <UINT8 *>p_mem_buffer.data

    if p_mem_buffer.pos + p_nb_bytes > p_mem_buffer.length:
        actual_bytes_nb = p_mem_buffer.length - p_mem_buffer.pos
    memcpy(out_buffer, &in_buffer[p_mem_buffer.pos], actual_bytes_nb)

    p_mem_buffer.pos += actual_bytes_nb
    return actual_bytes_nb

cdef OPJ_SIZE_T opj_write_from_mem_buffer(void *p_data_buffer, OPJ_SIZE_T p_nb_bytes, MemBuffer_t *p_mem_buffer) nogil:
    #with gil:
    #    raise BufferError()
    #return -1 # Not implemented - read only buffer
    cdef:
        UINT32 actual_bytes_nb = p_nb_bytes
        UINT8 *data_buffer = <UINT8 *>p_data_buffer

    if p_mem_buffer.pos + p_nb_bytes > p_mem_buffer.length:
        actual_bytes_nb = p_mem_buffer.length - p_mem_buffer.pos
    memcpy(&p_mem_buffer.data[p_mem_buffer.pos], data_buffer, actual_bytes_nb)
    p_mem_buffer.pos += actual_bytes_nb
    return actual_bytes_nb

cdef OPJ_OFF_T opj_skip_from_mem_buffer(OPJ_OFF_T p_nb_bytes, MemBuffer_t *p_mem_buffer) nogil:
    if p_mem_buffer.pos + p_nb_bytes < 0 or p_mem_buffer.pos + p_nb_bytes > p_mem_buffer.length:
        return -1

    p_mem_buffer.pos += p_nb_bytes
    return p_nb_bytes


cdef OPJ_BOOL opj_seek_from_mem_buffer(OPJ_OFF_T p_nb_bytes, MemBuffer_t *p_mem_buffer) nogil:
    if p_nb_bytes > p_mem_buffer.length:
        return 0

    p_mem_buffer.pos = p_nb_bytes
    return 1 #OPJ_TRUE



cdef opj_stream_t* opj_stream_create_memory_stream (
        UINT8 *in_stream,
        OPJ_SIZE_T stream_size) nogil:
    cdef:
        MemBuffer_t *buffer = NULL
        opj_stream_t * l_stream = NULL
        OPJ_SIZE_T p_size = 0x100000 # same as OPJ_J2K_STREAM_CHUNK_SIZE

    if in_stream is NULL:
        return NULL

    buffer = <MemBuffer_t *> malloc(sizeof(MemBuffer_t))

    if buffer is NULL:
        with gil:
            raise MemoryError()

    # Create the out stream, and make it read-only
    l_stream = opj_stream_create(p_size, 1)
    if l_stream is NULL:
        free(<void *> buffer)
        return NULL

    # Initialize the buffer elements
    buffer.data = in_stream
    buffer.pos = 0
    buffer.length = stream_size

    opj_stream_set_user_data(l_stream, buffer, <opj_stream_free_user_data_fn> opj_destroy_mem_buffer)
    opj_stream_set_user_data_length(l_stream, stream_size)
    opj_stream_set_read_function(l_stream, <opj_stream_read_fn> opj_read_from_mem_buffer)
    opj_stream_set_write_function(l_stream, <opj_stream_write_fn> opj_write_from_mem_buffer)
    opj_stream_set_skip_function(l_stream, <opj_stream_skip_fn> opj_skip_from_mem_buffer)
    opj_stream_set_seek_function(l_stream, <opj_stream_seek_fn> opj_seek_from_mem_buffer)

    return l_stream


cpdef opj_stream_create_memory_stream_python (
        np.ndarray[np.uint8_t, ndim=1, mode="c"] buffer):

    return <uintptr_t> opj_stream_create_memory_stream(&buffer[0], buffer.shape[0])

