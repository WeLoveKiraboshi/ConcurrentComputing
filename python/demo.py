from numba import cuda
import numpy as np

@cuda.jit
def increment_by_one(arr):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    pos = tx + ty * bw
    arr[pos] += 1

arr = np.arange(5)
print('Input: {}'.format(arr))

threadsperblock = 32
blockspergrid = 1
increment_by_one[blockspergrid, threadsperblock](arr)
print('Output: {}'.format(arr))
