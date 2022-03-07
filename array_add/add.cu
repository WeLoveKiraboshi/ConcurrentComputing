#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

cudaError_t checkCuda(cudaError_t status)
{
#if defined(NDEBUG) || defined(_NDEBUG)
    if (status != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(status)
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }
#endif
    return status;
}

__global__ void add_n_kernel(int* ptr, const unsigned int n, const size_t size)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < size; i += stride)
    {
        for (unsigned int j = 0; j < n; j++)
        {
            ptr[i] += 1;
        }
    }
}

void cpu_add_n(int* ptr, const unsigned int n, const size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        for (unsigned int j = 0; j < n; j++)
        {
            ptr[i] += 1;
        }
    }
}

void cuda_add_n(int* h_data, int* d_data, const unsigned int n,
                const size_t size, const unsigned int num_streams,
                cudaStream_t* streams)
{
    const size_t block_size{256};
    const size_t stream_size{size / num_streams};
    size_t grid_size = 1;
    if (stream_size / block_size != 0)
    {
        grid_size = stream_size / block_size;
    }
    const size_t stream_bytes{stream_size * sizeof(int)};

    for (unsigned int i = 0; i < num_streams - 1; i++)
    {
        const size_t offset = i * stream_size;
        checkCuda(cudaMemcpyAsync(d_data + offset, h_data + offset,
                                  stream_bytes, cudaMemcpyHostToDevice,
                                  streams[i]));
        add_n_kernel<<<grid_size, block_size, 0, streams[i]>>>(d_data + offset,
                                                               n, stream_size);
        checkCuda(cudaMemcpyAsync(h_data + offset, d_data + offset,
                                  stream_bytes, cudaMemcpyDeviceToHost,
                                  streams[i]));
    }
    const size_t stream_size_remain = size - (num_streams - 1) * stream_size;
    const size_t stream_bytes_remain = stream_size_remain * sizeof(int);
    const size_t offset = (num_streams - 1) * stream_size;
    checkCuda(cudaMemcpyAsync(d_data + offset, h_data + offset,
                              stream_bytes_remain, cudaMemcpyHostToDevice,
                              streams[num_streams - 1]));
    add_n_kernel<<<grid_size, block_size, 0, streams[num_streams - 1]>>>(
            d_data + offset, n, stream_size_remain);
    checkCuda(cudaMemcpyAsync(h_data + offset, d_data + offset,
                              stream_bytes_remain, cudaMemcpyDeviceToHost,
                              streams[num_streams - 1]));

    return;
}

void thread_add_n(int* h_data, int* d_data, const unsigned int n,
                  const size_t size, const unsigned int num_streams,
                  cudaStream_t* streams)
{
    // CPU add
    if (num_streams == 0)
    {
        cpu_add_n(h_data, n, size);
    }
        // CUDA add
    else
    {
        cuda_add_n(h_data, d_data, n, size, num_streams, streams);
    }
    return;
}

// Multithread add_n
// Each thread uses n stream
void multithread_add_n(int* h_data, int* d_data, const unsigned int n,
                       const size_t size, const unsigned int num_threads,
                       const unsigned int num_streams_per_thread,
                       const bool verbose, const unsigned int num_tests)
{

    const unsigned int num_streams{num_threads * num_streams_per_thread};

    std::vector<cudaStream_t> streams(num_streams);
    for (unsigned int i = 0; i < streams.size(); i++)
    {
        checkCuda(cudaStreamCreate(&streams.at(i)));
    }

    float duration_total = 0;

    for (int k = 0; k < num_tests; k++)
    {
        std::vector<std::thread> threads;
        const size_t thread_size{size / num_threads};

        std::chrono::steady_clock::time_point begin =
                std::chrono::steady_clock::now();

        for (unsigned int i = 0; i < num_threads - 1; i++)
        {
            const size_t offset = i * thread_size;
            threads.emplace_back(thread_add_n, h_data + offset, d_data + offset,
                                 n, thread_size, num_streams_per_thread,
                                 streams.data() + i * num_streams_per_thread);
        }
        const size_t thread_size_remain =
                size - (num_threads - 1) * thread_size;
        const size_t offset = (num_threads - 1) * thread_size;
        threads.emplace_back(thread_add_n, h_data + offset, d_data + offset, n,
                             thread_size_remain, num_streams_per_thread,
                             streams.data() +
                             (num_threads - 1) * num_streams_per_thread);

        for (unsigned int i = 0; i < num_streams; i++)
        {
            checkCuda(cudaStreamSynchronize(streams.at(i)));
        }

        for (unsigned int i = 0; i < num_threads; i++)
        {
            threads.at(i).join();
        }

        std::chrono::steady_clock::time_point end =
                std::chrono::steady_clock::now();

        duration_total +=
                std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
                        .count();
    }

    for (unsigned int i = 0; i < streams.size(); i++)
    {
        checkCuda(cudaStreamDestroy(streams.at(i)));
    }

    if (verbose)
    {
        std::cout << "Average Latency: " << std::setprecision(2) << std::fixed
                  << duration_total / 1000 / num_tests << " ms"
                  << std::endl;
    }

    return;
}

bool verify_add_n(const std::vector<int>& vector,
                  const std::vector<int>& vector_original, const unsigned int n)
{
    if (vector.size() != vector_original.size())
    {
        return false;
    }
    for (size_t i = 0; i < vector.size(); i++)
    {
        if (vector.at(i) - vector_original.at(i) != n)
        {
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[])
{
    size_t size{10000000}; // 10 ** 7
    unsigned int n{100};
    unsigned int num_threads{1};
    unsigned int num_streams_per_thread{16};

    if (argc == 5)
    {
        size = atoi(argv[1]);
        n = atoi(argv[2]);
        num_threads = atoi(argv[3]);
        num_streams_per_thread = atoi(argv[4]);
    }

    std::cout << "Array Size: " << size << std::endl;
    std::cout << "Number of Additions: " << n << std::endl;
    std::cout << "Number of Threads: " << num_threads << std::endl;
    std::cout << "Number of Streams Per Thread: " << num_streams_per_thread
              << std::endl;

    // Set CUDA device
    checkCuda(cudaSetDevice(0));

    // Create a vector and initialize it with zeros
    std::vector<int> vector(size, 0);
    std::vector<int> vector_clone{vector};

    int* h_data = vector.data();
    int* d_data;
    const size_t bytes = size * sizeof(int);
    checkCuda(cudaMalloc((void**)&d_data, bytes));

    multithread_add_n(h_data, d_data, n, size, num_threads,
                      num_streams_per_thread, false, 1);

    assert(verify_add_n(vector, vector_clone, n) &&
           "The add_n implementation is incorrect.");

    // Warm up
    multithread_add_n(h_data, d_data, n, size, num_threads,
                      num_streams_per_thread, false, 100);
    // Measure latency
    multithread_add_n(h_data, d_data, n, size, num_threads,
                      num_streams_per_thread, true, 1000);

    checkCuda(cudaFree(d_data));
    // Reserved for cuda-memcheck
    cudaDeviceReset();
}