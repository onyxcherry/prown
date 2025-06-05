#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <random>
#include <cstring>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
// url: https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/optimize-data-transfers/bandwidthtest.cu
inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime Error: %s\n",
                cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}

__device__ unsigned char set_last_2_bits(unsigned char byte, unsigned char bits)
{
    return (byte & 0xFC) | (bits & 0x03);
}

__device__ unsigned char get_2_bits(unsigned char byte, int pos)
{
    return (byte >> (6 - 2 * pos)) & 0x03;
}

__global__ void embed_kernel(unsigned char *payload, unsigned char *image, int payload_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= payload_size)
        return;

    unsigned char byte = payload[idx];
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        unsigned char bits = get_2_bits(byte, j);
        image[4 * idx + j] = set_last_2_bits(image[4 * idx + j], bits);
    }
}

__global__ void extract_kernel(unsigned char *image, unsigned char *extracted, int data_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= data_size)
        return;

    unsigned char byte = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        byte |= (image[4 * idx + j] & 0x03) << (6 - 2 * j);
    }
    extracted[idx] = byte;
}

void embed_data_cuda(unsigned char *h_image,
                     unsigned char *h_payload,
                     int payload_size,
                     size_t image_bytes,
                     unsigned char *d_image,
                     unsigned char *d_payload)
{
    checkCuda(cudaMemcpy(d_image, h_image, image_bytes,
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_payload, h_payload, payload_size,
                         cudaMemcpyHostToDevice));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int threadsPerBlock = prop.maxThreadsPerBlock;
    int blocks = (payload_size + threadsPerBlock - 1) / threadsPerBlock;
    embed_kernel<<<blocks, threadsPerBlock>>>(d_payload, d_image, payload_size);

    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy(h_image, d_image, image_bytes, cudaMemcpyDeviceToHost));
}

void extract_data_cuda(unsigned char *h_image,
                       unsigned char *h_extracted,
                       int payload_size,
                       size_t image_bytes,
                       unsigned char *d_image,
                       unsigned char *d_extracted)
{
    checkCuda(cudaMemcpy(d_image, h_image, image_bytes, cudaMemcpyHostToDevice));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int threadsPerBlock = prop.maxThreadsPerBlock;
    int blocks = (payload_size + threadsPerBlock - 1) / threadsPerBlock;
    extract_kernel<<<blocks, threadsPerBlock>>>(d_image, d_extracted, payload_size);

    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy(h_extracted, d_extracted, payload_size, cudaMemcpyDeviceToHost));
}

int measure(int times, int base, size_t image_size)
{
    const size_t total_image_pixels = 4 * image_size; // RGBA = 4 channels
    const long long different_payload_sizes_set_count = image_size / base;

    unsigned char *h_image = nullptr;
    unsigned char *h_payload = nullptr;
    unsigned char *h_extracted = nullptr;
    checkCuda(cudaMallocHost(&h_image, total_image_pixels));
    checkCuda(cudaMallocHost(&h_payload, image_size));
    checkCuda(cudaMallocHost(&h_extracted, image_size));

    unsigned char *d_image = nullptr;
    unsigned char *d_payload = nullptr;
    unsigned char *d_extracted = nullptr;
    checkCuda(cudaMalloc(&d_image, total_image_pixels));
    checkCuda(cudaMalloc(&d_payload, image_size));
    checkCuda(cudaMalloc(&d_extracted, image_size));

    std::mt19937 gen(42);
    std::uniform_int_distribution<> dist(0, 255);
    for (size_t i = 0; i < total_image_pixels; ++i)
        h_image[i] = static_cast<unsigned char>(dist(gen));

    for (int s = 1; s <= different_payload_sizes_set_count; ++s)
    {
        std::vector<int64_t> embeding_durs;
        std::vector<int64_t> extracting_durs;
        const long long payload_size = s * base;

        for (int i = 0; i < times; ++i)
        {
            for (int i = 0; i < payload_size; ++i)
                h_payload[i] = static_cast<unsigned char>(dist(gen));

            auto start = std::chrono::high_resolution_clock::now();
            embed_data_cuda(
                h_image,
                h_payload,
                payload_size,
                total_image_pixels,
                d_image,
                d_payload);

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            embeding_durs.push_back(duration.count());

            auto start2 = std::chrono::high_resolution_clock::now();

            extract_data_cuda(
                h_image,
                h_extracted,
                payload_size,
                total_image_pixels,
                d_image,
                d_extracted);

            auto end2 = std::chrono::high_resolution_clock::now();
            auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
            extracting_durs.push_back(duration2.count());

            assert(std::memcmp(h_payload, h_extracted, payload_size) == 0);
        }
        std::cout << "enc(payload, pixels)" << '\t' << payload_size << '\t' << total_image_pixels;
        for (const auto d : embeding_durs)
        {
            std::cout << '\t' << d;
        }
        std::cout << std::endl;

        std::cout << "dec(payload, pixels)" << '\t' << payload_size << '\t' << total_image_pixels;
        for (const auto d : extracting_durs)
        {
            std::cout << '\t' << d;
        }
        std::cout << std::endl;
    }

    checkCuda(cudaFree(d_image));
    checkCuda(cudaFree(d_payload));
    checkCuda(cudaFree(d_extracted));
    checkCuda(cudaFreeHost(h_image));
    checkCuda(cudaFreeHost(h_payload));
    checkCuda(cudaFreeHost(h_extracted));

    return 0;
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cerr << "Too few arguments\n";
        return 1;
    }
    return measure(std::stoi(argv[1]), std::stoi(argv[2]), std::stoi(argv[3])) ? 0 : 1;
}
