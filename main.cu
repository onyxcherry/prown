#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <fstream>
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

void embed_data_cuda(std::vector<unsigned char> &payload, std::vector<unsigned char> &image)
{
    int payload_size = payload.size();
    int image_size = image.size();

    unsigned char *d_payload, *d_image;

    cudaMalloc(&d_payload, payload_size);
    cudaMalloc(&d_image, image_size);

    cudaMemcpy(d_payload, payload.data(), payload_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_image, image.data(), image_size, cudaMemcpyHostToDevice);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int threadsPerBlock = prop.maxThreadsPerBlock;
    int blocks = (payload_size + threadsPerBlock - 1) / threadsPerBlock;

    embed_kernel<<<blocks, threadsPerBlock>>>(d_payload, d_image, payload_size);
    cudaDeviceSynchronize();

    cudaMemcpy(image.data(), d_image, image_size, cudaMemcpyDeviceToHost);

    cudaFree(d_payload);
    cudaFree(d_image);
}

std::vector<unsigned char> extract_data_cuda(const std::vector<unsigned char> &image, int data_size)
{
    size_t image_bytes = image.size();
    size_t data_bytes = data_size;

    unsigned char *h_imagePinned = nullptr;
    unsigned char *h_extractedPinned = nullptr;
    checkCuda(cudaMallocHost(&h_imagePinned, image_bytes));
    checkCuda(cudaMallocHost(&h_extractedPinned, data_bytes));

    memcpy(h_imagePinned, image.data(), image_bytes);
    memset(h_extractedPinned, 0, data_bytes);

    unsigned char *d_image = nullptr;
    unsigned char *d_extracted = nullptr;
    checkCuda(cudaMalloc(&d_image, image_bytes));
    checkCuda(cudaMalloc(&d_extracted, data_bytes));

    checkCuda(cudaMemcpy(d_image, h_imagePinned, image_bytes, cudaMemcpyHostToDevice));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int threadsPerBlock = prop.maxThreadsPerBlock;
    int blocks = (data_size + threadsPerBlock - 1) / threadsPerBlock;

    extract_kernel<<<blocks, threadsPerBlock>>>(d_image, d_extracted, data_size);

    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy(h_extractedPinned, d_extracted, data_bytes,
                         cudaMemcpyDeviceToHost));

    std::vector<unsigned char> extracted(data_size);
    memcpy(extracted.data(), h_extractedPinned, data_bytes);

    checkCuda(cudaFree(d_image));
    checkCuda(cudaFree(d_extracted));
    checkCuda(cudaFreeHost(h_imagePinned));
    checkCuda(cudaFreeHost(h_extractedPinned));

    return extracted;
}

int measure(int times, const int base, const int image_size)
{
    const long long total_image_pixels = 4 * image_size; // RGBA = 4 channels
    const long long different_payload_sizes_set_count = image_size / base;

    std::vector<unsigned char> image(total_image_pixels);
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dist(0, 255);
    for (auto &c : image)
        c = dist(gen);

    for (int s = 1; s <= different_payload_sizes_set_count; ++s)
    {
        std::vector<int64_t> embeding_durs;
        std::vector<int64_t> extracting_durs;
        const long long payload_size = s * base;

        for (int i = 0; i < times; ++i)
        {
            std::vector<unsigned char> payload(payload_size);
            for (auto &c : payload)
                c = dist(gen);

            std::vector<unsigned char> image_copy = image;

            auto start = std::chrono::high_resolution_clock::now();

            embed_data_cuda(payload, image_copy);

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            embeding_durs.push_back(duration.count());

            auto start2 = std::chrono::high_resolution_clock::now();

            auto extracted = extract_data_cuda(image_copy, payload_size);

            auto end2 = std::chrono::high_resolution_clock::now();
            auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
            extracting_durs.push_back(duration2.count());

            assert(payload == extracted);
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