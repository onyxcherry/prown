#include "lodepng.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <chrono>
#include <random>
#include <cassert>
#ifdef _OPENMP
#include <omp.h>
#endif

std::vector<unsigned char> generate_random_chars(const size_t length)
{
  unsigned int seed = 42;
  std::mt19937 gen(seed);
  // bytes codes
  std::uniform_int_distribution<> dist(0, 255);

  std::vector<unsigned char> data(length);
  for (size_t i = 0; i < length; ++i)
  {
    data[i] = static_cast<unsigned char>(dist(gen));
  }
  return data;
}

unsigned char set_last_2_bits(unsigned char byte, unsigned char bits)
{
  return (byte & 0xFC) | (bits & 0x03);
}

unsigned char get_2_bits(unsigned char byte, int pos)
{
  return (byte >> (6 - 2 * pos)) & 0x03;
}

void embed_data(std::vector<unsigned char> &payload, std::vector<unsigned char> &image)
{
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(payload, image)
#endif
  for (int i = 0; i < (int)payload.size(); ++i)
  {
    unsigned char byte = payload[i];
    // walk over RGBA channels separately
    for (int j = 0; j < 4; ++j)
    {
      unsigned char bits = get_2_bits(byte, j);
      image[4 * i + j] = set_last_2_bits(image[4 * i + j], bits);
    }
  }
}

std::vector<unsigned char> extract_data(std::vector<unsigned char> &image, uint32_t data_size)
{
  std::vector<unsigned char> extracted(data_size);

#ifdef _OPENMP
#pragma omp parallel for default(none) shared(extracted, image, data_size)
#endif
  for (uint32_t i = 0; i < data_size; ++i)
  {
    unsigned char byte = 0;
    for (int j = 0; j < 4; ++j)
    {
      byte |= (image[4 * i + j] & 0x03) << (6 - 2 * j);
    }
    extracted[i] = byte;
  }
  return extracted;
}

int measure(const int times, const int base, const int image_size)
{
  const long long total_image_pixels = 4 * image_size; // 4 channels
  const long long different_payload_sizes_set_count = image_size / base;
  auto image = generate_random_chars(total_image_pixels);

  for (int s = 1; s <= different_payload_sizes_set_count; s++)
  {
    std::vector<int64_t> embeding_durs;
    std::vector<int64_t> extracting_durs;
    const long long payload_size = s * base;
    for (int i = 0; i < times; i++)
    {
      auto payload = generate_random_chars(payload_size);

      auto start = std::chrono::high_resolution_clock::now();

      embed_data(payload, image);

      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      embeding_durs.push_back(duration.count());

      auto start2 = std::chrono::high_resolution_clock::now();

      auto extracted = extract_data(image, payload_size);

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

bool embed_file_omp(const char *image_file, const char *data_file, const char *output_file)
{
  std::vector<unsigned char> image;
  unsigned width, height;

  if (lodepng::decode(image, width, height, image_file))
  {
    std::cerr << "Error loading image.\n";
    return false;
  }

  std::ifstream fin(data_file, std::ios::binary);
  std::vector<unsigned char> data((std::istreambuf_iterator<char>(fin)), {});
  fin.close();

  uint32_t data_size = static_cast<uint32_t>(data.size());
  std::vector<unsigned char> payload;

  // Header for storing payload length
  for (int i = 3; i >= 0; --i)
  {
    // & 0xFF is for trimming the output to only 8 bits
    payload.push_back((data_size >> (i * 8)) & 0xFF);
  }
  payload.insert(payload.end(), data.begin(), data.end());

  size_t total_pixels = image.size() / 4;
  if (payload.size() > total_pixels)
  {
    std::cerr << "Image too small.\n";
    return false;
  }

  embed_data(payload, image);

  if (lodepng::encode(output_file, image, width, height))
  {
    std::cerr << "Error writing image.\n";
    return false;
  }

  std::cout << "Embedded into: " << output_file << "\n";
  return true;
}

bool extract_file_omp(const char *image_file, const char *output_data_file)
{
  std::vector<unsigned char> image;
  unsigned width, height;

  if (lodepng::decode(image, width, height, image_file))
  {
    std::cerr << "Error loading image.\n";
    return false;
  }

  // 4 is byte-size of int, the type guarantees this
  uint32_t data_size = 0;
  size_t pixel_count = image.size() / 4;

  for (int i = 0; i < 4; ++i)
  {
    unsigned char byte = 0;
    for (int j = 0; j < 4; ++j)
    {
      byte |= (image[4 * i + j] & 0x03) << (6 - 2 * j);
    }
    data_size = (data_size << 8) | byte;
  }

  if (data_size > pixel_count - 4)
  {
    std::cerr << "Invalid or corrupted data.\n";
    return false;
  }

  auto extracted = extract_data(image, data_size);

  std::ofstream fout(output_data_file, std::ios::binary);
  fout.write(reinterpret_cast<char *>(extracted.data() + 4), extracted.size() - 4);
  fout.close();

  std::cout << "Extracted to: " << output_data_file << "\n";
  return true;
}

int main(int argc, char *argv[])
{
  if (argc < 2)
  {
    std::cerr << "Usage:\n"
              << "  " << argv[0] << " encode input.png data.bin output.png\n"
              << "  " << argv[0] << " decode stego.png output.bin\n"
              << "  " << argv[0] << " measure 6 1000000 8294400\n";
    return 1;
  }

  std::string command = argv[1];

  if (command == "encode")
  {
    if (argc < 5)
    {
      std::cerr << "Too few arguments\n";
      return 1;
    }
    return embed_file_omp(argv[2], argv[3], argv[4]) ? 0 : 1;
  }
  else if (command == "decode")
  {
    if (argc < 4)
    {
      std::cerr << "Too few arguments\n";
      return 1;
    }
    return extract_file_omp(argv[2], argv[3]) ? 0 : 1;
  }
  else if (command == "measure")
  {
    if (argc < 5)
    {
      std::cerr << "Too few arguments\n";
      return 1;
    }
    return measure(std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4])) ? 0 : 1;
  }

  std::cerr << "Unknown command: " << command << "\n";
  return 1;
}
