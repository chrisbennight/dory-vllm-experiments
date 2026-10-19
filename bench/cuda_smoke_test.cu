// Checks that the project CUDA toolchain builds code the installed driver can run.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void square(int* values, int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) values[i] = i * i;
}

int main() {
  const int count = 1024;
  int* device = nullptr;
  int host[count];
  if (cudaMalloc(&device, count * sizeof(int)) != cudaSuccess) return 2;
  square<<<(count + 255) / 256, 256>>>(device, count);
  cudaError_t status = cudaMemcpy(host, device, count * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(device);
  if (status != cudaSuccess) {
    std::printf("CUDA error: %s\n", cudaGetErrorString(status));
    return 1;
  }
  for (int i = 0; i < count; ++i) {
    if (host[i] != i * i) {
      std::printf("mismatch at %d\n", i);
      return 1;
    }
  }
  std::printf("CUDA smoke test passed\n");
  return 0;
}
