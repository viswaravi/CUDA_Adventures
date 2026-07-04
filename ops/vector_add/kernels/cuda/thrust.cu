#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/transform.h>

cudaError_t addWithCudaThrust(unsigned int size) {

  thrust::device_vector<int> tdev_a(size, 1);
  thrust::device_vector<int> tdev_b(size, 2);
  thrust::device_vector<int> tdev_c(size, 0);

  thrust::host_vector<int> host_c;

  // Get Raw pointer
  int* raw_ptr_a = thrust::raw_pointer_cast(tdev_a.data());
  int* raw_ptr_b = thrust::raw_pointer_cast(tdev_b.data());
  int* raw_ptr_c = thrust::raw_pointer_cast(tdev_c.data());

  cudaError_t cudaStatus;
  // Choose which GPU to run on, change this on a multi-GPU system.
  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr,
            "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    goto Error;
  }

  // Launch a kernel on the GPU with one thread for each element.
  // addKernel << <1, size >> > (raw_ptr_c, raw_ptr_a, raw_ptr_b);

  // Check for any errors launching the kernel
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "addKernel launch failed: %s\n",
            cudaGetErrorString(cudaStatus));
    goto Error;
  }

  // cudaDeviceSynchronize waits for the kernel to finish, and returns
  // any errors encountered during the launch.
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr,
            "cudaDeviceSynchronize returned error code %d after launching "
            "addKernel!\n",
            cudaStatus);
    goto Error;
  }

  // Copy result back to host for verification
  host_c = tdev_c;
  for (int i = 0; i < 5; ++i) {
    std::cout << host_c[i] << " ";
  }
  std::cout << std::endl;

Error:
  return cudaStatus;
}