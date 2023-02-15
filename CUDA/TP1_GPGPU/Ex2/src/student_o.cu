/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 2: Addition de vecteurs
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"

namespace IMAC
{
	#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
	inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
	{
   		if (code != cudaSuccess)
   		{
      	fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      	if (abort) exit(code);
   		}
	}

	__global__ void sumArraysCUDA(const int n, const int *const dev_a, const int *const dev_b, int *const dev_res)
	{
		int idx = threadIdx.x;
		dev_res[idx]=dev_a[idx]+dev_b[idx];
		printf("%i",idx);printf(" ");printf("%i",dev_res[idx]);printf(" ");printf("%i",dev_a[idx]);printf(" ");printf("%i",dev_b[idx]);printf("\n");
	}

    void studentJob(const int size, const int *const a, const int *const b, int *const res)
	{
		ChronoGPU chrGPU;

		// 3 arrays for GPU
		int *dev_a = NULL;
		int *dev_b = NULL;
		int *dev_res = NULL;

		// Allocate arrays on device (input and ouput)
		const size_t bytes = size * sizeof(int);
		std::cout 	<< "Allocating input (3 arrays): " 
					<< ( ( 3 * bytes ) >> 20 ) << " MB on Device" << std::endl;
		chrGPU.start();
		
		gpuErrchk(cudaMalloc((void**) &dev_a,bytes));
		gpuErrchk(cudaMalloc((void**) &dev_b,bytes));
		gpuErrchk(cudaMalloc((void**) &dev_res,bytes));
		
		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from host to device (input arrays) 
		gpuErrchk(cudaMemcpy(dev_a,a,bytes,cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(dev_b,b,bytes,cudaMemcpyHostToDevice));

		// Launch kernel
		sumArraysCUDA<<<1,256>>>(size,dev_a,dev_b,dev_res);
		
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

		// Copy data from device to host (output array)  
		gpuErrchk(cudaMemcpy(dev_res,res,bytes,cudaMemcpyDeviceToHost));

		/*for(int i = 0 ; i < size; i++){
			printf("%i",a[i]);printf(" ");
			printf("%i",b[i]);printf(" ");
			printf("%i",res[i]);
			printf("\n");
		}*/

		// Free arrays on device
		/*cudaFree(d_b);
		cudaFree(d_a);
		cudaFree(d_res);*/
	}
}

