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
	__global__ void sumArraysCUDA(const int n, const int *const dev_a, const int *const dev_b, int *const dev_res)
	{
		int idx = blockDim.x * blockIdx.x + threadIdx.x;
		if (n>idx){
		dev_res[idx]=dev_a[idx]+dev_b[idx];}
		//printf("%d %d %d %d\n",idx,dev_res[idx],dev_a[idx],dev_b[idx]);
	}

    void studentJob(const int size, const int *const a, const int *const b, int *const res)
	{
		ChronoGPU chrGPU;

		// 3 arrays for GPU
		int *dev_a = NULL;
		int *dev_b = NULL;
		int *dev_res = NULL;
		float elapsedTime;

		// Allocate arrays on device (input and ouput)
		const size_t bytes = size * sizeof(int);
		std::cout 	<< "Allocating input (3 arrays): " 
					<< ( ( 3 * bytes ) >> 20 ) << " MB on Device" << std::endl;
		chrGPU.start();
		
		cudaMalloc((void**) &dev_a,bytes);
		cudaMalloc((void**) &dev_b,bytes);
		cudaMalloc((void**) &dev_res,bytes);
		
		chrGPU.stop();
		std::cout 	<< "-> Done (Allocation time) : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
		
		// Copy data from host to device (input arrays)
		chrGPU.start();
		cudaMemcpy(dev_a,a,bytes,cudaMemcpyHostToDevice);
		cudaMemcpy(dev_b,b,bytes,cudaMemcpyHostToDevice);
		chrGPU.stop();
		std::cout 	<< "-> Done (Data Transfer to GPU time) : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
		elapsedTime=chrGPU.elapsedTime();
		int nbBlocks = (int) size / 1024 +1;
		int nbThreads = min (size,1024);

		// Launch kernel
		chrGPU.start();
		sumArraysCUDA<<<nbBlocks,nbThreads>>>(size,dev_a,dev_b,dev_res);
		chrGPU.stop();
		std::cout 	<< "-> Done (Processing on GPU time) : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
		// Copy data from device to host (output array)  
		chrGPU.start();
		cudaMemcpy(res,dev_res,bytes,cudaMemcpyDeviceToHost);
		chrGPU.stop(); 
		std::cout 	<< "-> Done (Data Transfer from GPU time) : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
		std::cout 	<< "-> Done (Data Transfer total time) : " << chrGPU.elapsedTime()+elapsedTime << " ms" << std::endl << std::endl;


		// Free arrays on device
		cudaFree(dev_b);
		cudaFree(dev_a);
		cudaFree(dev_res);

		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

	}
}

