/*
* TP 3 - Réduction CUDA
* --------------------------
* Mémoire paratagée, synchronisation, optimisation
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"

namespace IMAC
{
	
	// ==================================================== EX 1
    __global__
    void maxReduce_ex1(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		extern __shared__ uint sharedMemory[];
		uint idx = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned int i=1;

		sharedMemory[threadIdx.x] = idx < size ? dev_array[idx] : 0;
		__syncthreads();

		while (2 * i * threadIdx.x +i  < blockDim.x){
			sharedMemory[2 * i * threadIdx.x ] = umax(sharedMemory[2 * i * threadIdx.x],sharedMemory[2 * i * threadIdx.x +i]);
			__syncthreads();
			i*=2;
		}
		if (threadIdx.x==0){
			dev_partialMax[blockIdx.x]=sharedMemory[0];
		}
	}
	__global__
    void maxReduce_ex2(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		extern __shared__ uint sharedMemory[];
		uint idx = blockDim.x * blockIdx.x + threadIdx.x;
		uint i= blockDim.x >> 1;
		bool keepGoing = threadIdx.x+1  <= i;
		sharedMemory[threadIdx.x] = idx < size ? dev_array[idx] : 0;
		__syncthreads();
		while (keepGoing){
			sharedMemory[threadIdx.x ] = umax(sharedMemory[threadIdx.x],sharedMemory[threadIdx.x + i]);
			i >>= 1;
			keepGoing = threadIdx.x+1  <= i;
			__syncthreads();
		}
		if (threadIdx.x==0){
			dev_partialMax[blockIdx.x]=sharedMemory[0];
		}
	}

	__global__
    void maxReduce_ex3(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		extern __shared__ uint sharedMemory[];
		uint idx = blockDim.x * (2*blockIdx.x) + threadIdx.x;
		uint i= blockDim.x >> 1;
		bool keepGoing = threadIdx.x+1  <= i;
		sharedMemory[threadIdx.x] = idx < size ? umax(dev_array[idx],dev_array[idx+blockDim.x]) : 0;
		__syncthreads();
		while (keepGoing){
			sharedMemory[threadIdx.x ] = umax(sharedMemory[threadIdx.x],sharedMemory[threadIdx.x + i]);
			i >>= 1;
			keepGoing = threadIdx.x+1  <= i;
			__syncthreads();
		}
		if (threadIdx.x==0){
			dev_partialMax[blockIdx.x]=sharedMemory[0];
		}
	}

	__global__
    void maxReduce_ex4(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		extern __shared__ uint sharedMemory[];
		uint idx = blockDim.x * (2*blockIdx.x) + threadIdx.x;
		uint i= blockDim.x >> 1;
		bool keepGoing = threadIdx.x+1  <= i;
		sharedMemory[threadIdx.x] = idx < size ? umax(dev_array[idx],dev_array[idx+blockDim.x]) : 0;
		__syncthreads();
		while (keepGoing){
			sharedMemory[threadIdx.x ] = umax(sharedMemory[threadIdx.x],sharedMemory[threadIdx.x + i]);
			i >>= 1;
			if (i<=32) break;
			keepGoing = threadIdx.x+1  <= i;
			__syncthreads();
		}
		if (threadIdx.x < 32){
			volatile uint *sharedMem = sharedMemory;
			if (blockDim.x >= 64) sharedMem[threadIdx.x] = umax(sharedMem[threadIdx.x],sharedMem[threadIdx.x + 32]);
			if (blockDim.x >= 32) sharedMem[threadIdx.x] = umax(sharedMem[threadIdx.x],sharedMem[threadIdx.x + 16]);
			if (blockDim.x >= 16) sharedMem[threadIdx.x] = umax(sharedMem[threadIdx.x],sharedMem[threadIdx.x + 8]);
			if (blockDim.x >= 8) sharedMem[threadIdx.x] = umax(sharedMem[threadIdx.x],sharedMem[threadIdx.x + 4]);
			if (blockDim.x >= 4) sharedMem[threadIdx.x] = umax(sharedMem[threadIdx.x],sharedMem[threadIdx.x + 2]);
			if (blockDim.x >= 2) sharedMem[threadIdx.x] = umax(sharedMem[threadIdx.x],sharedMem[threadIdx.x + 1]);
		}
		if (threadIdx.x==0){
			dev_partialMax[blockIdx.x]=sharedMemory[0];
		}
	}

	template <unsigned int N>
	__global__
    void maxReduce_ex5(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		extern __shared__ uint sharedMemory[];
		uint idx = blockDim.x * (2*blockIdx.x) + threadIdx.x;
		sharedMemory[threadIdx.x] = idx < size ? umax(dev_array[idx],dev_array[idx+blockDim.x]) : 0;
		__syncthreads();
		volatile uint *sharedMem = sharedMemory;
		if (blockDim.x >= 1024 && threadIdx.x < 512) sharedMem[threadIdx.x] = umax(sharedMem[threadIdx.x],sharedMem[threadIdx.x + 512]);
		if (blockDim.x >= 512 && threadIdx.x < 256) sharedMem[threadIdx.x] = umax(sharedMem[threadIdx.x],sharedMem[threadIdx.x + 256]);
		if (blockDim.x >= 256 && threadIdx.x < 128) sharedMem[threadIdx.x] = umax(sharedMem[threadIdx.x],sharedMem[threadIdx.x + 128]);
		if (blockDim.x >= 128 && threadIdx.x < 64) sharedMem[threadIdx.x] = umax(sharedMem[threadIdx.x],sharedMem[threadIdx.x + 64]);
		if (blockDim.x >= 64 && threadIdx.x < 32) sharedMem[threadIdx.x] = umax(sharedMem[threadIdx.x],sharedMem[threadIdx.x + 32]);
		if (blockDim.x >= 32 && threadIdx.x < 16) sharedMem[threadIdx.x] = umax(sharedMem[threadIdx.x],sharedMem[threadIdx.x + 16]);
		if (blockDim.x >= 16 && threadIdx.x < 8) sharedMem[threadIdx.x] = umax(sharedMem[threadIdx.x],sharedMem[threadIdx.x + 8]);
		if (blockDim.x >= 8 && threadIdx.x < 4) sharedMem[threadIdx.x] = umax(sharedMem[threadIdx.x],sharedMem[threadIdx.x + 4]);
		if (blockDim.x >= 4 && threadIdx.x < 2) sharedMem[threadIdx.x] = umax(sharedMem[threadIdx.x],sharedMem[threadIdx.x + 2]);
		if (blockDim.x >= 2 && threadIdx.x < 1) sharedMem[threadIdx.x] = umax(sharedMem[threadIdx.x],sharedMem[threadIdx.x + 1]);
		if (threadIdx.x==0){
			dev_partialMax[blockIdx.x]=sharedMemory[0];
		}
	}

	void studentJob(const std::vector<uint> &array, const uint resCPU /* Just for comparison */, const uint nbIterations)
    {
		uint *dev_array = NULL;
        const size_t bytes = array.size() * sizeof(uint);

		// Allocate array on GPU
		HANDLE_ERROR( cudaMalloc( (void**)&dev_array, bytes ) );
		// Copy data from host to device
		HANDLE_ERROR( cudaMemcpy( dev_array, array.data(), bytes, cudaMemcpyHostToDevice ) );

		std::cout << "Test with " << nbIterations << " iterations" << std::endl;

		std::cout << "========== Ex 1 " << std::endl;
		uint res1 = 0; // result
		// Launch reduction and get timing
		float2 timing1 = reduce<KERNEL_EX1>(nbIterations, dev_array, array.size(), res1);
		
        std::cout << " -> Done: ";
        printTiming(timing1);
		compare(res1, resCPU); // Compare results

		std::cout << "========== Ex 2 " << std::endl;
		uint res2 = 0; // result
		// Launch reduction and get timing
		float2 timing2 = reduce<KERNEL_EX2>(nbIterations, dev_array, array.size(), res2);
		
        std::cout << " -> Done: ";
        printTiming(timing2);
		compare(res2, resCPU);

		std::cout << "========== Ex 3 " << std::endl;
		uint res3 = 0; // result
		// Launch reduction and get timing
		float2 timing3 = reduce<KERNEL_EX3>(nbIterations, dev_array, array.size(), res3);
		
        std::cout << " -> Done: ";
        printTiming(timing3);
		compare(res3, resCPU);

		std::cout << "========== Ex 4 " << std::endl;
		uint res4 = 0; // result
		// Launch reduction and get timing
		float2 timing4 = reduce<KERNEL_EX4>(nbIterations, dev_array, array.size(), res4);
		
        std::cout << " -> Done: ";
        printTiming(timing4);
		compare(res4, resCPU);

		std::cout << "========== Ex 5 " << std::endl;
		uint res5 = 0; // result
		// Launch reduction and get timing
		float2 timing5 = reduce<KERNEL_EX5>(nbIterations, dev_array, array.size(), res5);
		
        std::cout << " -> Done: ";
        printTiming(timing5);
		compare(res5, resCPU);

		// Free array on GPU
		cudaFree( dev_array );
    }

	void printTiming(const float2 timing)
	{
		std::cout << ( timing.x < 1.f ? 1e3f * timing.x : timing.x ) << " us on device and ";
		std::cout << ( timing.y < 1.f ? 1e3f * timing.y : timing.y ) << " us on host." << std::endl;
	}

    void compare(const uint resGPU, const uint resCPU)
	{
		if (resGPU == resCPU)
		{
			std::cout << "Well done ! " << resGPU << " == " << resCPU << " !!!" << std::endl;
		}
		else
		{
			std::cout << "You failed ! " << resGPU << " != " << resCPU << " !!!" << std::endl;
		}
	}
}