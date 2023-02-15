/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 3: Filtre d'images sepia
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"

namespace IMAC
{
	__global__ void sepiaTransformCUDA(const uchar *const dev_input, uchar * dev_output, const int width){
		int idx = blockDim.x * blockIdx.x + threadIdx.x;
		int idy = blockDim.y * blockIdx.y + threadIdx.y;

		float checkGreen = .349f*dev_input[3*(idy*width+idx)]+.686f*dev_input[3*(idy*width+idx)+1]+.168f*dev_input[3*(idy*width+idx)+2]; //checks if green pixel is >= 255
		if (checkGreen>255){ // if green > 255 then red and green > 255
			dev_output[3*(idy*width+idx)+1]=255.f;
			dev_output[3*(idy*width+idx)]=255.f;
		}else{ 
			float checkRed = .393f*dev_input[3*(idy*width+idx)+0]+.769f*dev_input[3*(idy*width+idx)+1]+.189f*dev_input[3*(idy*width+idx)+2];
			dev_output[3*(idy*width+idx)+1]= checkGreen;
			dev_output[3*(idy*width+idx)]=min(255.f,checkRed); // if red> 255 clamp red to 255
		}
		dev_output[3*(idy*width+idx)+2]=.272f*dev_input[3*(idy*width+idx)+0]+.534f*dev_input[3*(idy*width+idx)+1]+.131f*dev_input[3*(idy*width+idx)+2];// Blue is always < 255 (.272+.534+.131 < 1)
		//if(idx==0 && idy==0)printf("Pixel %d %d : %d %d %d %d %d %d \n",idx,idy,dev_input[3*(idy*width+idx)],dev_input[3*(idy*width+idx)+1],dev_input[3*(idy*width+idx)+2],dev_output[3*(idy*width+idx)],dev_output[3*(idy*width+idx)+1],dev_output[3*(idy*width+idx)+2]);
	}


	void studentJob(const std::vector<uchar> &input, const uint width, const uint height, std::vector<uchar> &output)
	{
		ChronoGPU chrGPU;

		// 2 arrays for GPU
		uchar *dev_input = NULL;
		uchar *dev_output = NULL;
		float elapsedTime;
		
		const size_t bytes = input.size()*sizeof(uchar);

		chrGPU.start();
		cudaMalloc((void**) &dev_input,bytes); 
		cudaMalloc((void**) &dev_output,bytes);
		chrGPU.stop();
		std::cout 	<< "-> Done (Allocation time) : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		chrGPU.start();
		cudaMemcpy(dev_input,input.data(),bytes,cudaMemcpyHostToDevice);
		chrGPU.stop();
		std::cout 	<< "-> Done (Data Transfer to GPU time) : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
		elapsedTime=chrGPU.elapsedTime();

		int nbBlocksX = width / 32 +1;
		int nbBlocksY = height / 32 +1;
		int nbThreadsX = min(width,32);
		int nbThreadsY = min(height,32);

		std::cout << "Sepia filter on GPU (" 	<< dim3(nbBlocksX,nbBlocksY).x << "x" << dim3(nbBlocksX,nbBlocksY).y << " blocks - " 
												<< dim3(nbThreadsX,nbThreadsY).x << "x" << dim3(nbThreadsX,nbThreadsY).y << " threads)" << std::endl;
		
		chrGPU.start();
		sepiaTransformCUDA<<<dim3(nbBlocksX,nbBlocksY),dim3(nbThreadsX,nbThreadsY)>>>(dev_input,dev_output,width);
		chrGPU.stop();
		std::cout 	<< "-> Done (Processing on GPU time) : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		chrGPU.start();
		cudaMemcpy(output.data(),dev_output,bytes,cudaMemcpyDeviceToHost);
		chrGPU.stop(); 
		std::cout 	<< "-> Done (Data Transfer from GPU time) : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
		std::cout 	<< "-> Done (Data Transfer total time) : " << chrGPU.elapsedTime()+elapsedTime << " ms" << std::endl << std::endl;


		cudaFree(dev_input);cudaFree(dev_output);

	}
}