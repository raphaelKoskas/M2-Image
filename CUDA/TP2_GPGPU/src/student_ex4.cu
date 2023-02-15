/*
* TP 2 - Convolution d'images
* --------------------------
* Mémoire constante et textures
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"



namespace IMAC
{
	__constant__ float gdev_matConv[15*15];
	texture <uchar4, 2, cudaReadModeElementType> dev_img_2D;
	__global__ void conv2DCUDA(	const uint imgWidth, const uint imgHeight,const uint matSize, uchar4* output){
		int idx = blockDim.x * blockIdx.x + threadIdx.x,
			idy = blockDim.y * blockIdx.y + threadIdx.y;
		if(idx >= imgWidth || idy >= imgHeight)
		{
			return;
		}
		float3 sum = make_float3(0.f,0.f,0.f);
		int  idOut = idy * imgWidth + idx,dX,dY;
		uint idMat, idPixel;
		uchar4 pixel;
		for (int j = 0; j < matSize; j++){
			dY = min(imgHeight-1,  max(0, (idy+j - (int) matSize / 2)  ));
			for (int i = 0; i < matSize; i++){
				dX = min(imgWidth-1,  max(0, (idx+i - (int) matSize / 2)  ));
				idMat		= j * matSize + i;
				idPixel	= dY * imgWidth + dX;
				pixel = tex2D(dev_img_2D, dX, dY);
				sum.x += (float)pixel.x * gdev_matConv[idMat];
				sum.y += (float)pixel.y * gdev_matConv[idMat];
				sum.z += (float)pixel.z * gdev_matConv[idMat];
			}
		}
		output[idOut].x = (uchar)min(255.f,  max(0.f, sum.x)  );
		output[idOut].y = (uchar)min(255.f,  max(0.f, sum.y)  );
		output[idOut].z = (uchar)min(255.f,  max(0.f, sum.z)  );
		output[idOut].w = 255;
			
	}

// ================================================== For image comparison
	std::ostream &operator <<(std::ostream &os, const uchar4 &c)
	{
		os << "[" << uint(c.x) << "," << uint(c.y) << "," << uint(c.z) << "," << uint(c.w) << "]";  
    	return os; 
	}

	void compareImages(const std::vector<uchar4> &a, const std::vector<uchar4> &b)
	{
		bool error = false;
		if (a.size() != b.size())
		{
			std::cout << "Size is different !" << std::endl;
			error = true;
		}
		else
		{
			for (uint i = 0; i < a.size(); ++i)
			{
				// Floating precision can cause small difference between host and device
				if (	std::abs(a[i].x - b[i].x) > 2 || std::abs(a[i].y - b[i].y) > 2 
					|| std::abs(a[i].z - b[i].z) > 2 || std::abs(a[i].w - b[i].w) > 2)
				{
					std::cout << "Error at index " << i << ": a = " << a[i] << " - b = " << b[i] << " - " << std::abs(a[i].x - b[i].x) << std::endl;
					error = true;
					break;
				}
			}
		}
		if (error)
		{
			std::cout << " -> You failed, retry!" << std::endl;
		}
		else
		{
			std::cout << " -> Well done!" << std::endl;
		}
	}
// ==================================================


    void studentJob(const std::vector<uchar4> &inputImg, // Input image
					const uint imgWidth, const uint imgHeight, // Image size
                    const std::vector<float> &matConv, // Convolution matrix (square)
					const uint matSize, // Matrix size (width or height)
					const std::vector<uchar4> &resultCPU, // Just for comparison
                    std::vector<uchar4> &output // Output image
					)
	{
		ChronoGPU chrGPU;

		// Pointers
		uchar4 	*dev_inputImg=NULL,
				*dev_output=NULL;

		float elapsedTime;

		const size_t bytes = inputImg.size()*sizeof(uchar4);
		const size_t matrix_bytes = matConv.size()*sizeof(float);

		unsigned int nbBlocksX = imgWidth / 32 + 1;
		unsigned int nbBlocksY = imgHeight / 32 + 1;
		unsigned int nbThreadsX = min(32,imgWidth);
		unsigned int nbThreadsY = min(32,imgHeight);

		//Exercice 4
		size_t pitch;

		chrGPU.start();
		cudaMallocPitch( &dev_inputImg, &pitch, imgWidth * sizeof(uchar4), imgHeight);
		cudaMalloc((void **) &dev_output,bytes);
		chrGPU.stop();
		std::cout 	<< "-> 2D Texture Version (Allocation time) : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		chrGPU.start();
		cudaMemcpy2D(dev_inputImg, pitch, inputImg.data(),  imgWidth * sizeof(uchar4),  imgWidth * sizeof(uchar4), imgHeight, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(gdev_matConv,matConv.data(),matrix_bytes,0,cudaMemcpyHostToDevice);
		cudaBindTexture2D(NULL, dev_img_2D, dev_inputImg, imgHeight, imgWidth,  pitch);
		chrGPU.stop();
		std::cout 	<< "-> 2D Texture Version (Data Transfer to GPU time) : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
		elapsedTime=chrGPU.elapsedTime();

		std::cout << "Convolution on GPU (" 	<< dim3(nbBlocksX,nbBlocksY).x << "x" << dim3(nbBlocksX,nbBlocksY).y << " blocks - " 
												<< dim3(nbThreadsX,nbThreadsY).x << "x" << dim3(nbThreadsX,nbThreadsY).y << " threads)" << std::endl;
		

		chrGPU.start();
		conv2DCUDA<<<dim3(nbBlocksX,nbBlocksY),dim3(nbThreadsX,nbThreadsY)>>>(imgWidth,imgHeight,matSize,dev_output);
		chrGPU.stop();
		std::cout 	<< "-> 2D Texture Version (Processing on GPU time) : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		chrGPU.start();
		cudaMemcpy(output.data(),dev_output,bytes,cudaMemcpyDeviceToHost);
		chrGPU.stop(); 
		std::cout 	<< "-> 2D Texture Version (Data Transfer from GPU time) : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
		std::cout 	<< "-> 2D Texture Version (Data Transfer total time) : " << chrGPU.elapsedTime()+elapsedTime << " ms" << std::endl << std::endl;
		compareImages(resultCPU, output);

		cudaFree(dev_inputImg);cudaFree(dev_output);

	}
}
