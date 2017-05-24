/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <curand_kernel.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
#include "Kernel.h"

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

// Trig constants:
#define PI (3.1416)

#define BETA            (2.0)
// Right angle constants:
#define THETA_R (10.0 / 180.0 * PI) // 5 degrees

// Sampling constants:
#define S_SIGMA_P (0.8)
#define S_SIGMA_T (15.0 / 90.0 * PI)

struct positionAndRotation
{
	double x;
	double y;
	double z;

	double rotX;
	double rotY;
	double rotZ;
	bool frozen;

	double length;
	double width;
};

struct targetRangeStruct {
	double targetRangeStart;
	double targetRangeEnd;
};

struct relationshipStruct
{
	targetRangeStruct TargetRange;
	int SourceIndex;
	int TargetIndex;
	double DegreesOfAtrraction;
};

struct Surface
{
	int nObjs;
	int nRelationships;

	// Weights
	float WeightFocalPoint;
	float WeightPairWise;
	float WeightVisualBalance;
	float WeightSymmetry;

	// Centroid
	double centroidX;
	double centroidY;

	// Focal point
	double focalX;
	double focalY;
	double focalRot;
};

struct gpuConfig
{
	int gridxDim;
	int gridyDim;
	int blockxDim;
	int blockyDim;
	int blockzDim;
	int iterations;
};

struct point
{
	float x, y, z, rotX, rotY, rotZ;
};

__global__ void initRNG(curandState *const rngStates, const unsigned int seed)
{
	// Determine thread ID
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	printf("tid: %d\n", tid);
	printf("seed: %d\n", seed);
	// Initialise the RNG
	curand_init(seed + tid, tid, 0, &rngStates[tid]);
}

__device__ double Distance(float xi, float yi, float xj, float yj) 
{
	double dX = xi - xj;
	double dY = yi - yj;
	return sqrt(dX * dX + dY * dY);
}

// Theta is the rotation
__device__ float phi(float xi, float yi, float xj, float yj, float tj)
{
	return atan2(yi - yj, xi - xj) - tj + PI / 2.0;
}

__device__ double VisualBalanceCosts(Surface *srf, positionAndRotation *cfg)
{
	float nx = 0;
	float ny = 0;
	float denom = 0;

	for (int i = 0; i < srf->nObjs; i++)
	{
		float area = cfg[i].length * cfg[i].width;
		nx += area * cfg[i].x;
		ny += area * cfg[i].y;
		denom += area;
	}

	// Distance between all summed areas and points divided by the areas and the room's centroid
	return Distance(nx / denom, ny / denom, srf->centroidX / 2, srf->centroidY / 2);
}

__device__ double PairWiseCosts(Surface *srf, positionAndRotation* cfg, relationshipStruct *rs)
{
	double result = 0;
	for (int i = 0; i < srf->nRelationships; i++)
	{
		// Look up source index from relationship and retrieve object using that index.
		double distance = Distance(cfg[rs[i].SourceIndex].x, cfg[rs[i].SourceIndex].y, cfg[rs[i].TargetIndex].x, cfg[rs[i].TargetIndex].y);
		printf("Distance: %f Range start: %f Range end: %f\n", distance, rs[i].TargetRange.targetRangeStart, rs[i].TargetRange.targetRangeEnd);
		if (distance < rs[i].TargetRange.targetRangeStart)
		{
			double fraction = distance / rs[i].TargetRange.targetRangeStart;
			result += (fraction * fraction);
		}
		else if (distance > rs[i].TargetRange.targetRangeEnd) 
		{
			double fraction = rs[i].TargetRange.targetRangeEnd / distance;
			result += (fraction * fraction);
		}
		else
		{
			result += 1;
		}
	}
	return result;
}

__device__ double FocalPointCosts(Surface *srf, positionAndRotation* cfg)
{
	double sum = 0;
	for (int i = 0; i < srf->nObjs; i++)
	{
		float phi_fi = phi(srf->focalX, srf->focalY, cfg[i].x, cfg[i].y, cfg[i].rotY);
		// Old implementation of grouping, all objects that belong to the category seat are used in the focal point calculation
		// For now we default to all objects, focal point grouping will come later
		//int s_i = s(r.c[i]);

		// sum += s_i * cos(phi_fi);
		sum += cos(phi_fi);
	}

	return sum;
}

__device__ float SymmetryCosts(Surface *srf, positionAndRotation* cfg)
{
	float sum = 0;
	for (int i = 0; i < srf->nObjs; i++)
	{
		float maxVal = 0;

		float ux = cos(srf->focalRot);
		float uy = sin(srf->focalRot);
		float s = 2 * (srf->focalX * ux + srf->focalY * uy - (cfg[i].x * ux + cfg[i].y * uy));  // s = 2 * (f * u - v * u)

		// r is the reflection of g across the symmetry axis defined by p.
		float rx_i = cfg[i].x + s * ux;
		float ry_i = cfg[i].y + s * uy;
		float rRot_i = 2 * srf->focalRot - cfg[i].rotY;
		if (rRot_i < -PI)
			rRot_i += 2 * PI;

		for (int j = 0; j < srf->nObjs; j++)
		{
			// Types should be the same, this probably works great with their limited amount of types but will probably not work that great for us. Perhaps define a group?
			int gamma_ij = 1;
			float dp = Distance(cfg[j].x, cfg[j].y, rx_i, ry_i);
			float dt = cfg[j].rotY - rRot_i;
			if (dt > PI)
				dt -= 2 * PI;

			float val = gamma_ij * (5 - sqrt(dp) - 0.4 * fabs(dt));
			maxVal = fmaxf(maxVal, val);
		}

		sum += maxVal;
	}

	return sum;
}

__device__ double Costs(Surface *srf, positionAndRotation* cfg, relationshipStruct *rs)
{
	double cost = 0;
	cost += srf->WeightPairWise * PairWiseCosts(srf, cfg, rs);
	cost += srf->WeightVisualBalance * VisualBalanceCosts(srf, cfg);
	cost += srf->WeightFocalPoint * FocalPointCosts(srf, cfg);
	cost += srf->WeightSymmetry * SymmetryCosts(srf, cfg);
	return cost;
}

__device__ int generateRandomIntInRange(curandState *rngStates, unsigned int tid, int max, int min)
{
	curandState localState = rngStates[tid];
	float p_rand = curand_uniform(&localState);
	rngStates[tid] = localState;
	p_rand *= (max - min + 0.999999);
	p_rand += min;
	return (int)truncf(p_rand);
}

__device__ void propose(Surface *srf, positionAndRotation *cfgStar, curandState *rngStates, unsigned int tid)
{
	int p = generateRandomIntInRange(rngStates, tid, 2, 0);
	printf("Selected mode: %d\n", p);
	// Translate location using normal distribution
	if (p == 0)
	{
		// randomly choose an object
		int obj = generateRandomIntInRange(rngStates, tid, srf->nObjs, 0);

		// Potential never ending loop when everything is frozen
		while (cfgStar[obj].frozen)
			obj = generateRandomIntInRange(rngStates, tid, srf->nObjs, 0);

		//printf("Selected object #: %d\n", obj);
		float dx = curand_normal(&rngStates[tid]);
		dx = dx * S_SIGMA_P;
		//printf("dx: %f\n", dx);
		float dy = curand_normal(&rngStates[tid]);
		dy = dy * S_SIGMA_P;
		//printf("dy: %f\n", dy);
		//printf("Before translation: %f\n", cfgStar[obj].x);
		//printf("Before translation: %f\n", cfgStar[obj].y);
		cfgStar[obj].x += dx;
		cfgStar[obj].y += dy;
		//printf("After translation: %f\n", cfgStar[obj].x);
		//printf("After translation: %f\n", cfgStar[obj].y);
	}
	// Translate rotation using normal distribution
	else if (p == 1)
	{
		int obj = generateRandomIntInRange(rngStates, tid, srf->nObjs, 0);
		while (cfgStar[obj].frozen)
			obj = generateRandomIntInRange(rngStates, tid, srf->nObjs, 0);
		// printf("Selected object #: %d\n", obj);
		float dRot = curand_normal(&rngStates[tid]);
		dRot = dRot * S_SIGMA_T;
		// printf("dRot: %f\n", dRot);
		// printf("before rotation: %f\n", cfgStar[obj].rotY);
		cfgStar[obj].rotY += dRot;
		// printf("After rotation: %f\n", cfgStar[obj].rotY);
		
		if (cfgStar[obj].rotY < 0)
			cfgStar[obj].rotY += 2 * PI;
		else if (cfgStar[obj].rotY > 2 * PI)
			cfgStar[obj].rotY -= 2 * PI;
	}
	// Swap two objects for both location and rotation
	else
	{
		if (srf->nObjs < 2) {
			return;
		}
		// This can result in the same object, chance becomes increasingly smaller given more objects
		int obj1 = generateRandomIntInRange(rngStates, tid, srf->nObjs, 0);
		while (cfgStar[obj1].frozen)
			obj1 = generateRandomIntInRange(rngStates, tid, srf->nObjs, 0);

		int obj2 = generateRandomIntInRange(rngStates, tid, srf->nObjs, 0);
		while (cfgStar[obj2].frozen)
			obj2 = generateRandomIntInRange(rngStates, tid, srf->nObjs, 0);
		// printf("Selected object #: %d\n", obj1);
		// printf("Selected object #: %d\n", obj2);

		// Temporarily store cfgStar[obj1] values
		float x = cfgStar[obj1].x;
		float y = cfgStar[obj1].y;
		float z = cfgStar[obj1].z;
		float rotX = cfgStar[obj1].rotX;
		float rotY = cfgStar[obj1].rotY;
		float rotZ = cfgStar[obj1].rotZ;

		// Move values of obj2 to obj1
		cfgStar[obj1].x = cfgStar[obj2].x;
		cfgStar[obj1].y = cfgStar[obj2].y;
		cfgStar[obj1].z = cfgStar[obj2].z;
		cfgStar[obj1].rotX = cfgStar[obj2].rotX;
		cfgStar[obj1].rotY = cfgStar[obj2].rotY;
		cfgStar[obj1].rotZ = cfgStar[obj2].rotZ;

		// Move stored values of obj1 to obj2
		cfgStar[obj2].x = x;
		cfgStar[obj2].y = y;
		cfgStar[obj2].z = z;
		cfgStar[obj2].rotX = rotX;
		cfgStar[obj2].rotY = rotY;
		cfgStar[obj2].rotZ = rotZ;
	}
}

__device__ bool Accept(double costStar, double costCur, curandState *rngStates, unsigned int tid)
{
	printf("(costStar - costCur):  %f\n", (costStar - costCur));
	printf("(float) exp(-BETA * (costStar - costCur)): %f\n", (float)exp(-BETA * (costStar - costCur)));
	float randomNumber = curand_uniform(&rngStates[tid]);
	printf("Random number: %f\n", randomNumber);
	return  randomNumber < fminf(1.0f, (float) exp(-BETA * (costStar - costCur)));
}

// result is a [,] array with 1 dimension equal to the amount of blocks used and the other dimension equal to the amount of objects
// rs is an array with the length equal to the amount of relationships
// cfg is an array with the length equal to the amount of objects
// Surface is a basic struct
__global__ void Kernel(point *p, relationshipStruct *rs, positionAndRotation* cfg, Surface *srf, gpuConfig *gpuCfg, curandState *rngStates)
{
//    printf("current block [%d, %d]:\n",\
//            blockIdx.y*gridDim.x+blockIdx.x,\
//            threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x);

	// Calculate current tid
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// Retrieve local state from rng states
	//printf("test random number 1: %f\n", curand_uniform(&rngStates[tid]));
	
	// Initialize current configuration
	positionAndRotation* cfgCurrent = (positionAndRotation*) malloc(srf->nObjs * sizeof(positionAndRotation));
	for (int i = 0; i < srf->nObjs; i++)
	{
		cfgCurrent[i] = cfg[i];
	}

	double costCurrent = Costs(srf, cfgCurrent, rs);
	
	positionAndRotation* cfgBest = (positionAndRotation*)malloc(srf->nObjs * sizeof(positionAndRotation));
	for (int i = 0; i < srf->nObjs; i++)
	{
		cfgBest[i] = cfgCurrent[i];
	}
	double costBest = costCurrent;

	for (int i = 0; i < gpuCfg->iterations; i++)
	{
		// Create cfg Star and initialize it to cfgcurrent that will have a proposition done to it.
		positionAndRotation* cfgStar = (positionAndRotation*)malloc(srf->nObjs * sizeof(positionAndRotation));
		for (int j = 0; j < srf->nObjs; j++)
		{
			cfgStar[j] = cfgCurrent[j];
		}
		// cfgStar contains an array with translated objects
		propose(srf, cfgStar, rngStates, tid);
		double costStar = Costs(srf, cfgStar, rs);
		printf("Cost star configuration: %f\n", costStar);
		printf("Cost best configuration: %f\n", costBest);
		if (costStar < costBest)
		{
			printf("New best %f\n", costBest);

			// Copy star into best for storage
			for (int j = 0; j < srf->nObjs; j++)
			{
				cfgBest[j] = cfgStar[j];
			}
			costBest = costStar;
		}

		if (Accept(costStar, costCurrent, rngStates, tid))
		{
			// Possible different approach: Set pointer of current to star, free up memory used by current? reinitialize star?
			printf("Star accepted as new current.\n");
			// Copy star into current
			for (int j = 0; j < srf->nObjs; j++)
			{
				printf("Old current of result jndex %d. X: %f Y: %f Z: %f rotX: %f rotY: %f rotZ: %f\n", j, cfgCurrent[j].x, cfgCurrent[j].y, cfgCurrent[j].z, cfgCurrent[j].rotX, cfgCurrent[j].rotY, cfgCurrent[j].rotZ);
				cfgCurrent[j] = cfgStar[j];
				printf("Star values of result jndex %d. X: %f Y: %f Z: %f rotX: %f rotY: %f rotZ: %f\n", j, cfgStar[j].x, cfgStar[j].y, cfgStar[j].z, cfgStar[j].rotX, cfgStar[j].rotY, cfgStar[j].rotZ);
				printf("New current of result jndex %d. X: %f Y: %f Z: %f rotX: %f rotY: %f rotZ: %f\n", j, cfgCurrent[j].x, cfgCurrent[j].y, cfgCurrent[j].z, cfgCurrent[j].rotX, cfgCurrent[j].rotY, cfgCurrent[j].rotZ);
				
			}
			costCurrent = costStar;
		}
		free(cfgStar);
	}

	__syncthreads();
	
	// Copy best config (now set to input config) to result of this block
	for (unsigned int i = 0; i < srf->nObjs; i++)
	{
		// BlockId counts from 0, so to properly multiply
		int index = blockIdx.x * srf->nObjs + i;
		p[index].x = cfgBest[i].x;
		p[index].y = cfgBest[i].y;
		p[index].z = cfgBest[i].z;
		p[index].rotX = cfgBest[i].rotX;
		p[index].rotY = cfgBest[i].rotY;
		p[index].rotZ = cfgBest[i].rotZ;

		// Print out best
	}

	free(cfgCurrent);
	free(cfgBest);
}

extern "C" __declspec(dllexport) point* KernelWrapper(relationshipStruct *rss, positionAndRotation* cfg, Surface *srf, gpuConfig *gpuCfg)
{
	// Create pointer for on gpu
	// Determine memory size of object to transfer
	// Malloc on GPU size
	// Cpy memory from cpu to gpu
	relationshipStruct *gpuRS;
	int rsSize = sizeof(relationshipStruct) * srf->nRelationships;
	checkCudaErrors(cudaMalloc(&gpuRS, rsSize));
	checkCudaErrors(cudaMemcpy(gpuRS, rss, rsSize, cudaMemcpyHostToDevice));

	positionAndRotation *gpuAlgoCFG;
	int algoCFGSize = sizeof(positionAndRotation) * srf->nObjs;
	checkCudaErrors(cudaMalloc(&gpuAlgoCFG, algoCFGSize));
	checkCudaErrors(cudaMemcpy(gpuAlgoCFG, cfg, algoCFGSize, cudaMemcpyHostToDevice));

	Surface *gpuSRF;
	int srfSize = sizeof(Surface);
	checkCudaErrors(cudaMalloc(&gpuSRF, srfSize));
	checkCudaErrors(cudaMemcpy(gpuSRF, srf, srfSize, cudaMemcpyHostToDevice));

	gpuConfig *gpuGpuConfig;
	int gpuCFGSize = sizeof(gpuConfig);
	checkCudaErrors(cudaMalloc(&gpuGpuConfig, gpuCFGSize));
	checkCudaErrors(cudaMemcpy(gpuGpuConfig, gpuCfg, gpuCFGSize, cudaMemcpyHostToDevice));

	point *gpuPointArray;
	int pointArraySize = srf->nObjs * sizeof(point) * gpuCfg->gridxDim;
	point *outPointArray = (point *) malloc(pointArraySize);
	checkCudaErrors(cudaMalloc((void**)&gpuPointArray, pointArraySize));
	// cudaMemcpy(gpuPointArray, result, pointArraySize, cudaMemcpyHostToDevice);

	// Setup GPU random generator
	curandState *d_rngStates = 0;
	checkCudaErrors(cudaMalloc((void **)&d_rngStates, gpuCfg->gridxDim * gpuCfg->blockxDim * sizeof(curandState)));

	// Initialise random number generator
	initRNG <<<gpuCfg->gridxDim, gpuCfg->blockxDim >> > (d_rngStates, time(NULL));

	// Commented for possible later usage
	// dim3 dimGrid(gpuCfg->gridxDim, gpuCfg->gridyDim);
	// dim3 dimBlock(gpuCfg->blockxDim, gpuCfg->blockyDim, gpuCfg->blockzDim);
	
	// Block 1 dimensional, amount of threads available, configurable
	// Grid 1 dimension, amount of suggestions to be made.
	Kernel <<<gpuCfg->gridxDim, gpuCfg->blockxDim >>>(gpuPointArray, gpuRS, gpuAlgoCFG, gpuSRF, gpuGpuConfig, d_rngStates);
	checkCudaErrors(cudaDeviceSynchronize());
	if (cudaSuccess != cudaGetLastError()) {
		fprintf(stderr, "cudaSafeCall() failed : %s\n",
			cudaGetErrorString(cudaGetLastError()));
	}

	// copy back results from gpu to cpu
	checkCudaErrors(cudaMemcpy(outPointArray, gpuPointArray, pointArraySize, cudaMemcpyDeviceToHost));

	// Free all allocated GPU memory
	cudaFree(gpuRS);
	cudaFree(gpuAlgoCFG);
	cudaFree(gpuSRF);
	cudaFree(gpuGpuConfig);
	cudaFree(gpuPointArray);

	return outPointArray;
}

void basicCudaDeviceInformation(int argc, char **argv) {
	int devID;
	cudaDeviceProp props;

	// This will pick the best possible CUDA capable device
	devID = findCudaDevice(argc, (const char **)argv);

	//Get GPU information
	checkCudaErrors(cudaGetDevice(&devID));
	checkCudaErrors(cudaGetDeviceProperties(&props, devID));
	printf("Device %d: \"%s\" with Compute %d.%d capability\n",
		devID, props.name, props.major, props.minor);

	printf("printf() is called. Output:\n\n");
}


int main(int argc, char **argv)
{
	basicCudaDeviceInformation(argc, argv);

	const int N = 2;
	const int NRel = 1;
	Surface srf;
	srf.nObjs = N;
	srf.nRelationships = NRel;
	srf.WeightFocalPoint = -2.0f;
	srf.WeightPairWise = -2.0f;
	srf.WeightVisualBalance = 1.5f;
	srf.WeightSymmetry = -2.0;
	srf.centroidX = 0.0;
	srf.centroidY = 0.0;
	srf.focalX = 5.0;
	srf.focalY = 5.0;
	srf.focalRot = 0.0;

	positionAndRotation cfg[N];
	for (int i = 0; i < N; i++) {
		cfg[i].x = 2.0;
		cfg[i].y = 3.0;
		cfg[i].z = 4.0;
		cfg[i].rotX = 5.0;
		cfg[i].rotY = 6.0;
		cfg[i].rotZ = 7.0;
		cfg[i].frozen = false;
		cfg[i].length = 1.0;
		cfg[i].width = 1.0;
	}

	// Create relationship
	relationshipStruct rss[1];
	rss[0].TargetRange.targetRangeStart = 2.0;
	rss[0].TargetRange.targetRangeEnd = 4.0;
	rss[0].DegreesOfAtrraction = 2.0;
	rss[0].SourceIndex = 0;
	rss[0].TargetIndex = 1;

	//for (int i = 0; i < NRel; i++) {
	//	rss[i].TargetRange.targetRangeStart = 0.0;
	//	rss[i].TargetRange.targetRangeEnd = 2.0;
	//	rss[i].Source.x = 0.0 + i;
	//	rss[i].Source.y = 0.0 + i;
	//	rss[i].Source.z = 0.0;
	//	rss[i].Source.rotX = 1.0;
	//	rss[i].Source.rotY = 1.0;
	//	rss[i].Source.rotZ = 1.0;
	//	rss[i].Target.x = 3.0 + i;
	//	rss[i].Target.y = 3.0 + i;
	//	rss[i].Target.z = 0.0;
	//	rss[i].Target.rotX = 1.0;
	//	rss[i].Target.rotY = 1.0;
	//	rss[i].Target.rotZ = 1.0;
	//	rss[i].DegreesOfAtrraction = 2.0;
	//}

	gpuConfig gpuCfg;

	gpuCfg.gridxDim = 1;
	gpuCfg.gridyDim = 0;
	gpuCfg.blockxDim = 1;
	gpuCfg.blockyDim = 0;
	gpuCfg.blockzDim = 0;
	gpuCfg.iterations = 1000;

	// Point test code:

	point *result = KernelWrapper(rss, cfg, &srf, &gpuCfg);

	for (int i = 0; i < srf.nObjs * gpuCfg.gridxDim; i++)
	{
		printf("Values of result index %d. X: %f Y: %f Z: %f rotX: %f rotY: %f rotZ: %f\n", i, result[i].x, result[i].y, result[i].z, result[i].rotX, result[i].rotY, result[i].rotZ);
	}

 	return EXIT_SUCCESS;
}