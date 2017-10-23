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

//Working group code
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
#include "Kernel.h"

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

// Trig constants:
#define PI (3.1416)

//#define BETA (0.1)
// Right angle constants:
#define THETA_R (10.0 / 180.0 * PI) // 5 degrees

// Sampling constants:
#define S_SIGMA_P (0.8)
#define S_SIGMA_T (15.0 / 90.0 * PI)

//In the original implementation, number of threads in a group was set to the WARP size, which we can do with 32
#define WARP_SIZE 32


struct vertex
{
	double x;
	double y;
	double z;
};

struct rectangle
{
	int point1Index;
	int point2Index;
	int point3Index;
	int point4Index;
	int SourceIndex;
};

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
	targetRangeStruct AngleRange;
	int SourceIndex;
	int TargetIndex;
	double DegreesOfAtrraction;
};

struct Surface
{
	int nObjs;
	int nRelationships;
	int nClearances;

	// Weights
	float WeightFocalPoint;
	float WeightPairWise;
	float WeightVisualBalance;
	float WeightSymmetry;
	float WeightOffLimits;
	float WeightClearance;
	float WeightSurfaceArea;

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
	bool frozen;

	double length;
	double width;
};

struct resultCosts
{
	float totalCosts;
	float PairWiseCosts;
	float VisualBalanceCosts;
	float FocalPointCosts;
	float SymmetryCosts;
	float ClearanceCosts;
	float OffLimitsCosts;
	float SurfaceAreaCosts;
};

struct result {
	point *points;
	resultCosts costs;
};

__global__ void initRNG(curandState *const rngStates, const unsigned int seed)
{
	// Determine thread ID
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("tid: %d\n", tid);
	//printf("seed: %d\n", seed);
	// Initialise the RNG
	curand_init(seed + tid, tid, 0, &rngStates[tid]);
}

__device__ double Distance(float xi, float yi, float xj, float yj) 
{
	double dX = xi - xj;
	double dY = yi - yj;
	return sqrt(dX * dX + dY * dY);
}

//Determines the angular difference between two objects where i is oriented to j (i is bearing to j)
__device__ double theta(float xi, float yi, float xj, float yj, float ti) {
	double dX = xi - xj;
	double dY = yi - yj;
	double theta_p = atan2(dY, dX); //gives us the angle between -PI and PI

									//and now between 0 and 2*pi
	theta_p = (theta_p < 0) ? 2 * PI + theta_p : theta_p;
	//printf("theta_p=%f,ti=%f\n",theta_p,ti);
	//return the re-oriented angle
	double theta = theta_p - ti;
	return (theta < 0) ? 2 * PI + theta : theta;

}

// Tj is the rotation
__device__ float phi(float xi, float yi, float xj, float yj, float tj)
{
	return atan2(yi - yj, xi - xj) - tj + PI / 2.0;
}

//To get coop groups working, we need to remove the useage of shared memory. We can do this using a shuffle/reduce
//taken from https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
template<int tile_sz>
 __device__ float reduce(cg::thread_block_tile<tile_sz> group, float val) { //Group size is static, so no need to include that
	for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
		val += group.shfl_down(val, offset);
	}
	return val;
}

//Modest reduction algorithm with a lot of room to improve upon
/*__device__ void reduce(cg::thread_group group,float *values, int n) { //Size of the array (from how we use it, it's at most of size blockDim.x)
	int stride = group.size()/2;
	int tid = group.thread_rank();
	int size = n;
	//We make the very important for parallel reduction assumptions that blockDim is a power of two and values is a multiple of blockdim
	//We can do this because we control those
	while (size > 1) {
		for (int i = tid + stride; i < size; i += stride) {
			values[tid] += values[i];
			//printf("tid = %d with value %f\n", tid, values);
		}
		size = size / 2;
		stride = stride / 2;
		group.sync();
		//local variable per thread, so no race condition
	}
}*/
template<int tile_sz>
__device__ double VisualBalanceCosts(cg::thread_block_tile<tile_sz> group, Surface *srf, positionAndRotation *cfg)
{
	int tid =  group.thread_rank();
	int step = group.size();
	float nx;// = 0;
	float ny;// = 0;
	float denom;// = 0;
	//because of multiple share blocks, we do an atomic add instead of the reduce method
	nx = 0.0;
	ny = 0.0;
	denom = 0.0;
	for (int i = tid; i < srf->nObjs; i+=step)
	{
		float area = cfg[i].length * cfg[i].width;
		nx += area * cfg[i].x;
		ny += area * cfg[i].y;
		denom += area;
	}
	group.sync();
	reduce<tile_sz>(group, nx);
	reduce<tile_sz>(group, ny);
	reduce<tile_sz>(group, denom);
	// Distance between all summed areas and points divided by the areas and the room's centroid
	return  Distance(nx / denom, ny / denom, srf->centroidX / 2, srf->centroidY / 2); //Because we are all reducing, all values should be the same
}

template<int tile_sz>
__device__ double PairWiseCosts(cg::thread_block_tile<tile_sz> group,Surface *srf, positionAndRotation* cfg, relationshipStruct *rs)
{
	int tid = group.thread_rank();
	int step = group.size();
	float values;
	values = 0.0f; //Since it's size blockDim, we can have each of them treat it as the starting value
	for (int i = 0; i < srf->nRelationships; i+=step)
	{
		// Look up source index from relationship and retrieve object using that index.
		double distance = Distance(cfg[rs[i].SourceIndex].x, cfg[rs[i].SourceIndex].y, cfg[rs[i].TargetIndex].x, cfg[rs[i].TargetIndex].y);
		//printf("Distance: %f Range start: %f Range end: %f\n", distance, rs[i].TargetRange.targetRangeStart, rs[i].TargetRange.targetRangeEnd);
		//penalize if we are too close
		if (distance < rs[i].TargetRange.targetRangeStart)
		{
			double fraction = distance / rs[i].TargetRange.targetRangeStart;
			values -= (fraction * fraction);
		}
		//penalize if we are too far
		else if (distance > rs[i].TargetRange.targetRangeEnd)
		{
			double fraction = rs[i].TargetRange.targetRangeEnd / distance;
			values -= (fraction * fraction);
		}
		else {
			values -= 1;
		}
		// Else don't do anything as 0 indicates a perfect solution
	}
	group.sync();
	reduce<tile_sz>(group, values);
	//printf("Clearance costs error: %f\n", error);
	return (double)values;
}

//This functional principle uses a lookup (relationshipStruct) to determine weights from a recommended angle
//This is not the facing angle but the distance angle (so, the target rotated around the source)
template<int tile_sz>
__device__ double PairWiseTotalCosts(cg::thread_block_tile<tile_sz> group, Surface *srf, positionAndRotation* cfg, relationshipStruct *rs)
{
	int tid = group.thread_rank();
	int step = group.size();
	float values;
	values = 0.0f; //Since it's size blockDim, we can have each of them treat it as the starting value
						//assuming (0,2*PI]
	for (int i = tid; i < srf->nRelationships; i += step)
	{
		// We use phi to calculate the angle between the rotation of the object and the target object
		double distance = Distance(cfg[rs[i].SourceIndex].x, cfg[rs[i].SourceIndex].y, cfg[rs[i].TargetIndex].x, cfg[rs[i].TargetIndex].y);
		double angle = theta(cfg[rs[i].SourceIndex].x, cfg[rs[i].SourceIndex].y, cfg[rs[i].TargetIndex].x, cfg[rs[i].TargetIndex].y, cfg[rs[i].TargetIndex].rotY);
		
		//Score distance calculation
		double score = (distance < rs[i].TargetRange.targetRangeStart) ? powf(distance / rs[i].TargetRange.targetRangeStart, rs[i].DegreesOfAtrraction) : 1.0;
		score        = (distance > rs[i].TargetRange.targetRangeEnd)   ? powf(rs[i].TargetRange.targetRangeEnd / distance  , rs[i].DegreesOfAtrraction)  : 1.0;

		//For now, we assume start is greater than end
		double norm    = (rs[i].TargetRange.targetRangeStart < rs[i].TargetRange.targetRangeEnd)? rs[i].AngleRange.targetRangeEnd - rs[i].AngleRange.targetRangeStart : 
																								  rs[i].AngleRange.targetRangeStart - rs[i].AngleRange.targetRangeEnd; //The max distance away is half the slice that is in the no zone 
		norm = (2.0 * PI - norm) / 2.0;
		double a_score = (rs[i].AngleRange.targetRangeEnd < angle || angle < rs[i].AngleRange.targetRangeEnd) ? fmin(fabs(distance - rs[i].AngleRange.targetRangeStart), 
																													 fabs(distance - rs[i].AngleRange.targetRangeEnd)) / norm : 1.0;
		values -= score*a_score; //So, best score we can do is -1, and everything else degrades from there
	}
	group.sync();
	reduce<tile_sz>(group, values);
	//printf("Clearance costs error: %f\n", error);
	return values;
}

template<int tile_sz>
__device__ double FocalPointCosts(cg::thread_block_tile<tile_sz> group, Surface *srf, positionAndRotation* cfg)
{
	int tid = group.thread_rank();
	int step = group.size();
	float values;
	values = 0.0f; //Since it's size blockDim, we can have each of them treat it as the starting value
	for (int i = tid; i < srf->nObjs; i += step)
	{
		float phi_fi = phi(srf->focalX, srf->focalY, cfg[i].x, cfg[i].y, cfg[i].rotY);
		// Old implementation of grouping, all objects that belong to the seat category are used in the focal point calculation
		// For now we default to all objects, focal point grouping will come later
		//int s_i = s(r.c[i]);

		// sum += s_i * cos(phi_fi);
		values -= cos(phi_fi);
	}
	group.sync();
	reduce<tile_sz>(group, values);
	//printf("tid = %d, value = %f\n", tid, values[tid]);
	//printf("Clearance costs error: %f\n", error);
	return (double)values;
}

template<int tile_sz>
__device__ float SymmetryCosts(cg::thread_block_tile<tile_sz> group, Surface *srf, positionAndRotation* cfg)
{
	int tid = group.thread_rank();
	int step = group.size();
	float values;
	values = 0.0f; //Since it's size blockDim, we can have each of them treat it as the starting value
	for (int i = tid; i < srf->nObjs; i += step)
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

		values -= maxVal;
	}

	group.sync();
	reduce<tile_sz>(group, values);
	//printf("Clearance costs error: %f\n", error);
	return values;
}


__device__ float calculateIntersectionArea(vertex rect1Min, vertex rect1Max, vertex rect2Min, vertex rect2Max) {
	// printf("Clearance rectangle 1: Min X: %f Y: %f Max X: %f Y: %f\n", rect1Min.x, rect1Min.y, rect1Max.x, rect1Max.y);
	// printf("Clearance rectangle 2: Min X: %f Y: %f Max X: %f Y: %f\n", rect2Min.x, rect2Min.y, rect2Max.x, rect2Max.y);
	// for each two rectangles, find out their intersection. Increase the error using the area
	float x5 = fmaxf(rect1Min.x, rect2Min.x);
	float y5 = fmaxf(rect1Min.y, rect2Min.y);
	float x6 = fminf(rect1Max.x, rect2Max.x);
	float y6 = fminf(rect1Max.y, rect2Max.y);

	// Check if proper rectangle, if so it is an intersection.
	if (x5 >= x6 || y5 >= y6)
		return 0.0f;

	// printf("Intersection rectangle: Min X: %f Y: %f Max X: %f Y: %f\n", x5, y5, x6, y6);

	// Calculate area and add to error
	float area = (x6 - x5) * (y6 - y5);
	// printf("Area intersection rectangle: %f\n", area);
	return area;
}

__device__ void createComplementRectangle(vertex srfRectMin, vertex srfRectMax, vertex *complementRectangle1, vertex *complementRectangle2, vertex *complementRectangle3, vertex *complementRectangle4) {
	// 0 is min value, 1 is max value
	complementRectangle1[0].x = -DBL_MAX;
	complementRectangle1[0].y = -DBL_MAX;
	complementRectangle1[1].x = DBL_MAX;
	complementRectangle1[1].y = srfRectMin.y;

	complementRectangle2[0].x = -DBL_MAX;
	complementRectangle2[0].y = srfRectMin.y;
	complementRectangle2[1].x = srfRectMin.x;
	complementRectangle2[1].y = srfRectMax.y;

	complementRectangle3[0].x = -DBL_MAX;
	complementRectangle3[0].y = srfRectMax.y;
	complementRectangle3[1].x = DBL_MAX;
	complementRectangle3[1].y = DBL_MAX;

	complementRectangle4[0].x = srfRectMax.x;
	complementRectangle4[0].y = srfRectMin.y;
	complementRectangle4[1].x = DBL_MAX;
	complementRectangle4[1].y = srfRectMax.y;
}

__device__ vertex minValue(vertex *vertices, int startIndexVertices, float xtranslation, float ytranslation) {
	vertex rect1;
	rect1.x = DBL_MAX;
	rect1.y = DBL_MAX;
	rect1.z = 0;
	rect1.x = (rect1.x > vertices[startIndexVertices].x + xtranslation) ? vertices[startIndexVertices].x : rect1.x;
	rect1.x = (rect1.x > vertices[startIndexVertices + 1].x + xtranslation) ? vertices[startIndexVertices + 1].x + xtranslation : rect1.x;
	rect1.x = (rect1.x > vertices[startIndexVertices + 2].x + xtranslation) ? vertices[startIndexVertices + 2].x + xtranslation : rect1.x;
	rect1.x = (rect1.x > vertices[startIndexVertices + 3].x + xtranslation) ? vertices[startIndexVertices + 3].x + xtranslation : rect1.x;

	rect1.y = (rect1.y > vertices[startIndexVertices].y + ytranslation) ? vertices[startIndexVertices].y + ytranslation : rect1.y;
	rect1.y = (rect1.y > vertices[startIndexVertices + 1].y + ytranslation) ? vertices[startIndexVertices + 1].y + ytranslation : rect1.y;
	rect1.y = (rect1.y > vertices[startIndexVertices + 2].y + ytranslation) ? vertices[startIndexVertices + 2].y + ytranslation : rect1.y;
	rect1.y = (rect1.y > vertices[startIndexVertices + 3].y + ytranslation) ? vertices[startIndexVertices + 3].y + ytranslation : rect1.y;
	//printf("Min value vector after translation: X: %f Y: %f\n", rect1.x, rect1.y);
	return rect1;
}

__device__ vertex maxValue(vertex *vertices, int startIndexVertices, float xtranslation, float ytranslation) {
	vertex rect1;
	rect1.x = -DBL_MAX;
	rect1.y = -DBL_MAX;
	rect1.z = 0;

	rect1.x = (rect1.x < vertices[startIndexVertices].x + xtranslation) ? vertices[startIndexVertices].x + xtranslation : rect1.x;
	rect1.x = (rect1.x < vertices[startIndexVertices + 1].x + xtranslation) ? vertices[startIndexVertices + 1].x + xtranslation : rect1.x;
	rect1.x = (rect1.x < vertices[startIndexVertices + 2].x + xtranslation) ? vertices[startIndexVertices + 2].x + xtranslation : rect1.x;
	rect1.x = (rect1.x < vertices[startIndexVertices + 3].x + xtranslation) ? vertices[startIndexVertices + 3].x + xtranslation : rect1.x;

	rect1.y = (rect1.y < vertices[startIndexVertices].y + ytranslation) ? vertices[startIndexVertices].y + ytranslation : rect1.y;
	rect1.y = (rect1.y < vertices[startIndexVertices + 1].y + ytranslation) ? vertices[startIndexVertices + 1].y + ytranslation : rect1.y;
	rect1.y = (rect1.y < vertices[startIndexVertices + 2].y + ytranslation) ? vertices[startIndexVertices + 2].y + ytranslation : rect1.y;
	rect1.y = (rect1.y < vertices[startIndexVertices + 3].y + ytranslation) ? vertices[startIndexVertices + 3].y + ytranslation : rect1.y;
	//printf("Max value vector after translation: X: %f Y: %f\n", rect1.x, rect1.y);
	return rect1;
}

// Clearance costs is calculated by determining any intersections between clearances and offlimits. Clearances may overlap with other clearances
template<int tile_sz>
__device__ float ClearanceCosts(cg::thread_block_tile<tile_sz> group, Surface *srf, positionAndRotation* cfg, vertex *vertices, rectangle *clearances, rectangle *offlimits)
{
	int tid = group.thread_rank();
	int step = group.size();
	float values;
	values = 0.0f; //Since it's size blockDim, we can have each of them treat it as the starting value
	for (int i = tid; i < srf->nClearances; i+=step) {
		vertex rect1Min = minValue(vertices, clearances[i].point1Index, cfg[clearances[i].SourceIndex].x, cfg[clearances[i].SourceIndex].y);
		vertex rect1Max = maxValue(vertices, clearances[i].point1Index, cfg[clearances[i].SourceIndex].x, cfg[clearances[i].SourceIndex].y);
		for (int j = tid; j < srf->nObjs; j += blockDim.x) {
			vertex rect2Min = minValue(vertices, offlimits[j].point1Index, cfg[j].x, cfg[j].y);
			vertex rect2Max = maxValue(vertices, offlimits[j].point1Index, cfg[j].x, cfg[j].y);
			// Determine max and min vectors of clearance rectangles
			// rectangle #1
			//printf("Clearance\n");
			//printf("Translation: X: %f Y: %f\n", cfg[i].x, cfg[i].y);
			// printf("Clearance rectangle %d: Min X: %f Y: %f Max X: %f Y: %f\n", i, rect1Min.x, rect1Min.y, rect1Max.x, rect1Max.y);
			// rectangle #2
			//printf("Off limits\n");
			//printf("Translation: X: %f Y: %f\n", cfg[j].x, cfg[j].y);
			// printf("Clearance rectangle %d: Min X: %f Y: %f Max X: %f Y: %f\n", i, rect2Min.x, rect2Min.y, rect2Max.x, rect2Max.y);
			float area = calculateIntersectionArea(rect1Min, rect1Max, rect2Min, rect2Max);
			//printf("Area intersection rectangle %d and %d: %f\n", i, j, area);
			values = area; //Clearence penalty should be positive
		}
	}
	group.sync();
	reduce<tile_sz>(group, values);
	//printf("Clearance costs error: %f\n", error);
	return values;
}

// Both clearance as offlimits may not lie outside of the surface area
template<int tile_sz>
__device__ float SurfaceAreaCosts(cg::thread_block_tile<tile_sz> group, Surface *srf, positionAndRotation* cfg, vertex *vertices, rectangle *clearances, rectangle *offlimits, vertex *surfaceRectangle) {
	//printf("Surface cost calculation\n");

	int tid = group.thread_rank();
	int step = group.size();
	float values;
	values = 0.0f; //Since it's size blockDim, we can have each of them treat it as the starting value
	// Describe the complement of surfaceRectangle as four rectangles (using their min and max values)
	vertex complementRectangle1[2];
	vertex complementRectangle2[2];
	vertex complementRectangle3[2];
	vertex complementRectangle4[2];

	// Figure out min and max vectors of surface rectangle
	vertex srfRect1Min = minValue(surfaceRectangle, 0, 0, 0);
	vertex srfRect1Max = maxValue(surfaceRectangle, 0, 0, 0);

	//This gives us the total rectangle outside our surface area
	createComplementRectangle(srfRect1Min, srfRect1Max, complementRectangle1, complementRectangle2, complementRectangle3, complementRectangle4);

	for (int i = tid; i < srf->nClearances; i += step) {
		// Determine max and min vectors of clearance rectangles
		// rectangle #1
		//Old way of doing things through a memory error
		//vertex rect1Min = minValue(vertices, clearances[i].point1Index, cfg[i].x, cfg[i].y);
		//vertex rect1Max = maxValue(vertices, clearances[i].point1Index, cfg[i].x, cfg[i].y);
		vertex rect1Min = minValue(vertices, clearances[i].point1Index, cfg[clearances[i].SourceIndex].x, cfg[clearances[i].SourceIndex].y);
		vertex rect1Max = maxValue(vertices, clearances[i].point1Index, cfg[clearances[i].SourceIndex].x, cfg[clearances[i].SourceIndex].y);

		// printf("Clearance rectangle %d: Min X: %f Y: %f Max X: %f Y: %f\n", i, rect1Min.x, rect1Min.y, rect1Max.x, rect1Max.y);


		// printf("Area intersection rectangle %d and %d: %f\n", i, j, area);
		values -= calculateIntersectionArea(rect1Min, rect1Max, complementRectangle1[0], complementRectangle1[1]);
		values -= calculateIntersectionArea(rect1Min, rect1Max, complementRectangle2[0], complementRectangle2[1]);
		values -= calculateIntersectionArea(rect1Min, rect1Max, complementRectangle3[0], complementRectangle3[1]);
		values -= calculateIntersectionArea(rect1Min, rect1Max, complementRectangle4[0], complementRectangle4[1]);
	}

	for (int j = tid; j < srf->nObjs; j += step) {
		// Determine max and min vectors of off limit rectangles
		// rectangle #1
		//offlimits is the size of cfg
		vertex rect1Min = minValue(vertices, offlimits[j].point1Index, cfg[j].x, cfg[j].y);
		vertex rect1Max = maxValue(vertices, offlimits[j].point1Index, cfg[j].x, cfg[j].y);

		// printf("Clearance rectangle %d: Min X: %f Y: %f Max X: %f Y: %f\n", i, rect1Min.x, rect1Min.y, rect1Max.x, rect1Max.y);
		values -= calculateIntersectionArea(rect1Min, rect1Max, complementRectangle1[0], complementRectangle1[1]);
		values -= calculateIntersectionArea(rect1Min, rect1Max, complementRectangle2[0], complementRectangle2[1]);
		values -= calculateIntersectionArea(rect1Min, rect1Max, complementRectangle3[0], complementRectangle3[1]);
		values -= calculateIntersectionArea(rect1Min, rect1Max, complementRectangle4[0], complementRectangle4[1]);
	}

	group.sync();
	reduce<tile_sz>(group, values);
	//printf("Clearance costs error: %f\n", error);
	return values;
}

template<int tile_sz>
__device__ float OffLimitsCosts(cg::thread_block_tile<tile_sz> group, Surface *srf, positionAndRotation *cfg, vertex *vertices, rectangle *offlimits) {
	int tid = group.thread_rank();
	int step = group.size();
	float values;
	values = 0.0f; //Since it's size blockDim, we can have each of them treat it as the starting value
	for (int i = tid; i < srf->nObjs; i += step) {
		vertex rect1Min = minValue(vertices, offlimits[i].point1Index, cfg[i].x, cfg[i].y);
		vertex rect1Max = maxValue(vertices, offlimits[i].point1Index, cfg[i].x, cfg[i].y);
		for (int j = i + 1; j < srf->nObjs; j++) {
			// Determine max and min vectors of clearance rectangles
			// rectangle #1
			//printf("Clearance\n");
			//printf("Translation: X: %f Y: %f\n", cfg[i].x, cfg[i].y);
			// printf("Clearance rectangle %d: Min X: %f Y: %f Max X: %f Y: %f\n", i, rect1Min.x, rect1Min.y, rect1Max.x, rect1Max.y);
			// rectangle #2
			//printf("Off limits\n");
			//printf("Translation: X: %f Y: %f\n", cfg[j].x, cfg[j].y);
			//offlimits is the size of cfg
			vertex rect2Min = minValue(vertices, offlimits[j].point1Index, cfg[j].x, cfg[j].y);
			vertex rect2Max = maxValue(vertices, offlimits[j].point1Index, cfg[j].x, cfg[j].y);

			// printf("Clearance rectangle %d: Min X: %f Y: %f Max X: %f Y: %f\n", i, rect2Min.x, rect2Min.y, rect2Max.x, rect2Max.y);

			float area = calculateIntersectionArea(rect1Min, rect1Max, rect2Min, rect2Max);
			//printf("Area intersection rectangle %d and %d: %f\n", i, j, area);
			values -= area;
		}

	}

	group.sync();
	reduce<tile_sz>(group, values);
	//printf("Clearance costs error: %f\n", error);
	return values;
}

template<int tile_sz>
__device__ void Costs(cg::thread_block_tile<tile_sz> group, Surface *srf, resultCosts* costs, positionAndRotation* cfg, relationshipStruct *rs, vertex *vertices, rectangle *clearances, rectangle *offlimits, vertex *surfaceRectangle)
{
	int gid = group.thread_rank();
	//float pairWiseCosts = 0;
	float pairWiseCosts =  PairWiseTotalCosts<tile_sz>(group, srf, cfg, rs);
	pairWiseCosts *= srf->WeightPairWise;

	// printf("Pair wise costs with weight %f\n", pairWiseCosts);

	//float visualBalanceCosts = 0;
	float visualBalanceCosts = srf->WeightVisualBalance * VisualBalanceCosts<tile_sz>(group, srf, cfg);
	
	// printf("Visual balance costs with weight %f\n", visualBalanceCosts);

	//float focalPointCosts = 0;
	float focalPointCosts = srf->WeightFocalPoint * FocalPointCosts<tile_sz>(group, srf, cfg);
	
	// printf("Focal point costs with weight %f\n", focalPointCosts);

	//float symmertryCosts = 0;
	float symmertryCosts = srf->WeightSymmetry * SymmetryCosts<tile_sz>(group, srf, cfg);
	
	// printf("Symmertry costs with weight %f\n", symmertryCosts);

	//float offlimitsCosts = 0;
	float offlimitsCosts = srf->WeightOffLimits * OffLimitsCosts<tile_sz>(group, srf, cfg, vertices, offlimits);
	// printf("OffLimits costs with weight %f\n", offlimitsCosts);
	

	//float clearanceCosts = 0;
	float clearanceCosts = srf->WeightClearance * ClearanceCosts<tile_sz>(group, srf, cfg, vertices, clearances, offlimits);
	// printf("Clearance costs with weight %f\n", clearanceCosts);
	

	//float surfaceAreaCosts = 0;
	float surfaceAreaCosts = srf->WeightSurfaceArea * SurfaceAreaCosts<tile_sz>(group, srf, cfg, vertices, clearances, offlimits, surfaceRectangle);
	// printf("Surface area costs with weight %f\n", surfaceAreaCosts);
	
	float totalCosts = pairWiseCosts + visualBalanceCosts + focalPointCosts + symmertryCosts + clearanceCosts + surfaceAreaCosts;
	if (gid == 0) {
		costs->PairWiseCosts = pairWiseCosts;
		costs->VisualBalanceCosts = visualBalanceCosts;
		costs->FocalPointCosts = focalPointCosts;
		costs->SymmetryCosts = symmertryCosts;
		costs->OffLimitsCosts = offlimitsCosts;
		costs->ClearanceCosts = clearanceCosts;
		costs->SurfaceAreaCosts = surfaceAreaCosts;
		costs->totalCosts = totalCosts;
	}
	group.sync();
	
	// printf("Total costs %f\n", totalCosts);
	
}

__device__ void CopyCosts(resultCosts* copyFrom, resultCosts* copyTo) 
{
	copyTo->PairWiseCosts = copyFrom->PairWiseCosts;
	copyTo->VisualBalanceCosts = copyFrom->VisualBalanceCosts;
	copyTo->FocalPointCosts = copyFrom->FocalPointCosts;
	copyTo->SymmetryCosts = copyFrom->SymmetryCosts;
	//printf("Copying Clearance costs with weight %f\n", copyFrom->ClearanceCosts);
	copyTo->ClearanceCosts = copyFrom->ClearanceCosts;
	copyTo->OffLimitsCosts = copyFrom->OffLimitsCosts;
	//printf("Copying Surface area costs with weight %f\n", copyFrom->SurfaceAreaCosts);
	copyTo->SurfaceAreaCosts = copyFrom->SurfaceAreaCosts;
	copyTo->totalCosts = copyFrom->totalCosts;
}

//The next two device helper functions generate random numbers
__inline__ __device__ int generateRandomIntInRange(curandState *rngStates, unsigned int tid, int max, int min)
{
	curandState localState = rngStates[tid];
	float p_rand = curand_uniform(&localState);
	rngStates[tid] = localState;
	p_rand *= (max - min + 0.999999);
	p_rand += min;
	return (int)truncf(p_rand);
}

__inline__ __device__ float generateRandomFloatInRange(curandState *rngStates, unsigned int tid, int max, int min)
{
	curandState localState = rngStates[tid];
	float p_rand = curand_uniform(&localState);
	rngStates[tid] = localState;
	p_rand *= (max - min + 0.999999);
	p_rand += min;
	return p_rand; //The only difference between float and int is that we do not trucate the float in this one
}

template<int tile_sz>
__device__ void propose(cg::thread_block_tile<tile_sz> group, Surface *srf, positionAndRotation *cfg, vertex * surfaceRectangle, curandState *rngStates, unsigned int tid)
{
	int gid = group.thread_rank();
	/*for (int j = 0; j < srf->nObjs; j++)
	{
		printf("Star values inside proposition jndex %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", j, cfgStar[j].x, cfgStar[j].y, cfgStar[j].z, cfgStar[j].rotX, cfgStar[j].rotY, cfgStar[j].rotZ);
	}*/
	int p = generateRandomIntInRange(rngStates, tid, 2, 0);

	//Get everyone on the same page
	p = group.shfl(p, 0); //broadcast out to p
	//group.sync(); shlf_sync is broadcast and so should sync
	// Determine width and length of surface rectangle
	vertex srfRect1Min = minValue(surfaceRectangle, 0, 0, 0);
	vertex srfRect1Max = maxValue(surfaceRectangle, 0, 0, 0);
	float width = srfRect1Max.x - srfRect1Min.x;
	float height = srfRect1Max.y - srfRect1Min.y;
	// Dividing the width by 2 makes sure that it stays withing a 95% percentile range that is usable, dividing it by 4 makes sure that it stretches the half of the length/width or lower (and that inside a 95% interval).
	float stdXAxis = width / 16;
	float stdYAxis = height / 16;

	// printf("Selected mode: %d\n", p);
	// Translate location using normal distribution
	if (p == 0)
	{
		//bool found = false;
		int obj = -1;
		// Take 100 tries to find a random nonfrozen object
		//for (int i = 0; i < 100 && !found; i++) {
		obj = generateRandomIntInRange(rngStates, tid, srf->nObjs - 1, 0);
		if (!group.any(!cfg[obj].frozen)) {
			return;
		}
		int mask = group.ballot(!cfg[obj].frozen);
		int leader = __ffs(mask);
		obj = group.shfl(obj, leader);


		//printf("Selected object #: %d\n", obj);
		if (gid == 0) {
			float dx = curand_normal(&rngStates[tid]);
			dx = dx * stdXAxis;
			//printf("dx: %f\n", dx);
			float dy = curand_normal(&rngStates[tid]);
			dy = dy * stdYAxis;
			// printf("Before translation, obj %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", obj, cfgStar[obj].x, cfgStar[obj].y, cfgStar[obj].z, cfgStar[obj].rotX, cfgStar[obj].rotY, cfgStar[obj].rotZ);

			// When object exceeds surfacearea, snap it back.
			if (cfg[obj].x + dx > srfRect1Max.x) {
				cfg[obj].x = srfRect1Max.x;
			}
			else if (cfg[obj].x + dx < srfRect1Min.x) {
				cfg[obj].x = srfRect1Min.x;
			}
			else {
				cfg[obj].x += dx;
			}
			if (cfg[obj].y + dy > srfRect1Max.y) {
				cfg[obj].y = srfRect1Max.y;
			}
			else if (cfg[obj].y + dy < srfRect1Min.y) {
				cfg[obj].y = srfRect1Min.y;
			}
			else {
				cfg[obj].y += dy;
			}
		}
		// printf("After rotation, obj %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", obj, cfgStar[obj].x, cfgStar[obj].y, cfgStar[obj].z, cfgStar[obj].rotX, cfgStar[obj].rotY, cfgStar[obj].rotZ);
	}
	// Translate rotation using normal distribution
	else if (p == 1)
	{
		int obj = -1;
		// Take 100 tries to find a random nonfrozen object
		//for (int i = 0; i < 100 && !found; i++) {
		obj = generateRandomIntInRange(rngStates, tid, srf->nObjs - 1, 0);
		if (!group.any(!cfg[obj].frozen)) {
			return;
		}
		int mask = group.ballot(!cfg[obj].frozen);
		int leader = __ffs(mask);
		obj = group.shfl(obj, leader);


		if (gid == 0) {
			// printf("Selected object #: %d\n", obj);
			// printf("Before rotation, obj %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", obj, cfgStar[obj].x, cfgStar[obj].y, cfgStar[obj].z, cfgStar[obj].rotX, cfgStar[obj].rotY, cfgStar[obj].rotZ);
			float dRot = curand_normal(&rngStates[tid]);
			dRot = dRot * S_SIGMA_T;
			// printf("dRot: %f\n", dRot);
			// printf("before rotation: %f\n", cfgStar[obj].rotY);
			cfg[obj].rotY += dRot;
			// printf("After rotation: %f\n", cfgStar[obj].rotY);

			if (cfg[obj].rotY < 0)
				cfg[obj].rotY += 2 * PI;
			else if (cfg[obj].rotY > 2 * PI)
				cfg[obj].rotY -= 2 * PI;
		}
		// printf("After rotation, obj %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", obj, cfgStar[obj].x, cfgStar[obj].y, cfgStar[obj].z, cfgStar[obj].rotX, cfgStar[obj].rotY, cfgStar[obj].rotZ);
	}
	// Swap two objects for both location and rotation
	else
	{
		if (srf->nObjs < 2) {
			return;
		}
		// This can result in the same object, chance becomes increasingly smaller given more objects
		int obj1 = -1;
		int obj2 = -1;
		obj1 = generateRandomIntInRange(rngStates, tid, srf->nObjs - 1, 0);
		if (!group.any(!cfg[obj1].frozen)) {
			return;
		}
		int mask = group.ballot(!cfg[obj1].frozen);
		int leader = __ffs(mask);
		obj1 = group.shfl(obj1, leader);

		obj2 = generateRandomIntInRange(rngStates, tid, srf->nObjs - 1, 0);
		if (!group.any(!cfg[obj2].frozen)) {
			return;
		}
		mask = group.ballot(!cfg[obj2].frozen);
		leader = __ffs(mask);
		obj2 = group.shfl(obj2, leader);

		if (obj1 == obj2) {
			return; //No point at this step
		}
		// printf("First selected object #: %d\n", obj1);
		// printf("Second selected object #: %d\n", obj2);

		// printf("Values, obj %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", obj1, cfgStar[obj1].x, cfgStar[obj1].y, cfgStar[obj1].z, cfgStar[obj1].rotX, cfgStar[obj1].rotY, cfgStar[obj1].rotZ);
		// printf("Values of, obj %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", obj2, cfgStar[obj2].x, cfgStar[obj2].y, cfgStar[obj2].z, cfgStar[obj2].rotX, cfgStar[obj2].rotY, cfgStar[obj2].rotZ);
		if (gid == 0) {
			// Temporarily store cfgStar[obj1] values
			float x = cfg[obj1].x;
			float y = cfg[obj1].y;
			float z = cfg[obj1].z;
			float rotX = cfg[obj1].rotX;
			float rotY = cfg[obj1].rotY;
			float rotZ = cfg[obj1].rotZ;
			// printf("After copy obj1 to temp, obj %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", obj1, cfgStar[obj1].x, cfgStar[obj1].y, cfgStar[obj1].z, cfgStar[obj1].rotX, cfgStar[obj1].rotY, cfgStar[obj1].rotZ);
			// printf("After copy obj1 to temp, obj %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", obj2, cfgStar[obj2].x, cfgStar[obj2].y, cfgStar[obj2].z, cfgStar[obj2].rotX, cfgStar[obj2].rotY, cfgStar[obj2].rotZ);

			// Move values of obj2 to obj1
			cfg[obj1].x = cfg[obj2].x;
			cfg[obj1].y = cfg[obj2].y;
			cfg[obj1].z = cfg[obj2].z;
			cfg[obj1].rotX = cfg[obj2].rotX;
			cfg[obj1].rotY = cfg[obj2].rotY;
			cfg[obj1].rotZ = cfg[obj2].rotZ;
			// printf("After copy obj2 into obj1, obj %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", obj1, cfgStar[obj1].x, cfgStar[obj1].y, cfgStar[obj1].z, cfgStar[obj1].rotX, cfgStar[obj1].rotY, cfgStar[obj1].rotZ);
			// printf("After copy obj2 into obj1, obj %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", obj2, cfgStar[obj2].x, cfgStar[obj2].y, cfgStar[obj2].z, cfgStar[obj2].rotX, cfgStar[obj2].rotY, cfgStar[obj2].rotZ);

			// Move stored values of obj1 to obj2
			cfg[obj2].x = x;
			cfg[obj2].y = y;
			cfg[obj2].z = z;
			cfg[obj2].rotX = rotX;
			cfg[obj2].rotY = rotY;
			cfg[obj2].rotZ = rotZ;
		}
		// printf("After copy temp into obj2, obj %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", obj1, cfgStar[obj1].x, cfgStar[obj1].y, cfgStar[obj1].z, cfgStar[obj1].rotX, cfgStar[obj1].rotY, cfgStar[obj1].rotZ);
		// printf("After copy temp into obj2, obj %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", obj2, cfgStar[obj2].x, cfgStar[obj2].y, cfgStar[obj2].z, cfgStar[obj2].rotX, cfgStar[obj2].rotY, cfgStar[obj2].rotZ);
	}
}

__device__ bool Accept(double costStar, double costCur, curandState *rngStates, unsigned int tid,float beta)
{
	//printf("(costStar - costCur):  %f\n", (costStar - costCur));
	//printf("(float) exp(-BETA * (costStar - costCur)): %f\n", (float)exp(-BETA * (costStar - costCur)));
	float randomNumber = curand_uniform(&rngStates[tid]);
	//printf("Random number: %f\n", randomNumber);
	return  randomNumber < fminf(1.0f, (float) exp(beta * (costStar - costCur)));
}

template<int tile_sz>
__device__ void Copy(cg::thread_block_tile<tile_sz> group, positionAndRotation* cfg1, positionAndRotation* cfg2, Surface* srf)
{
	int tid = group.thread_rank();
	int step = group.size();
	for (unsigned int i = tid; i < srf->nObjs; i += step)
	{
		cfg1[i].x = cfg2[i].x;
		cfg1[i].y = cfg2[i].y;
		cfg1[i].z = cfg2[i].z;
		cfg1[i].rotX = cfg2[i].rotX;
		cfg1[i].rotY = cfg2[i].rotY;
		cfg1[i].rotZ = cfg2[i].rotZ;
		cfg1[i].frozen = cfg2[i].frozen;
		cfg1[i].length = cfg2[i].length;
		cfg1[i].width = cfg2[i].width;
	}
	group.sync();
}

template<int tile_sz>
__device__ void groupKernel(cg::thread_block_tile<tile_sz> group,
	positionAndRotation* cfgBest,
	resultCosts* bestCosts,
	positionAndRotation* cfgStar,
	resultCosts* starCosts,
	relationshipStruct *rs,
	rectangle *clearances, rectangle *offlimits,
	vertex *vertices, vertex *surfaceRectangle, Surface *srf,
	int iterations, curandState *rngStates)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int gtid = group.thread_rank();//The thread id in the working group
	int step = group.size();//The working group block size
							// Read out starting configuration from resultArray
							// Copy best config (now set to input config) to result of this block

	bool accept;
	float beta = generateRandomFloatInRange(rngStates, tid, 3, 0);
	beta = group.shfl(beta, 0);//shf calls shfl_sync, which as a broadcast should sync
	Copy<tile_sz>(group, cfgStar, cfgBest, srf);

	Costs<tile_sz>(group, srf, bestCosts, cfgBest, rs, vertices, clearances, offlimits, surfaceRectangle); //possible race condition here
	CopyCosts(bestCosts, starCosts);
	//printf("Threadblock: %d, Best costs before: %f\n", blockIdx.x, bestCosts->totalCosts);

	for (int i = 0; i < iterations; i++)
	{
		
		propose<tile_sz>(group, srf, cfgStar, surfaceRectangle, rngStates, tid);
		group.sync();
		Costs<tile_sz>(group, srf, starCosts, cfgStar, rs, vertices, clearances, offlimits, surfaceRectangle);
		if (gtid == 0) {
			accept = Accept(starCosts->totalCosts, bestCosts->totalCosts, rngStates, tid,beta);

		}
		accept = group.shfl(accept, 0);
		if (accept)
		{
			// Possible different approach: Set pointer of current to star, free up memory used by current? reinitialize star?
			//printf("Star accepted as new current.\n");
			// Copy star into current
			Copy<tile_sz>(group, cfgBest, cfgStar, srf);
			CopyCosts(starCosts, bestCosts);
		}
		else { //Reject it
			Copy<tile_sz>(group, cfgStar, cfgBest, srf);
			CopyCosts(bestCosts, starCosts);
		}

		// Check whether we continue with current or we continue with star
	}

	group.sync();
}

//Helper function to copy the information from global memory into a shared array
template<int tile_sz>
__device__ void copyToSharedMemory(cg::thread_block_tile<tile_sz> group,
	point *p,
	Surface *srf,
	positionAndRotation* configuration) {
	int gid = group.thread_rank();
	for (unsigned int i = gid; i < srf->nObjs; i += WARP_SIZE)
	{
		// BlockId counts from 0, so to properly multiply
		int index = blockIdx.x * srf->nObjs + i;
		configuration[i].x = p[index].x;
		configuration[i].y = p[index].y;
		configuration[i].z = p[index].z;
		configuration[i].rotX = p[index].rotX;
		configuration[i].rotY = p[index].rotY;
		configuration[i].rotZ = p[index].rotZ;
		configuration[i].frozen = p[index].frozen;
		configuration[i].length = p[index].length;
		configuration[i].width = p[index].width;
	}

}
//Helper function to copy the information from shared to global.
__device__ void copyToGlobalMemory(
	point *p,
	Surface *srf,
	resultCosts* resultCostsArray,
	positionAndRotation* configuration,
	resultCosts* costs,
	int lowest_cost) {
	//Copy current config back into the global memory
	// Copy best config (now set to input config) to result of this block
	for (unsigned int i = threadIdx.x; i < srf->nObjs; i += blockDim.x)
	{
		// BlockId counts from 0, so to properly multiply
		int index = blockIdx.x * srf->nObjs + i;
		p[index].x = configuration[lowest_cost * srf->nObjs + i].x;
		p[index].y = configuration[lowest_cost * srf->nObjs + i].y;
		p[index].z = configuration[lowest_cost * srf->nObjs + i].z;
		p[index].rotX = configuration[lowest_cost * srf->nObjs + i].rotX;
		p[index].rotY = configuration[lowest_cost * srf->nObjs + i].rotY;
		p[index].rotZ = configuration[lowest_cost * srf->nObjs + i].rotZ;
		p[index].frozen = configuration[lowest_cost * srf->nObjs + i].frozen;
		p[index].length = configuration[lowest_cost * srf->nObjs + i].length;
		p[index].width = configuration[lowest_cost * srf->nObjs + i].width;
	}
	//printf("Threadblock: %d, Result costs before: %f\n", blockIdx.x, bestCosts->totalCosts);
	resultCostsArray[blockIdx.x].totalCosts = costs[lowest_cost].totalCosts;
	resultCostsArray[blockIdx.x].PairWiseCosts = costs[lowest_cost].PairWiseCosts;
	resultCostsArray[blockIdx.x].VisualBalanceCosts = costs[lowest_cost].VisualBalanceCosts;
	resultCostsArray[blockIdx.x].FocalPointCosts = costs[lowest_cost].FocalPointCosts;
	resultCostsArray[blockIdx.x].SymmetryCosts = costs[lowest_cost].SymmetryCosts;
	//printf("Best surface area costs: %f\n", bestCosts->SurfaceAreaCosts);
	resultCostsArray[blockIdx.x].SurfaceAreaCosts = costs[lowest_cost].SurfaceAreaCosts;
	//printf("Best clearance costs: %f\n", bestCosts->ClearanceCosts);
	resultCostsArray[blockIdx.x].ClearanceCosts = costs[lowest_cost].ClearanceCosts;
	resultCostsArray[blockIdx.x].OffLimitsCosts = costs[lowest_cost].OffLimitsCosts;

}

//This function figures out the lowest cost of our search
//It can be written as a reduction problem, and definitely should
__device__ int lowestIndex(resultCosts* best_costs, int active_warps) {
	int best_cost = 0;
	for (int i = 0; i < active_warps; i++) {
		if (best_costs[i].totalCosts < best_costs[best_cost].totalCosts) {
			best_cost = i;
		}
	}
	return best_cost;
}
// result is a [,] array with 1 dimension equal to the amount of blocks used and the other dimension equal to the amount of objects
// rs is an array with the length equal to the amount of relationships
// cfg is an array with the length equal to the amount of objects
// Surface is a basic struct

__global__ void Kernel(resultCosts* resultCostsArray,
	point *p, relationshipStruct *rs,
	rectangle *clearances, rectangle *offlimits,
	vertex *vertices, vertex *surfaceRectangle, Surface *srf,
	gpuConfig *gpuCfg, curandState *rngStates) {

	extern __shared__ int all_shared_memory[];
	int jumper = blockDim.x / WARP_SIZE;
	positionAndRotation* configurations = (positionAndRotation*)&all_shared_memory;
	resultCosts* costs = (resultCosts*)&configurations[2 * jumper * srf->nObjs]; 
	__syncthreads();
   //create the working groups
	int rank = threadIdx.x / WARP_SIZE;
	positionAndRotation* best_conf = &configurations[rank * srf->nObjs];
	positionAndRotation* star_conf = &configurations[srf->nObjs * blockDim.x / WARP_SIZE + rank];
	resultCosts* best_cost = &costs[rank];
	resultCosts* star_cost = &costs[jumper + rank];
	auto tile_warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block()); //Broken up by our warp size, which is our static shared memory size!

	//This is the actual work done
	copyToSharedMemory<WARP_SIZE>(tile_warp, p, srf, best_conf);
	groupKernel<WARP_SIZE>(tile_warp,best_conf,best_cost,star_conf,star_cost, rs, clearances, offlimits, vertices, surfaceRectangle, srf, gpuCfg->iterations, rngStates);
	__syncthreads();
	int lowest_cost = lowestIndex(best_cost, jumper);
	copyToGlobalMemory(p, srf, resultCostsArray, best_conf, costs, lowest_cost);
	__syncthreads();

}


extern "C" __declspec(dllexport) result* KernelWrapper(relationshipStruct *rss, point *previouscfgs, rectangle *clearances, rectangle *offlimits, vertex *vertices, vertex *surfaceRectangle, Surface *srf, gpuConfig *gpuCfg)
{
	// Create pointer for on gpu
	// Determine memory size of object to transfer
	// Malloc on GPU size
	// Cpy memory from cpu to gpu
	relationshipStruct *gpuRS;
	int rsSize = sizeof(relationshipStruct) * srf->nRelationships;
	checkCudaErrors(cudaMalloc(&gpuRS, rsSize));
	checkCudaErrors(cudaMemcpy(gpuRS, rss, rsSize, cudaMemcpyHostToDevice));

	// Input
	rectangle *gpuClearances;
	int clearancesSize = sizeof(rectangle) * srf->nClearances;
	checkCudaErrors(cudaMalloc(&gpuClearances, clearancesSize));
	checkCudaErrors(cudaMemcpy(gpuClearances, clearances, clearancesSize, cudaMemcpyHostToDevice));

	rectangle *gpuOfflimits;
	int offlimitsSize = sizeof(rectangle) * srf->nObjs;
	checkCudaErrors(cudaMalloc(&gpuOfflimits, offlimitsSize));
	checkCudaErrors(cudaMemcpy(gpuOfflimits, offlimits, offlimitsSize, cudaMemcpyHostToDevice));

	vertex *gpuVertices;
	int verticesSize = sizeof(vertex) * (srf->nClearances * 4 + srf->nObjs * 4);
	checkCudaErrors(cudaMalloc(&gpuVertices, verticesSize));
	checkCudaErrors(cudaMemcpy(gpuVertices, vertices, verticesSize, cudaMemcpyHostToDevice));

	vertex *gpuSurfaceRectangle;
	int surfaceRectangleSize = sizeof(vertex) * 4;
	checkCudaErrors(cudaMalloc(&gpuSurfaceRectangle, surfaceRectangleSize));
	checkCudaErrors(cudaMemcpy(gpuSurfaceRectangle, surfaceRectangle, surfaceRectangleSize, cudaMemcpyHostToDevice));

	Surface *gpuSRF;
	int srfSize = sizeof(Surface);
	checkCudaErrors(cudaMalloc(&gpuSRF, srfSize));
	checkCudaErrors(cudaMemcpy(gpuSRF, srf, srfSize, cudaMemcpyHostToDevice));

	gpuConfig *gpuGpuConfig;
	int gpuCFGSize = sizeof(gpuConfig);
	checkCudaErrors(cudaMalloc(&gpuGpuConfig, gpuCFGSize));
	checkCudaErrors(cudaMemcpy(gpuGpuConfig, gpuCfg, gpuCFGSize, cudaMemcpyHostToDevice));

	// Output
	point *gpuPointArray;
	int pointArraySize = srf->nObjs * sizeof(point) * gpuCfg->gridxDim;
	point *outPointArray = (point *) malloc(pointArraySize);
	checkCudaErrors(cudaMalloc((void**)&gpuPointArray, pointArraySize));
	checkCudaErrors(cudaMemcpy(gpuPointArray, previouscfgs, pointArraySize, cudaMemcpyHostToDevice));

	resultCosts *gpuResultCosts;
	int resultCostsSize = sizeof(resultCosts) * gpuCfg->gridxDim;
	resultCosts *outResultCosts = (resultCosts *)malloc(resultCostsSize);
	checkCudaErrors(cudaMalloc((void**)&gpuResultCosts, resultCostsSize));

	// cudaMemcpy(gpuPointArray, result, pointArraySize, cudaMemcpyHostToDevice);
	//Size of the shared array that holds the configuration data
	
	// Setup GPU random generator
	curandState *d_rngStates = 0;
	checkCudaErrors(cudaMalloc((void **)&d_rngStates, gpuCfg->gridxDim * gpuCfg->blockxDim * sizeof(curandState)));

	// Initialise random number generator
	initRNG <<<gpuCfg->gridxDim, gpuCfg->blockxDim>> > (d_rngStates, time(NULL));

	// Commented for possible later usage
	// dim3 dimGrid(gpuCfg->gridxDim, gpuCfg->gridyDim);
	// dim3 dimBlock(gpuCfg->blockxDim, gpuCfg->blockyDim, gpuCfg->blockzDim);
	
	// Block 1 dimensional, amount of threads available, configurable
	// Grid 1 dimension, amount of suggestions to be made.
	//we make the dynamic memory 3 times because we have at least 3 arrays that use it in one function
	int share_size = gpuCfg->blockxDim / WARP_SIZE * 2 * srf->nObjs * sizeof(positionAndRotation) + gpuCfg->blockxDim / WARP_SIZE * 2 * sizeof(resultCosts);
	Kernel <<<gpuCfg->gridxDim, gpuCfg->blockxDim,share_size>>>(gpuResultCosts, gpuPointArray, gpuRS, gpuClearances, gpuOfflimits, gpuVertices, gpuSurfaceRectangle, gpuSRF, gpuGpuConfig, d_rngStates);
	checkCudaErrors(cudaDeviceSynchronize());
	if (cudaSuccess != cudaGetLastError()) {
		fprintf(stderr, "cudaSafeCall() failed : %s\n",
			cudaGetErrorString(cudaGetLastError()));
	}

	// copy back results from gpu to cpu
	checkCudaErrors(cudaMemcpy(outPointArray, gpuPointArray, pointArraySize, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(outResultCosts, gpuResultCosts, resultCostsSize, cudaMemcpyDeviceToHost));

	// Free all allocated GPU memory
	cudaFree(gpuRS);
	cudaFree(gpuClearances);
	cudaFree(gpuOfflimits);
	cudaFree(gpuVertices);
	cudaFree(gpuSurfaceRectangle);
	cudaFree(gpuSRF);
	cudaFree(gpuGpuConfig);
	cudaFree(gpuPointArray);
	cudaFree(gpuResultCosts);

	// Construct return result
	result *resultPointer = (result*)malloc(sizeof(result) * gpuCfg->gridxDim);
	for (int i = 0; i < gpuCfg->gridxDim; i++)
	{
		resultPointer[i].costs.FocalPointCosts = outResultCosts[i].FocalPointCosts;
		resultPointer[i].costs.PairWiseCosts = outResultCosts[i].PairWiseCosts;
		resultPointer[i].costs.SymmetryCosts = outResultCosts[i].SymmetryCosts;
		resultPointer[i].costs.totalCosts = outResultCosts[i].totalCosts;
		resultPointer[i].costs.VisualBalanceCosts = outResultCosts[i].VisualBalanceCosts;
		resultPointer[i].costs.ClearanceCosts = outResultCosts[i].ClearanceCosts;
		resultPointer[i].costs.OffLimitsCosts = outResultCosts[i].OffLimitsCosts;
		resultPointer[i].costs.SurfaceAreaCosts = outResultCosts[i].SurfaceAreaCosts;
		resultPointer[i].points = &(outPointArray[i * srf->nObjs]);
	}
	return resultPointer;
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

	const int N = 10;
	const int NRel = 1;
	const int NClearances = 30;
	Surface srf;
	srf.nObjs = N;
	srf.nRelationships = NRel;
	srf.nClearances = NClearances;
	srf.WeightFocalPoint = 1.0f;
	srf.WeightPairWise = 1.0f;
	srf.WeightVisualBalance = 1.0f;
	srf.WeightSymmetry = 1.0f;
	srf.WeightClearance = 1.0f;
	srf.WeightSurfaceArea = 1.0f;
	srf.WeightOffLimits = 1.0f;
	srf.centroidX = 0.0;
	srf.centroidY = 0.0;
	srf.focalX = 5.0;
	srf.focalY = 5.0;
	srf.focalRot = 0.0;

	const int dimensions = 8;

	gpuConfig gpuCfg;

	gpuCfg.gridxDim = dimensions;
	gpuCfg.gridyDim = 0;
	gpuCfg.blockxDim = 4*WARP_SIZE;
	gpuCfg.blockyDim = 0;
	gpuCfg.blockzDim = 0;
	gpuCfg.iterations = 1000;//a 10th of what they claimed in the paper

	vertex surfaceRectangle[4];
	surfaceRectangle[0].x = 10;
	surfaceRectangle[0].y = 10;
	surfaceRectangle[0].z = 0;

	surfaceRectangle[1].x = 10;
	surfaceRectangle[1].y = 0;
	surfaceRectangle[1].z = 0;

	surfaceRectangle[2].x = 0;
	surfaceRectangle[2].y = 0;
	surfaceRectangle[2].z = 0;

	surfaceRectangle[3].x = 0;
	surfaceRectangle[3].y = 10;
	surfaceRectangle[3].z = 0;

	const int vertices = (N + NClearances) * 4;
	vertex vtx[vertices];
	for (int i = 0; i < (N); i++) {
		vtx[i*16+0].x = -1.8853001594543457;
		vtx[i*16 + 0].y = 1.1240049600601196;
		vtx[i*16 +0].z = 0;

		vtx[i*16 + 1].x = -0.88530009984970093;
		vtx[i*16 + 1].y = 1.1240049600601196;
		vtx[i*16 + 1].z = 0;

		vtx[i*16 + 2].x = -0.88530009984970093;
		vtx[i*16 + 2].y = -1.1240470409393311;
		vtx[i*16 + 2].z = 0;

		vtx[i*16 + 3].x = -1.8853001594543457;
		vtx[i*16 + 3].y = -1.1240470409393311;
		vtx[i*16 + 3].z = 0;

		vtx[i*16 + 4].x = 0.88240820169448853;
		vtx[i*16 + 4].y = 1.1240049600601196;
		vtx[i*16 + 4].z = 0;

		vtx[i*16 + 5].x = 1.8824081420898437;
		vtx[i*16 + 5].y = 1.1240049600601196;
		vtx[i*16 + 5].z = 0;

		vtx[i*16 + 6].x = 1.8824081420898437;
		vtx[i*16 + 6].y = -1.1240470409393311;
		vtx[i*16 + 6].z = 0;

		vtx[i*16 + 7].x = 0.88240820169448853;
		vtx[i*16 + 7].y = -1.1240470409393311;
		vtx[i*16 + 7].z = 0;

		vtx[i*16 + 8].x = -0.88530009984970093;
		vtx[i*16 + 8].y = 2.12400484085083;
		vtx[i*16 + 8].z = 0;

		vtx[i*16 + 9].x = 0.88240820169448853;
		vtx[i*16 + 9].y = 2.12400484085083;
		vtx[i*16 + 9].z = 0;

		vtx[i*16 + 10].x = 0.88240820169448853;
		vtx[i*16 + 10].y = 1.1240049600601196;
		vtx[i*16 + 10].z = 0;

		vtx[i*16 + 11].x = -0.88530009984970093;
		vtx[i*16 + 11].y = 1.1240049600601196;
		vtx[i*16 + 11].z = 0;

		vtx[i*16 + 12].x = -7.3193349838256836;
		vtx[i*16 + 12].y = -0.99961233139038086;
		vtx[i*16 + 12].z = 1.2984378337860107;

		vtx[i*16 + 13].x = -5.5516266822814941;
		vtx[i*16 + 13].y = -0.99961233139038086;
		vtx[i*16 + 13].z = 1.2984378337860107;

		vtx[i*16 + 14].x = -5.5516266822814941;
		vtx[i*16 + 14].y = -3.2476644515991211;
		vtx[i*16 + 14].z = 1.2984378337860107;

		vtx[i*16 + 15].x = -7.3193349838256836;
		vtx[i*16 + 15].y = -3.2476644515991211;
		vtx[i*16 + 15].z = 1.2984378337860107;
	}

	rectangle clearances[NClearances];
	rectangle offlimits[N];
	for (int i = 0; i < N; i++) {
		clearances[i*3].point1Index = i*15 + 0;
		clearances[i * 3].point2Index = i * 15 + 1;
		clearances[i * 3].point3Index = i * 15 + 2;
		clearances[i * 3].point4Index = i * 15 + 3;
		clearances[i * 3].SourceIndex = i;

		clearances[i * 3  + 1].point1Index = i * 15 + 4;
		clearances[i * 3 + 1].point2Index = i * 15 + 5;
		clearances[i * 3 + 1].point3Index = i * 15 + 6;
		clearances[i * 3 + 1].point4Index = i * 15 + 7;
		clearances[i * 3 + 1].SourceIndex = i;

		clearances[i * 3 + 2].point1Index = i * 15 + 8;
		clearances[i * 3 + 2].point2Index = i * 15 + 9;
		clearances[i * 3 + 2].point3Index = i * 15 + 10;
		clearances[i * 3 + 2].point4Index = i * 15 + 11;
		clearances[i * 3 + 2].SourceIndex = i;

		offlimits[i].point1Index = i * 15 + 12;
		offlimits[i].point2Index = i * 15 + 13;
		offlimits[i].point3Index = i * 15 + 14;
		offlimits[i].point4Index = i * 15 + 15;
		offlimits[i].SourceIndex = 0;
	}

	point cfg[N*dimensions];
	for (int i = 0; i < dimensions; i++) {
		for (unsigned int j = 0; j < N; j++)
		{
			// BlockId counts from 0, so to properly multiply
			int index = i * N + j;

			cfg[index].x = -6.4340348243713379;
			cfg[index].y = -2.12361741065979;
			cfg[index].z = 0.0;
			cfg[index].rotX = 0.0;
			cfg[index].rotY = 5.5179219245910645;
			cfg[index].rotZ = 0.0;
			cfg[index].frozen = false;
			cfg[index].length = 1.7677083015441895;
			cfg[index].width = 2.2480521202087402;
		}
	}

	// Create relationship
	relationshipStruct rss[1];
	rss[0].TargetRange.targetRangeStart = 2.0;
	rss[0].TargetRange.targetRangeEnd = 4.0;
	rss[0].AngleRange.targetRangeStart = 0.01*PI;
	rss[0].AngleRange.targetRangeEnd = PI;
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

	// Point test code:

	result *result = KernelWrapper(rss, cfg, clearances, offlimits, vtx, surfaceRectangle, &srf, &gpuCfg);
	printf("Results:\n");
	for (int i = 0; i < gpuCfg.gridxDim; i++)
	{
		printf("Result %d\n", i);
		for (int j = 0; j < srf.nObjs; j++) {
			printf("Point [%d] X,Y,Z: %f, %f, %f	Rotation: %f, %f, %f\n", 
				j,
				result[i].points[j].x, 
				result[i].points[j].y, 
				result[i].points[j].z, 
				result[i].points[j].rotX, 
				result[i].points[j].rotY,
				result[i].points[j].rotZ);
		}
		
	}
 	return EXIT_SUCCESS;
}