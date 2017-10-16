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

#define BETA (0.1)
// Right angle constants:
#define THETA_R (10.0 / 180.0 * PI) // 5 degrees

// Sampling constants:
#define S_SIGMA_P (0.8)
#define S_SIGMA_T (15.0 / 90.0 * PI)



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
	return -1.0 * Distance(nx / denom, ny / denom, srf->centroidX / 2, srf->centroidY / 2);
}

__device__ double PairWiseCosts(Surface *srf, positionAndRotation* cfg, relationshipStruct *rs)
{
	double result = 0;
	for (int i = 0; i < srf->nRelationships; i++)
	{
		// Look up source index from relationship and retrieve object using that index.
		double distance = Distance(cfg[rs[i].SourceIndex].x, cfg[rs[i].SourceIndex].y, cfg[rs[i].TargetIndex].x, cfg[rs[i].TargetIndex].y);
		//printf("Distance: %f Range start: %f Range end: %f\n", distance, rs[i].TargetRange.targetRangeStart, rs[i].TargetRange.targetRangeEnd);
		if (distance < rs[i].TargetRange.targetRangeStart)
		{
			double fraction = distance / rs[i].TargetRange.targetRangeStart;
			result -= (fraction * fraction);
		}
		else if (distance > rs[i].TargetRange.targetRangeEnd) 
		{
			double fraction = rs[i].TargetRange.targetRangeEnd / distance;
			result -= (fraction * fraction);
		}
		// Else don't do anything as 0 indicates a perfect solution
	}
	return result;
}

__device__ double FocalPointCosts(Surface *srf, positionAndRotation* cfg)
{
	double sum = 0;
	for (int i = 0; i < srf->nObjs; i++)
	{
		float phi_fi = phi(srf->focalX, srf->focalY, cfg[i].x, cfg[i].y, cfg[i].rotY);
		// Old implementation of grouping, all objects that belong to the seat category are used in the focal point calculation
		// For now we default to all objects, focal point grouping will come later
		//int s_i = s(r.c[i]);

		// sum += s_i * cos(phi_fi);
		sum -= cos(phi_fi);
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

		sum -= maxVal;
	}

	return sum;
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
__device__ float ClearanceCosts(Surface *srf, positionAndRotation* cfg, vertex *vertices, rectangle *clearances, rectangle *offlimits)
{
	//printf("Clearance cost calculation\n");
	float error = 0.0f;
	for (int i = 0; i < srf->nClearances; i++) {
		vertex rect1Min = minValue(vertices, clearances[i].point1Index, cfg[clearances[i].SourceIndex].x, cfg[clearances[i].SourceIndex].y);
		vertex rect1Max = maxValue(vertices, clearances[i].point1Index, cfg[clearances[i].SourceIndex].x, cfg[clearances[i].SourceIndex].y);

		for (int j = 0; j < srf->nObjs; j++) {
			// Determine max and min vectors of clearance rectangles
			// rectangle #1
			//printf("Clearance\n");
			//printf("Translation: X: %f Y: %f\n", cfg[i].x, cfg[i].y);

			// printf("Clearance rectangle %d: Min X: %f Y: %f Max X: %f Y: %f\n", i, rect1Min.x, rect1Min.y, rect1Max.x, rect1Max.y);

			// rectangle #2
			//printf("Off limits\n");
			//printf("Translation: X: %f Y: %f\n", cfg[j].x, cfg[j].y);
			vertex rect2Min = minValue(vertices, offlimits[j].point1Index, cfg[j].x, cfg[j].y);
			vertex rect2Max = maxValue(vertices, offlimits[j].point1Index, cfg[j].x, cfg[j].y);

			// printf("Clearance rectangle %d: Min X: %f Y: %f Max X: %f Y: %f\n", i, rect2Min.x, rect2Min.y, rect2Max.x, rect2Max.y);

			float area = calculateIntersectionArea(rect1Min, rect1Max, rect2Min, rect2Max);
			//printf("Area intersection rectangle %d and %d: %f\n", i, j, area);
			error -= area;
		}
	}
	//printf("Clearance costs error: %f\n", error);
	return error;
}

// Both clearance as offlimits may not lie outside of the surface area
__device__ float SurfaceAreaCosts(Surface *srf, positionAndRotation* cfg, vertex *vertices, rectangle *clearances, rectangle *offlimits, vertex *surfaceRectangle) {
	//printf("Surface cost calculation\n");

	float error = 0.0f;
	// Describe the complement of surfaceRectangle as four rectangles (using their min and max values)
	vertex complementRectangle1[2];
	vertex complementRectangle2[2];
	vertex complementRectangle3[2];
	vertex complementRectangle4[2];

	// Figure out min and max vectors of surface rectangle
	vertex srfRect1Min = minValue(surfaceRectangle, 0, 0, 0);
	vertex srfRect1Max = maxValue(surfaceRectangle, 0, 0, 0);

	createComplementRectangle(srfRect1Min, srfRect1Max, complementRectangle1, complementRectangle2, complementRectangle3, complementRectangle4);

	for (int i = 0; i < srf->nClearances; i++) {
		// Determine max and min vectors of clearance rectangles
		// rectangle #1
		vertex rect1Min = minValue(vertices, clearances[i].point1Index, cfg[i].x, cfg[i].y);
		vertex rect1Max = maxValue(vertices, clearances[i].point1Index, cfg[i].x, cfg[i].y);

		// printf("Clearance rectangle %d: Min X: %f Y: %f Max X: %f Y: %f\n", i, rect1Min.x, rect1Min.y, rect1Max.x, rect1Max.y);


		// printf("Area intersection rectangle %d and %d: %f\n", i, j, area);
		error -= calculateIntersectionArea(rect1Min, rect1Max, complementRectangle1[0], complementRectangle1[1]);
		error -= calculateIntersectionArea(rect1Min, rect1Max, complementRectangle2[0], complementRectangle2[1]);
		error -= calculateIntersectionArea(rect1Min, rect1Max, complementRectangle3[0], complementRectangle3[1]);
		error -= calculateIntersectionArea(rect1Min, rect1Max, complementRectangle4[0], complementRectangle4[1]);
	}

	for (int j = 0; j < srf->nObjs; j++) {
		// Determine max and min vectors of off limit rectangles
		// rectangle #1
		vertex rect1Min = minValue(vertices, offlimits[j].point1Index, cfg[j].x, cfg[j].y);
		vertex rect1Max = maxValue(vertices, offlimits[j].point1Index, cfg[j].x, cfg[j].y);

		// printf("Clearance rectangle %d: Min X: %f Y: %f Max X: %f Y: %f\n", i, rect1Min.x, rect1Min.y, rect1Max.x, rect1Max.y);
		error -= calculateIntersectionArea(rect1Min, rect1Max, complementRectangle1[0], complementRectangle1[1]);
		error -= calculateIntersectionArea(rect1Min, rect1Max, complementRectangle2[0], complementRectangle2[1]);
		error -= calculateIntersectionArea(rect1Min, rect1Max, complementRectangle3[0], complementRectangle3[1]);
		error -= calculateIntersectionArea(rect1Min, rect1Max, complementRectangle4[0], complementRectangle4[1]);
	}
	//printf("Surface area costs error: %f\n", error);
	return error;
}

__device__ float OffLimitsCosts(Surface *srf, positionAndRotation *cfg, vertex *vertices, rectangle *offlimits) {
	//printf("OffLimits cost calculation\n");
	float error = 0.0f;
	for (int i = 0; i < srf->nObjs; i++) {
		for (int j = i+1; j < srf->nObjs; j++) {
			// Determine max and min vectors of clearance rectangles
			// rectangle #1
			//printf("Clearance\n");
			//printf("Translation: X: %f Y: %f\n", cfg[i].x, cfg[i].y);
			vertex rect1Min = minValue(vertices, offlimits[i].point1Index, cfg[i].x, cfg[i].y);
			vertex rect1Max = maxValue(vertices, offlimits[i].point1Index, cfg[i].x, cfg[i].y);

			// printf("Clearance rectangle %d: Min X: %f Y: %f Max X: %f Y: %f\n", i, rect1Min.x, rect1Min.y, rect1Max.x, rect1Max.y);

			// rectangle #2
			//printf("Off limits\n");
			//printf("Translation: X: %f Y: %f\n", cfg[j].x, cfg[j].y);
			vertex rect2Min = minValue(vertices, offlimits[j].point1Index, cfg[j].x, cfg[j].y);
			vertex rect2Max = maxValue(vertices, offlimits[j].point1Index, cfg[j].x, cfg[j].y);

			// printf("Clearance rectangle %d: Min X: %f Y: %f Max X: %f Y: %f\n", i, rect2Min.x, rect2Min.y, rect2Max.x, rect2Max.y);

			float area = calculateIntersectionArea(rect1Min, rect1Max, rect2Min, rect2Max);
			//printf("Area intersection rectangle %d and %d: %f\n", i, j, area);
			error -= area;
		}
	}
	//printf("Clearance costs error: %f\n", error);
	return error;
}

__device__ void Costs(Surface *srf, resultCosts* costs, positionAndRotation* cfg, relationshipStruct *rs, vertex *vertices, rectangle *clearances, rectangle *offlimits, vertex *surfaceRectangle)
{
	//float pairWiseCosts = 0;
	float pairWiseCosts = srf->WeightPairWise * PairWiseCosts(srf, cfg, rs);
	costs->PairWiseCosts = pairWiseCosts;
	// printf("Pair wise costs with weight %f\n", pairWiseCosts);

	//float visualBalanceCosts = 0;
	float visualBalanceCosts = srf->WeightVisualBalance * VisualBalanceCosts(srf, cfg);
	costs->VisualBalanceCosts = visualBalanceCosts;
	// printf("Visual balance costs with weight %f\n", visualBalanceCosts);

	//float focalPointCosts = 0;
	float focalPointCosts = srf->WeightFocalPoint * FocalPointCosts(srf, cfg);
	costs->FocalPointCosts = focalPointCosts;
	// printf("Focal point costs with weight %f\n", focalPointCosts);

	//float symmertryCosts = 0;
	float symmertryCosts = srf->WeightSymmetry * SymmetryCosts(srf, cfg);
	costs->SymmetryCosts = symmertryCosts;
	// printf("Symmertry costs with weight %f\n", symmertryCosts);

	//float offlimitsCosts = 0;
	float offlimitsCosts = srf->WeightOffLimits * OffLimitsCosts(srf, cfg, vertices, offlimits);
	// printf("OffLimits costs with weight %f\n", offlimitsCosts);
	costs->OffLimitsCosts = offlimitsCosts;

	//float clearanceCosts = 0;
	float clearanceCosts = srf->WeightClearance * ClearanceCosts(srf, cfg, vertices, clearances, offlimits);
	// printf("Clearance costs with weight %f\n", clearanceCosts);
	costs->ClearanceCosts = clearanceCosts;

	//float surfaceAreaCosts = 0;
	float surfaceAreaCosts = srf->WeightSurfaceArea * SurfaceAreaCosts(srf, cfg, vertices, clearances, offlimits, surfaceRectangle);
	// printf("Surface area costs with weight %f\n", surfaceAreaCosts);
	costs->SurfaceAreaCosts = surfaceAreaCosts;


	float totalCosts = costs->PairWiseCosts + costs->VisualBalanceCosts + costs->FocalPointCosts + costs->SymmetryCosts + costs->ClearanceCosts + costs->SurfaceAreaCosts;
	// printf("Total costs %f\n", totalCosts);
	costs->totalCosts = totalCosts;
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

__device__ int generateRandomIntInRange(curandState *rngStates, unsigned int tid, int max, int min)
{
	curandState localState = rngStates[tid];
	float p_rand = curand_uniform(&localState);
	rngStates[tid] = localState;
	p_rand *= (max - min + 0.999999);
	p_rand += min;
	return (int)truncf(p_rand);
}

__device__ void propose(Surface *srf, positionAndRotation *cfgStar, vertex * surfaceRectangle, curandState *rngStates, unsigned int tid)
{
	/*for (int j = 0; j < srf->nObjs; j++)
	{
		printf("Star values inside proposition jndex %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", j, cfgStar[j].x, cfgStar[j].y, cfgStar[j].z, cfgStar[j].rotX, cfgStar[j].rotY, cfgStar[j].rotZ);
	}*/
	int p = generateRandomIntInRange(rngStates, tid, 2, 0);

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
		bool found = false;
		int obj = -1;
		// Take 100 tries to find a random nonfrozen object
		for (int i = 0; i < 100; i++) {
			obj = generateRandomIntInRange(rngStates, tid, srf->nObjs - 1, 0);
			if (!cfgStar[obj].frozen) {
				found = true;
				break;
			}
		}
		if (!found) {
			return;
		}

		//printf("Selected object #: %d\n", obj);
		float dx = curand_normal(&rngStates[tid]);
		dx = dx * stdXAxis;
		//printf("dx: %f\n", dx);
		float dy = curand_normal(&rngStates[tid]);
		dy = dy * stdYAxis;
		// printf("Before translation, obj %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", obj, cfgStar[obj].x, cfgStar[obj].y, cfgStar[obj].z, cfgStar[obj].rotX, cfgStar[obj].rotY, cfgStar[obj].rotZ);

		// When object exceeds surfacearea, snap it back.
		if (cfgStar[obj].x + dx > srfRect1Max.x) {
			cfgStar[obj].x = srfRect1Max.x;
		}
		else if (cfgStar[obj].x + dx < srfRect1Min.x) {
			cfgStar[obj].x = srfRect1Min.x;
		}
		else {
			cfgStar[obj].x += dx;
		}
		if (cfgStar[obj].y + dy > srfRect1Max.y) {
			cfgStar[obj].y = srfRect1Max.y;
		}
		else if (cfgStar[obj].y + dy < srfRect1Min.y) {
			cfgStar[obj].y = srfRect1Min.y;
		}
		else {
			cfgStar[obj].y += dy;
		}
		// printf("After rotation, obj %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", obj, cfgStar[obj].x, cfgStar[obj].y, cfgStar[obj].z, cfgStar[obj].rotX, cfgStar[obj].rotY, cfgStar[obj].rotZ);
	}
	// Translate rotation using normal distribution
	else if (p == 1)
	{
		int obj = -1;
		bool found = false;

		// Take 100 tries to find a random nonfrozen object
		for (int i = 0; i < 100; i++) {
			obj = generateRandomIntInRange(rngStates, tid, srf->nObjs - 1, 0);
			if (!cfgStar[obj].frozen) {
				found = true;
				break;
			}
		}
		if (!found) {
			return;
		}
		// printf("Selected object #: %d\n", obj);
		// printf("Before rotation, obj %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", obj, cfgStar[obj].x, cfgStar[obj].y, cfgStar[obj].z, cfgStar[obj].rotX, cfgStar[obj].rotY, cfgStar[obj].rotZ);
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
		bool found = false;

		// Take 100 tries to find a random nonfrozen object
		for (int i = 0; i < 100; i++) {
			obj1 = generateRandomIntInRange(rngStates, tid, srf->nObjs - 1, 0);
			if (!cfgStar[obj1].frozen) {
				found = true;
				break;
			}
		}
		if (!found) {
			return;
		}

		int obj2 = -1;
		found = false;

		// Take 100 tries to find a random nonfrozen object
		for (int i = 0; i < 100; i++) {
			obj2 = generateRandomIntInRange(rngStates, tid, srf->nObjs - 1, 0);
			if (!cfgStar[obj2].frozen) {
				found = true;
				break;
			}
		}
		if (!found) {
			return;
		}
		// printf("First selected object #: %d\n", obj1);
		// printf("Second selected object #: %d\n", obj2);

		// printf("Values, obj %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", obj1, cfgStar[obj1].x, cfgStar[obj1].y, cfgStar[obj1].z, cfgStar[obj1].rotX, cfgStar[obj1].rotY, cfgStar[obj1].rotZ);
		// printf("Values of, obj %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", obj2, cfgStar[obj2].x, cfgStar[obj2].y, cfgStar[obj2].z, cfgStar[obj2].rotX, cfgStar[obj2].rotY, cfgStar[obj2].rotZ);

		// Temporarily store cfgStar[obj1] values
		float x = cfgStar[obj1].x;
		float y = cfgStar[obj1].y;
		float z = cfgStar[obj1].z;
		float rotX = cfgStar[obj1].rotX;
		float rotY = cfgStar[obj1].rotY;
		float rotZ = cfgStar[obj1].rotZ;
		// printf("After copy obj1 to temp, obj %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", obj1, cfgStar[obj1].x, cfgStar[obj1].y, cfgStar[obj1].z, cfgStar[obj1].rotX, cfgStar[obj1].rotY, cfgStar[obj1].rotZ);
		// printf("After copy obj1 to temp, obj %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", obj2, cfgStar[obj2].x, cfgStar[obj2].y, cfgStar[obj2].z, cfgStar[obj2].rotX, cfgStar[obj2].rotY, cfgStar[obj2].rotZ);

		// Move values of obj2 to obj1
		cfgStar[obj1].x = cfgStar[obj2].x;
		cfgStar[obj1].y = cfgStar[obj2].y;
		cfgStar[obj1].z = cfgStar[obj2].z;
		cfgStar[obj1].rotX = cfgStar[obj2].rotX;
		cfgStar[obj1].rotY = cfgStar[obj2].rotY;
		cfgStar[obj1].rotZ = cfgStar[obj2].rotZ;
		// printf("After copy obj2 into obj1, obj %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", obj1, cfgStar[obj1].x, cfgStar[obj1].y, cfgStar[obj1].z, cfgStar[obj1].rotX, cfgStar[obj1].rotY, cfgStar[obj1].rotZ);
		// printf("After copy obj2 into obj1, obj %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", obj2, cfgStar[obj2].x, cfgStar[obj2].y, cfgStar[obj2].z, cfgStar[obj2].rotX, cfgStar[obj2].rotY, cfgStar[obj2].rotZ);

		// Move stored values of obj1 to obj2
		cfgStar[obj2].x = x;
		cfgStar[obj2].y = y;
		cfgStar[obj2].z = z;
		cfgStar[obj2].rotX = rotX;
		cfgStar[obj2].rotY = rotY;
		cfgStar[obj2].rotZ = rotZ;
		// printf("After copy temp into obj2, obj %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", obj1, cfgStar[obj1].x, cfgStar[obj1].y, cfgStar[obj1].z, cfgStar[obj1].rotX, cfgStar[obj1].rotY, cfgStar[obj1].rotZ);
		// printf("After copy temp into obj2, obj %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", obj2, cfgStar[obj2].x, cfgStar[obj2].y, cfgStar[obj2].z, cfgStar[obj2].rotX, cfgStar[obj2].rotY, cfgStar[obj2].rotZ);
	}
}

__device__ bool Accept(double costStar, double costCur, curandState *rngStates, unsigned int tid)
{
	//printf("(costStar - costCur):  %f\n", (costStar - costCur));
	//printf("(float) exp(-BETA * (costStar - costCur)): %f\n", (float)exp(-BETA * (costStar - costCur)));
	float randomNumber = curand_uniform(&rngStates[tid]);
	//printf("Random number: %f\n", randomNumber);
	return  randomNumber < fminf(1.0f, (float) exp(BETA * (costStar - costCur)));
}

__device__ void Copy(positionAndRotation* cfg1, positionAndRotation* cfg2, Surface* srf) 
{
	for (unsigned int i = 0; i < srf->nObjs; i++)
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
}

// result is a [,] array with 1 dimension equal to the amount of blocks used and the other dimension equal to the amount of objects
// rs is an array with the length equal to the amount of relationships
// cfg is an array with the length equal to the amount of objects
// Surface is a basic struct
__global__ void Kernel(resultCosts* resultCostsArray, point *p, relationshipStruct *rs, positionAndRotation* cfg, rectangle *clearances, rectangle *offlimits, vertex *vertices, vertex *surfaceRectangle, Surface *srf, gpuConfig *gpuCfg, curandState *rngStates)
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
	Copy(cfgCurrent, cfg, srf);
	resultCosts* currentCosts = (resultCosts*)malloc(sizeof(resultCosts));
	Costs(srf, currentCosts, cfgCurrent, rs, vertices, clearances, offlimits, surfaceRectangle);
	
	positionAndRotation* cfgBest = (positionAndRotation*)malloc(srf->nObjs * sizeof(positionAndRotation));
	Copy(cfgBest, cfgCurrent, srf);
	resultCosts* bestCosts = (resultCosts*)malloc(sizeof(resultCosts));
	CopyCosts(currentCosts, bestCosts);
	//printf("Threadblock: %d, Best costs before: %f\n", blockIdx.x, bestCosts->totalCosts);

	for (int i = 0; i < gpuCfg->iterations; i++)
	{
		// Create cfg Star and initialize it to cfgcurrent that will have a proposition done to it.
		positionAndRotation* cfgStar = (positionAndRotation*)malloc(srf->nObjs * sizeof(positionAndRotation));
		resultCosts* starCosts = (resultCosts*)malloc(sizeof(resultCosts));
		/*for (int j = 0; j < srf->nObjs; j++)
		{
			printf("Current values before copy of result jndex %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", j, cfgCurrent[j].x, cfgCurrent[j].y, cfgCurrent[j].z, cfgCurrent[j].rotX, cfgCurrent[j].rotY, cfgCurrent[j].rotZ);
		}*/
		Copy(cfgStar, cfgCurrent, srf);
		/*for (int j = 0; j < srf->nObjs; j++)
		{
			printf("Star values after Copy, before proposition jndex %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", j, cfgStar[j].x, cfgStar[j].y, cfgStar[j].z, cfgStar[j].rotX, cfgStar[j].rotY, cfgStar[j].rotZ);
		}*/
		// cfgStar contains an array with translated objects
		propose(srf, cfgStar, surfaceRectangle, rngStates, tid);
		/* for (int j = 0; j < srf->nObjs; j++)
		{
			printf("Star values after proposition jndex %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", j, cfgStar[j].x, cfgStar[j].y, cfgStar[j].z, cfgStar[j].rotX, cfgStar[j].rotY, cfgStar[j].rotZ);
		} */

		Costs(srf, starCosts, cfgStar, rs, vertices, clearances, offlimits, surfaceRectangle);
		// printf("Cost star configuration: %f\n", starCosts->totalCosts);
		// printf("Cost best configuration: %f\n", bestCosts->totalCosts);
		// star has a better cost function than best cost, star is the new best
		if (starCosts->totalCosts < bestCosts->totalCosts)
		{
			//printf("New best %f\n", bestCosts->totalCosts);

			// Copy star into best for storage
			Copy(cfgBest, cfgStar, srf);
			CopyCosts(starCosts, bestCosts);
			//printf("Threadblock: %d, New costs before: %f\n", blockIdx.x, bestCosts->totalCosts);
		}

		// Check whether we continue with current or we continue with star
		if (Accept(starCosts->totalCosts, currentCosts->totalCosts, rngStates, tid))
		{
			// Possible different approach: Set pointer of current to star, free up memory used by current? reinitialize star?
			//printf("Star accepted as new current.\n");
			// Copy star into current
			Copy(cfgCurrent, cfgStar, srf);
			CopyCosts(starCosts, currentCosts);
		}
		free(cfgStar);
		free(starCosts);
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
	}
	//printf("Threadblock: %d, Result costs before: %f\n", blockIdx.x, bestCosts->totalCosts);
	resultCostsArray[blockIdx.x].totalCosts = bestCosts->totalCosts;
	resultCostsArray[blockIdx.x].PairWiseCosts = bestCosts->PairWiseCosts;
	resultCostsArray[blockIdx.x].VisualBalanceCosts = bestCosts->VisualBalanceCosts;
	resultCostsArray[blockIdx.x].FocalPointCosts = bestCosts->FocalPointCosts;
	resultCostsArray[blockIdx.x].SymmetryCosts = bestCosts->SymmetryCosts;
	//printf("Best surface area costs: %f\n", bestCosts->SurfaceAreaCosts);
	resultCostsArray[blockIdx.x].SurfaceAreaCosts = bestCosts->SurfaceAreaCosts;
	//printf("Best clearance costs: %f\n", bestCosts->ClearanceCosts);
	resultCostsArray[blockIdx.x].ClearanceCosts = bestCosts->ClearanceCosts;
	resultCostsArray[blockIdx.x].OffLimitsCosts = bestCosts->OffLimitsCosts;

	free(cfgCurrent);
	free(currentCosts);
	free(cfgBest);
	free(bestCosts);
}

extern "C" __declspec(dllexport) result* KernelWrapper(relationshipStruct *rss, positionAndRotation* cfg, rectangle *clearances, rectangle *offlimits, vertex *vertices, vertex *surfaceRectangle, Surface *srf, gpuConfig *gpuCfg)
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
	positionAndRotation *gpuAlgoCFG;
	int algoCFGSize = sizeof(positionAndRotation) * srf->nObjs;
	checkCudaErrors(cudaMalloc(&gpuAlgoCFG, algoCFGSize));
	checkCudaErrors(cudaMemcpy(gpuAlgoCFG, cfg, algoCFGSize, cudaMemcpyHostToDevice));

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

	resultCosts *gpuResultCosts;
	int resultCostsSize = sizeof(resultCosts) * gpuCfg->gridxDim;
	resultCosts *outResultCosts = (resultCosts *)malloc(resultCostsSize);
	checkCudaErrors(cudaMalloc((void**)&gpuResultCosts, resultCostsSize));

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
	Kernel <<<gpuCfg->gridxDim, gpuCfg->blockxDim >>>(gpuResultCosts, gpuPointArray, gpuRS, gpuAlgoCFG, gpuClearances, gpuOfflimits, gpuVertices, gpuSurfaceRectangle, gpuSRF, gpuGpuConfig, d_rngStates);
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
	cudaFree(gpuAlgoCFG);
	cudaFree(gpuSRF);
	cudaFree(gpuGpuConfig);
	cudaFree(gpuPointArray);

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

	const int N = 5;
	const int NRel = 0;
	const int NClearances = 15;
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
	clearances[0].point1Index = 0;
	clearances[0].point2Index = 1;
	clearances[0].point3Index = 2;
	clearances[0].point4Index = 3;
	clearances[0].SourceIndex = 0;

	clearances[1].point1Index = 4;
	clearances[1].point2Index = 5;
	clearances[1].point3Index = 6;
	clearances[1].point4Index = 7;
	clearances[1].SourceIndex = 0;

	clearances[2].point1Index = 8;
	clearances[2].point2Index = 9;
	clearances[2].point3Index = 10;
	clearances[2].point4Index = 11;
	clearances[2].SourceIndex = 0;

	offlimits[0].point1Index = 12;
	offlimits[0].point2Index = 13;
	offlimits[0].point3Index = 14;
	offlimits[0].point4Index = 15;
	offlimits[0].SourceIndex = 0;

	clearances[3].point1Index = 16;
	clearances[3].point2Index = 17;
	clearances[3].point3Index = 18;
	clearances[3].point4Index = 19;
	clearances[3].SourceIndex = 1;

	clearances[4].point1Index = 20;
	clearances[4].point2Index = 21;
	clearances[4].point3Index = 22;
	clearances[4].point4Index = 23;
	clearances[4].SourceIndex = 1;

	clearances[5].point1Index = 24;
	clearances[5].point2Index = 25;
	clearances[5].point3Index = 26;
	clearances[5].point4Index = 27;
	clearances[5].SourceIndex = 1;

	offlimits[1].point1Index = 28;
	offlimits[1].point2Index = 29;
	offlimits[1].point3Index = 30;
	offlimits[1].point4Index = 31;
	offlimits[1].SourceIndex = 1;

	clearances[6].point1Index = 32;
	clearances[6].point2Index = 33;
	clearances[6].point3Index = 34;
	clearances[6].point4Index = 35;
	clearances[6].SourceIndex = 2;

	clearances[7].point1Index = 36;
	clearances[7].point2Index = 37;
	clearances[7].point3Index = 38;
	clearances[7].point4Index = 39;
	clearances[7].SourceIndex = 2;

	clearances[8].point1Index = 40;
	clearances[8].point2Index = 41;
	clearances[8].point3Index = 42;
	clearances[8].point4Index = 43;
	clearances[8].SourceIndex = 2;

	offlimits[2].point1Index = 44;
	offlimits[2].point2Index = 45;
	offlimits[2].point3Index = 46;
	offlimits[2].point4Index = 47;
	offlimits[2].SourceIndex = 2;

	clearances[9].point1Index = 48;
	clearances[9].point2Index = 49;
	clearances[9].point3Index = 50;
	clearances[9].point4Index = 51;
	clearances[9].SourceIndex = 3;

	clearances[10].point1Index = 52;
	clearances[10].point2Index = 53;
	clearances[10].point3Index = 54;
	clearances[10].point4Index = 55;
	clearances[10].SourceIndex = 3;

	clearances[11].point1Index = 56;
	clearances[11].point2Index = 57;
	clearances[11].point3Index = 58;
	clearances[11].point4Index = 59;
	clearances[11].SourceIndex = 3;

	offlimits[3].point1Index = 60;
	offlimits[3].point2Index = 61;
	offlimits[3].point3Index = 62;
	offlimits[3].point4Index = 63;
	offlimits[3].SourceIndex = 3;

	clearances[12].point1Index = 64;
	clearances[12].point2Index = 65;
	clearances[12].point3Index = 66;
	clearances[12].point4Index = 67;
	clearances[12].SourceIndex = 4;

	clearances[13].point1Index = 68;
	clearances[13].point2Index = 69;
	clearances[13].point3Index = 70;
	clearances[13].point4Index = 71;
	clearances[13].SourceIndex = 4;

	clearances[14].point1Index = 72;
	clearances[14].point2Index = 73;
	clearances[14].point3Index = 74;
	clearances[14].point4Index = 75;
	clearances[14].SourceIndex = 4;

	offlimits[4].point1Index = 76;
	offlimits[4].point2Index = 77;
	offlimits[4].point3Index = 78;
	offlimits[4].point4Index = 79;
	offlimits[4].SourceIndex = 4;

	positionAndRotation cfg[N];
	for (int i = 0; i < N; i++) {
		cfg[i].x = -6.4340348243713379;
		cfg[i].y = -2.12361741065979;
		cfg[i].z = 0.0;
		cfg[i].rotX = 0.0;
		cfg[i].rotY = 5.5179219245910645;
		cfg[i].rotZ = 0.0;
		cfg[i].frozen = false;
		cfg[i].length = 1.7677083015441895;
		cfg[i].width = 2.2480521202087402;
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

	gpuCfg.gridxDim = 10;
	gpuCfg.gridyDim = 0;
	gpuCfg.blockxDim = 256;
	gpuCfg.blockyDim = 0;
	gpuCfg.blockzDim = 0;
	gpuCfg.iterations = 200;

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