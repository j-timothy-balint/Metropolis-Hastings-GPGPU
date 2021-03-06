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

struct relationshipAngleStruct{
	double angleMin;
	double angleMax;
	int SourceIndex;
	int TargetIndex;
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

//Sets up an inital pseudo-random state based on the experiment (which is the thread id and seed) and the sequence
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
__device__ double theta(float xi, float yi, float xj, float yj, float ti){
	double dX = xi - xj;
	double dY = yi - yj;
	double theta_p = atan2(dY,dX); //gives us the angle between -PI and PI
	
	//and now between 0 and 2*pi
	theta_p = (theta_p < 0)?2*PI+theta_p:theta_p;
	//printf("theta_p=%f,ti=%f\n",theta_p,ti);
	//return the re-oriented angle
	double theta = theta_p - ti;
	return (theta < 0)?2*PI+theta:theta;

}

// Theta is the rotation
__device__ float phi(float xi, float yi, float xj, float yj, float tj)
{
	return atan2(yi - yj, xi - xj) - tj + PI / 2.0;
}

//This visual principle places the distibution of visual weight (perceptual salency, based on size) in the middle of the room
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

//This functional principle uses a lookup (relationshipStruct) to determine weights from recommended distances
__device__ double PairWiseCosts(Surface *srf, positionAndRotation* cfg, relationshipStruct *rs)
{
	double result = 0;
	for (int i = 0; i < srf->nRelationships; i++)
	{
		// Look up source index from relationship and retrieve object using that index.
		double distance = Distance(cfg[rs[i].SourceIndex].x, cfg[rs[i].SourceIndex].y, cfg[rs[i].TargetIndex].x, cfg[rs[i].TargetIndex].y);
		//printf("Distance: %f Range start: %f Range end: %f\n", distance, rs[i].TargetRange.targetRangeStart, rs[i].TargetRange.targetRangeEnd);
		//penalize if we are too close
		if (distance < rs[i].TargetRange.targetRangeStart)
		{
			double fraction = distance / rs[i].TargetRange.targetRangeStart;
			result -= (fraction * fraction);
		}
		//penalize if we are too far
		else if (distance > rs[i].TargetRange.targetRangeEnd) 
		{
			double fraction = rs[i].TargetRange.targetRangeEnd / distance;
			result -= (fraction * fraction);
		}
		// Else don't do anything as 0 indicates a perfect solution
	}
	return result;
}

//This functional principle uses a lookup (relationshipStruct) to determine weights from a recommended angle
__device__ double PairWiseAngleCosts(Surface *srf, positionAndRotation* cfg, relationshipAngleStruct *rs)
{
	double result = 0;
	//assuming (0,2*PI]
	for (int i = 0; i < srf->nRelationships; i++)
	{
		// We use phi to calculate the angle between the rotation of the object and the target object
		double distance = theta(cfg[rs[i].SourceIndex].x, cfg[rs[i].SourceIndex].y, cfg[rs[i].TargetIndex].x, cfg[rs[i].TargetIndex].y,cfg[rs[i].TargetIndex].rotY);
		//printf("dist is %f, angle is %f\n",distance,cfg[rs[i].TargetIndex].rotY);
		if(rs[i].angleMin > rs[i].angleMax){
			double norm = (2*PI - (rs[i].angleMax +(2*PI - rs[i].angleMin)))/2.0;
			//In this case, we need to determine a slightly different range because we are crossing the zero boundary
			if(fmodf(rs[i].angleMin+distance, 2*PI) > rs[i].angleMax)
				result -= fmin(fabs(distance - rs[i].angleMin),fabs(distance-rs[i].angleMax))/norm; //Calculate the closest angular difference (un-normalized for now)
		}
		else if(rs[i].angleMin < distance || distance < rs[i].angleMax){
			double norm = (2*PI - (rs[i].angleMax - rs[i].angleMin))/2.0; //The max distance away is half the slice that is in the no zone 
			result -= fmin(fabs(distance - rs[i].angleMin),fabs(distance-rs[i].angleMax))/norm; //Calculate the closest angular difference (un-normalized for now)
		}
		//Stick to zero as the perfect solution
		//by doing percent error as an absolute value, we can go around the circle. Whichever bound is closer to is the one we want
		//result -= 1.0f - fmaxf( 
		//				       fabsf((distance - rs[i].angleMin)/rs[i].angleMin), 
		//					   fabsf((distance - rs[i].angleMax)/rs[i].angleMax)
		//					   );
	}
	return result;
}

//This visual criteria (emphasis in the paper) creates a dominant point in the room
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
//This visual criteria (emphasis in the paper) causes the system to focus on similar grouping symmetry 
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

//This function helper function is used to calculate the clearence of a room
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

//Helper function for calculating the clearance
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
		for (int j = 0; j < srf->nObjs; j++) {
			// Determine max and min vectors of clearance rectangles
			// rectangle #1
			//printf("Clearance\n");
			//printf("Translation: X: %f Y: %f\n", cfg[i].x, cfg[i].y);
			vertex rect1Min = minValue(vertices, clearances[i].point1Index, cfg[clearances[i].SourceIndex].x, cfg[clearances[i].SourceIndex].y);
			vertex rect1Max = maxValue(vertices, clearances[i].point1Index, cfg[clearances[i].SourceIndex].x, cfg[clearances[i].SourceIndex].y);

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

__device__ void Costs(Surface *srf, resultCosts* costs, positionAndRotation* cfg, relationshipStruct *rs, relationshipAngleStruct *ra, vertex *vertices, rectangle *clearances, rectangle *offlimits, vertex *surfaceRectangle)
{
	float pairWiseCosts = PairWiseCosts(srf, cfg, rs) *PairWiseAngleCosts(srf,cfg,ra);
	costs->PairWiseCosts = srf->WeightPairWise * pairWiseCosts;
	//printf("Pair wise costs  %f\n", pairWiseCosts);

	float visualBalanceCosts =  VisualBalanceCosts(srf, cfg);
	costs->VisualBalanceCosts = srf->WeightVisualBalance * visualBalanceCosts;
	//printf("Visual balance costs  %f\n", visualBalanceCosts);

	float focalPointCosts =  FocalPointCosts(srf, cfg);
	costs->FocalPointCosts = srf->WeightFocalPoint * focalPointCosts;
	//printf("Focal point costs %f\n", focalPointCosts);

	float symmertryCosts = SymmetryCosts(srf, cfg);
	costs->SymmetryCosts = srf->WeightSymmetry * symmertryCosts;
	//printf("Symmertry costs  %f\n", symmertryCosts);

	float offlimitsCosts =  OffLimitsCosts(srf, cfg, vertices, offlimits);
	//printf("OffLimits costs  %f\n", offlimitsCosts);
	costs->OffLimitsCosts =  srf->WeightOffLimits * offlimitsCosts;

	float clearanceCosts =  ClearanceCosts(srf, cfg, vertices, clearances, offlimits);
	//printf("Clearance costs %f\n", clearanceCosts);
	costs->ClearanceCosts = srf->WeightClearance * clearanceCosts;

	float surfaceAreaCosts =  SurfaceAreaCosts(srf, cfg, vertices, clearances, offlimits, surfaceRectangle);
	//printf("Surface area costs %f\n", surfaceAreaCosts);
	costs->SurfaceAreaCosts = srf->WeightSurfaceArea * surfaceAreaCosts;

	//The total cost is summed as the negitive weighting terms like alignment and emphasis are distributed in those functions themselves
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
		// randomly choose an object
		int obj = generateRandomIntInRange(rngStates, tid, srf->nObjs-1, 0);

		// Potential never ending loop when everything is frozen
		while (cfgStar[obj].frozen)
			obj = generateRandomIntInRange(rngStates, tid, srf->nObjs-1, 0);

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
		int obj = generateRandomIntInRange(rngStates, tid, srf->nObjs-1, 0);
		while (cfgStar[obj].frozen)
			obj = generateRandomIntInRange(rngStates, tid, srf->nObjs-1, 0);
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
		int obj1 = generateRandomIntInRange(rngStates, tid, srf->nObjs-1, 0);
		while (cfgStar[obj1].frozen)
			obj1 = generateRandomIntInRange(rngStates, tid, srf->nObjs-1, 0);

		int obj2 = generateRandomIntInRange(rngStates, tid, srf->nObjs-1, 0);
		while (cfgStar[obj2].frozen)
			obj2 = generateRandomIntInRange(rngStates, tid, srf->nObjs-1, 0);
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

//Copy now uses the thread id in the block (or group) to copy in a parralell manner
__device__ void Copy(positionAndRotation* cfg1, positionAndRotation* cfg2, Surface* srf) 
{
	if (blockDim.x > srf->nObjs) { //We have more workers than work, so some will be idle
		for (unsigned int i = threadIdx.x; i < srf->nObjs; i++)
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
	else {
		int breakup = srf->nObjs / blockDim.x;
		for (unsigned int i = threadIdx.x; i < srf->nObjs; i+=breakup)
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
	__syncthreads();
}

// result is a [,] array with 1 dimension equal to the amount of blocks used and the other dimension equal to the amount of objects
// rs is an array with the length equal to the amount of relationships
// cfg is an array with the length equal to the amount of objects
// Surface is a basic struct
__global__ void Kernel(resultCosts* resultCostsArray, point *p, relationshipStruct *rs,relationshipAngleStruct *ra, positionAndRotation* cfg, rectangle *clearances, rectangle *offlimits, vertex *vertices, vertex *surfaceRectangle, Surface *srf, gpuConfig *gpuCfg, curandState *rngStates)
{
//    printf("current block [%d, %d]:\n",\
//            blockIdx.y*gridDim.x+blockIdx.x,\
//            threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x);
	// Calculate current tid
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// Retrieve local state from rng states
	//printf("test random number 1: %f\n", curand_uniform(&rngStates[tid]));
	
	// Initialize current configuration
	__shared__ positionAndRotation* cfgCurrent;
	__shared__ resultCosts* currentCosts;
	__shared__ positionAndRotation* cfgStar;
	__shared__ resultCosts* starCosts;
	__syncthreads();
	if (threadIdx.x == 0) { //Note, we can remove these mallocs by dynamic allocation in the kernel, so they wouldn't be allocated on the stack
		cfgCurrent   = (positionAndRotation*)malloc(srf->nObjs * sizeof(positionAndRotation));
		currentCosts = (resultCosts*)malloc(sizeof(resultCosts));
		cfgStar      = (positionAndRotation*)malloc(srf->nObjs * sizeof(positionAndRotation));
		starCosts    = (resultCosts*)malloc(sizeof(resultCosts));
	}
	__syncthreads();
	Copy(cfgCurrent, cfg, srf);
	Costs(srf, currentCosts, cfgCurrent, rs, ra, vertices, clearances, offlimits, surfaceRectangle);
	//positionAndRotation* cfgBest = (positionAndRotation*)malloc(srf->nObjs * sizeof(positionAndRotation));
	//Copy(cfgBest, cfgCurrent, srf);
	//resultCosts* bestCosts = (resultCosts*)malloc(sizeof(resultCosts));
	//CopyCosts(currentCosts, bestCosts);
	//printf("Threadblock: %d, Best costs before: %f\n", blockIdx.x, bestCosts->totalCosts);
	// Create cfg Star and initialize it to cfgcurrent that will have a proposition done to it.
	for (int i = 0; i < gpuCfg->iterations; i++)
	{

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

		Costs(srf, starCosts, cfgStar, rs, ra, vertices, clearances, offlimits, surfaceRectangle);
		// printf("Cost star configuration: %f\n", starCosts->totalCosts);
		// printf("Cost best configuration: %f\n", bestCosts->totalCosts);
		// star has a better cost function than best cost, star is the new best
		/*if (starCosts->totalCosts < bestCosts->totalCosts)
		{
			//printf("New best %f\n", bestCosts->totalCosts);

			// Copy star into best for storage
			Copy(cfgBest, cfgStar, srf);
			CopyCosts(starCosts, bestCosts);
			//printf("Threadblock: %d, New costs before: %f\n", blockIdx.x, bestCosts->totalCosts);
		}*/

		// Check whether we continue with current or we continue with star
		if (Accept(starCosts->totalCosts, currentCosts->totalCosts, rngStates, tid))
		{
			// Possible different approach: Set pointer of current to star, free up memory used by current? reinitialize star?
			//printf("Star accepted as new current.\n");
			// Copy star into current
			Copy(cfgCurrent, cfgStar, srf);
			CopyCosts(starCosts, currentCosts);
		}

	}
	//Copy(cfgBest, cfgCurrent, srf);
	//CopyCosts(currentCosts, bestCosts);
	__syncthreads();
	
	// Copy best config (now set to input config) to result of this block
	for (unsigned int i = threadIdx.x; i < srf->nObjs; i+=blockDim.x)
	{
		int index = blockIdx.x * srf->nObjs + i;
		p[index].x = cfgCurrent[i].x;
		p[index].y = cfgCurrent[i].y;
		p[index].z = cfgCurrent[i].z;
		p[index].rotX = cfgCurrent[i].rotX;
		p[index].rotY = cfgCurrent[i].rotY;
		p[index].rotZ = cfgCurrent[i].rotZ;
		// BlockId counts from 0, so to properly multiply
		/*p[index].x = cfgBest[i].x;
		p[index].y = cfgBest[i].y;
		p[index].z = cfgBest[i].z;
		p[index].rotX = cfgBest[i].rotX;
		p[index].rotY = cfgBest[i].rotY;
		p[index].rotZ = cfgBest[i].rotZ;*/
	}
	//printf("Threadblock: %d, Result costs before: %f\n", blockIdx.x, bestCosts->totalCosts);
	/*resultCostsArray[blockIdx.x].totalCosts = bestCosts->totalCosts;
	resultCostsArray[blockIdx.x].PairWiseCosts = bestCosts->PairWiseCosts;
	resultCostsArray[blockIdx.x].VisualBalanceCosts = bestCosts->VisualBalanceCosts;
	resultCostsArray[blockIdx.x].FocalPointCosts = bestCosts->FocalPointCosts;
	resultCostsArray[blockIdx.x].SymmetryCosts = bestCosts->SymmetryCosts;
	//printf("Best surface area costs: %f\n", bestCosts->SurfaceAreaCosts);
	resultCostsArray[blockIdx.x].SurfaceAreaCosts = bestCosts->SurfaceAreaCosts;
	//printf("Best clearance costs: %f\n", bestCosts->ClearanceCosts);
	resultCostsArray[blockIdx.x].ClearanceCosts = bestCosts->ClearanceCosts;
	resultCostsArray[blockIdx.x].OffLimitsCosts = bestCosts->OffLimitsCosts;*/
	__syncthreads();
	if (threadIdx.x == 0) { //Note, we can remove these mallocs by dynamic allocation in the kernel, so they wouldn't be allocated on the stack
		free(cfgCurrent);
		free(currentCosts);
		free(cfgStar);
		free(starCosts);
	}
	//free(cfgBest);
	//free(bestCosts);
}

extern "C" __declspec(dllexport) result* KernelWrapper(relationshipStruct *rss,relationshipAngleStruct *rsa, positionAndRotation* cfg, rectangle *clearances, rectangle *offlimits, vertex *vertices, vertex *surfaceRectangle, Surface *srf, gpuConfig *gpuCfg)
{
	// Create pointer for on gpu
	// Determine memory size of object to transfer
	// Malloc on GPU size
	// Cpy memory from cpu to gpu
	relationshipStruct *gpuRS;
	int rsSize = sizeof(relationshipStruct) * srf->nRelationships;
	checkCudaErrors(cudaMalloc(&gpuRS, rsSize));
	checkCudaErrors(cudaMemcpy(gpuRS, rss, rsSize, cudaMemcpyHostToDevice));

	relationshipAngleStruct *gpuRA;
	int rsaSize = sizeof(relationshipAngleStruct) * srf->nRelationships;
	checkCudaErrors(cudaMalloc(&gpuRA, rsaSize));
	checkCudaErrors(cudaMemcpy(gpuRA, rsa, rsaSize, cudaMemcpyHostToDevice));

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

	// Initialise random number generator as the grids and the blocks
	initRNG <<<gpuCfg->gridxDim, gpuCfg->blockxDim >> > (d_rngStates, time(NULL));

	// Commented for possible later usage
	// dim3 dimGrid(gpuCfg->gridxDim, gpuCfg->gridyDim);
	// dim3 dimBlock(gpuCfg->blockxDim, gpuCfg->blockyDim, gpuCfg->blockzDim);
	
	// Block 1 dimensional, amount of threads available, configurable
	// Grid 1 dimension, amount of suggestions to be made.
	Kernel <<<gpuCfg->gridxDim, gpuCfg->blockxDim >>>(gpuResultCosts, gpuPointArray, gpuRS, gpuRA, gpuAlgoCFG, gpuClearances, gpuOfflimits, gpuVertices, gpuSurfaceRectangle, gpuSRF, gpuGpuConfig, d_rngStates);
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

	const int N = 32;
	const int NRel = 1;
	const int NClearances = 2;
	Surface srf;
	srf.nObjs = N;
	srf.nRelationships = NRel;
	srf.nClearances = NClearances;
	srf.WeightFocalPoint = -2.0f;
	srf.WeightPairWise = -2.0f;
	srf.WeightVisualBalance = 1.5f;
	srf.WeightSymmetry = -2.0;
	srf.WeightClearance = -2.0;
	srf.WeightSurfaceArea = -2.0;
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


	vertex vtx[16];
	// Clearance shapes
	vtx[0].x = 2;
	vtx[0].y = 2;
	vtx[0].z = 0;

	vtx[1].x = 2;
	vtx[1].y = 0;
	vtx[1].z = 0;

	vtx[2].x = 0;
	vtx[2].y = 0;
	vtx[2].z = 0;

	vtx[3].x = 0;
	vtx[3].y = 2;
	vtx[3].z = 0;

	vtx[4].x = 3;
	vtx[4].y = 2;
	vtx[4].z = 0;

	vtx[5].x = 3;
	vtx[5].y = 0;
	vtx[5].z = 0;

	vtx[6].x = 1;
	vtx[6].y = 0;
	vtx[6].z = 0;

	vtx[7].x = 1;
	vtx[7].y = 2;
	vtx[7].z = 0;

	// Off limits
	vtx[8].x = 2;
	vtx[8].y = 2;
	vtx[8].z = 0;

	vtx[9].x = 2;
	vtx[9].y = 0;
	vtx[9].z = 0;

	vtx[10].x = 0;
	vtx[10].y = 0;
	vtx[10].z = 0;

	vtx[11].x = 0;
	vtx[11].y = 2;
	vtx[11].z = 0;

	vtx[12].x = 3;
	vtx[12].y = 2;
	vtx[12].z = 0;

	vtx[13].x = 3;
	vtx[13].y = 0;
	vtx[13].z = 0;

	vtx[14].x = 1;
	vtx[14].y = 0;
	vtx[14].z = 0;

	vtx[15].x = 1;
	vtx[15].y = 2;
	vtx[15].z = 0;

	rectangle clearances[NClearances];
	clearances[0].point1Index = 0;
	clearances[0].point2Index = 1;
	clearances[0].point3Index = 2;
	clearances[0].point4Index = 3;
	clearances[0].SourceIndex = 0;

	clearances[1].point1Index = 4;
	clearances[1].point2Index = 5;
	clearances[1].point3Index = 6;
	clearances[1].point4Index = 7;
	clearances[1].SourceIndex = 1;

	rectangle offlimits[N];
	for (int i = 0; i < N; i++) {
		if (i % 2 == 0) {
			offlimits[i].point1Index = 8;
			offlimits[i].point2Index = 9;
			offlimits[i].point3Index = 10;
			offlimits[i].point4Index = 11;
			offlimits[i].SourceIndex = 0;
		}
		else {
			offlimits[i].point1Index = 12;
			offlimits[i].point2Index = 13;
			offlimits[i].point3Index = 14;
			offlimits[i].point4Index = 15;
			offlimits[i].SourceIndex = 1;
		}
	}
	positionAndRotation cfg[N];
	for (int i = 0; i < N; i++) {
		cfg[i].x = i * 2.0;
		cfg[i].y = i * 2.0;
		cfg[i].z = 0.0;
		cfg[i].rotX = 0.0;
		cfg[i].rotY = 0.0;
		cfg[i].rotZ = 0.0;
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

	relationshipAngleStruct rsa[1];
	rsa[0].angleMin = PI/4;
	rsa[0].angleMax = 5*PI/8;
	rsa[0].SourceIndex = 0;
	rsa[0].TargetIndex = 1;
	printf("Target angles are (%f,%f)\n",rsa[0].angleMin,rsa[0].angleMax);

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
	gpuCfg.blockxDim = 64;
	gpuCfg.blockyDim = 0;
	gpuCfg.blockzDim = 0;
	gpuCfg.iterations = 100;

	// Point test code:

	result *result = KernelWrapper(rss, rsa, cfg, clearances, offlimits, vtx, surfaceRectangle, &srf, &gpuCfg);
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
	char hold;
	scanf("%c",&hold);
 	return EXIT_SUCCESS;
}