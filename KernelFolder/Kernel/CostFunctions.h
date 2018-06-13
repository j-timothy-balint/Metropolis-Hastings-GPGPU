#pragma once
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

// Trig constants:
#define PI (3.1416)

#define GAUSS_PI (2.5066)

//#define BETA (0.1)
// Right angle constants:
#define THETA_R (45.0 / 180.0 * PI) // 45 degrees

// Sampling constants:
#define S_SIGMA_P (0.8)
#define S_SIGMA_T (15.0 / 90.0 * PI)

//In the original implementation, number of threads in a group was set to the WARP size, which we can do with 32
#define WARP_SIZE 32

__inline__ __device__ float Distance(float xi, float yi, float xj, float yj)
{
	return hypotf(xi - xj, yi - yj);
}

//Determines the angular difference between two objects where i is oriented to j (i is bearing to j)
__device__ float theta(float xi, float yi, float xj, float yj, float ti) {
	float dX = xi - xj;
	float dY = yi - yj;
	float theta_p = atan2f(dY, dX); //gives us the angle between -PI and PI

									//and now between 0 and 2*pi
	theta_p = (theta_p < 0) ? 2 * PI + theta_p : theta_p;
	//printf("theta_p=%f,ti=%f\n",theta_p,ti);
	//return the re-oriented angle
	float theta = theta_p - ti;
	return (theta < 0) ? 2 * PI + theta : theta;

}

// Tj is the rotation
__inline__ __device__ float phi(float xi, float yi, float xj, float yj, float tj)
{
	return atan2f(yi - yj, xi - xj) - tj + PI / 2.0;
}

__inline__ __device__ float gaussian(float val, float mean, float deviation) {
	//We don't worry about where the deviation puts it. What we want is a gaussian with an aplitude of -1 at the mean
	float x = (val - mean) / deviation;
	return expf(-0.5f *x*x);// / (deviation *GAUSS_PI);
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

template<int tile_sz>
__device__ float VisualBalanceCosts(cg::thread_block_tile<tile_sz> group, Surface *srf, point* cfg)
{
	int tid = group.thread_rank();
	int step = group.size();
	float nx;// = 0;
	float ny;// = 0;
	float denom;// = 0;
				//because of multiple share blocks, we do an atomic add instead of the reduce method
	nx = 0.0;
	ny = 0.0;
	denom = 0.0;
	for (int i = tid; i < srf->nObjs; i += step)
	{
		float area = cfg[i].length * cfg[i].width;
		nx += area * cfg[i].freedom[0];
		ny += area * cfg[i].freedom[1];
		denom += area;
	}
	group.sync();
	nx = reduce<tile_sz>(group, nx);
	ny = reduce<tile_sz>(group, ny);
	denom = reduce<tile_sz>(group, denom);
	// Distance between all summed areas and points divided by the areas and the room's centroid


	return  Distance(nx / denom, ny / denom, srf->centroidX, srf->centroidY); //Because we are all reducing, all values should be the same
}

template<int tile_sz>
__device__ float AlignmentCosts(cg::thread_block_tile<tile_sz> group, Group* groups, Surface *srf, point* cfg) {
	int tid = group.thread_rank();
	int step = group.size();
	float costs = 0.0;
	for (int i = tid; i < srf->nGroups; i += step) { //Eventually, this will transform into groups
			costs += cosf(4 * cfg[groups[i].SourceIndex].freedom[4] - cfg[groups[i].TargetIndex].freedom[4]);
	}
	group.sync();
	costs = reduce<tile_sz>(group, costs);
	return -costs;
}

//Euclidean distance is useful for 
template<int tile_sz>
__device__ float PairWiseEuclidean(cg::thread_block_tile<tile_sz> group, Surface *srf, point* cfg, relationshipStruct *rs, int freedom, int source_index)
{
	int tid = group.thread_rank();
	int step = group.size();
	float values;
	values = 0.0f; //Since it's size blockDim, we can have each of them treat it as the starting value
	if (srf->pwChoices[0].total) {
		for (int i= tid; i < srf->nRelationships; i += step)
		{
			// Look up source index from relationship and retrieve object using that index.
			float distance = Distance(cfg[rs[i].SourceIndex].freedom[0], cfg[rs[i].SourceIndex].freedom[1], cfg[rs[i].TargetIndex].freedom[0], cfg[rs[i].TargetIndex].freedom[1]);
			//printf("Distance: %f Range start: %f Range end: %f\n", distance, rs[i].TargetRange.targetRangeStart, rs[i].TargetRange.targetRangeEnd);
			//penalize if we are too close
			//Score distance calculation
			float score = (distance < rs[i].TargetRange.targetRangeStart) ? powf(distance / rs[i].TargetRange.targetRangeStart, rs[i].DegreesOfAtrraction) : 1.0;
			score = (distance > rs[i].TargetRange.targetRangeEnd) ? powf(rs[i].TargetRange.targetRangeEnd / distance, rs[i].DegreesOfAtrraction) : score;
			values -= score;
		}
	}
	else {
		for (int i = tid; i < srf->nRelationships; i += step)
		{
			// Look up source index from relationship and retrieve object using that index.
			float distance = fabs(cfg[rs[i].SourceIndex].freedom[freedom]-cfg[rs[i].TargetIndex].freedom[freedom]);//Distance(cfg[rs[i].SourceIndex].freedom[0], cfg[rs[i].SourceIndex].freedom[1], cfg[rs[i].TargetIndex].freedom[0], cfg[rs[i].TargetIndex].freedom[1]);
			//printf("Distance: %f Range start: %f Range end: %f\n", distance, rs[i].TargetRange.targetRangeStart, rs[i].TargetRange.targetRangeEnd);
			//penalize if we are too close
			//Score distance calculation
			float score = (distance < rs[i].TargetRange.targetRangeStart) ? powf(distance / rs[i].TargetRange.targetRangeStart, rs[i].DegreesOfAtrraction) : 1.0;
			score = (distance > rs[i].TargetRange.targetRangeEnd) ? powf(rs[i].TargetRange.targetRangeEnd / distance, rs[i].DegreesOfAtrraction) : score;
			values -= score;
		}
	}
	group.sync();
	values = reduce<tile_sz>(group, values);
	//printf("Clearance costs error: %f\n", error);
	return values;
}

//This functional principle uses a lookup (relationshipStruct) to determine weights from a recommended angle
//This is not the facing angle but the distance angle (so, the target rotated around the source)
template<int tile_sz>
__device__ float PairWiseAngle(cg::thread_block_tile<tile_sz> tb, Surface *srf, point* cfg, relationshipStruct *rs)
{
	int tid = tb.thread_rank();
	int step = tb.size();

	int i = tid; //Change when at nAngle
	int end = srf->nAngleRelationships;
	//Determines our bound on the array
	if (!srf->pwChoices[0].gaussian) {
		i += srf->nRelationships;
		end += srf->nRelationships;
	}

	float values = 0.0f; //Since it's size blockDim, we can have each of them treat it as the starting value
						 //assuming (0,2*PI]
	for (i; i < end; i += step)
	{
		// We use phi to calculate the angle between the rotation of the object and the target object
		float angle = theta(cfg[rs[i].SourceIndex].freedom[0], cfg[rs[i].SourceIndex].freedom[1], cfg[rs[i].TargetIndex].freedom[0], cfg[rs[i].TargetIndex].freedom[1], cfg[rs[i].TargetIndex].freedom[4]);
		//For now, we assume start is greater than end
		float norm = fabs(rs[i].TargetRange.targetRangeEnd - rs[i].TargetRange.targetRangeStart) / 2.0; //The max distance away is half the slice that is in the no zone 
		norm = (2.0 * PI - norm) / 2.0;
		values -= (rs[i].TargetRange.targetRangeEnd < angle || angle < rs[i].TargetRange.targetRangeEnd) ? fminf(fabsf(angle - rs[i].TargetRange.targetRangeStart),
			fabsf(angle - rs[i].TargetRange.targetRangeEnd)) / norm : 1.0;
	}
	tb.sync();
	values = reduce<tile_sz>(tb, values);
	//printf("Clearance costs error: %f\n", error);
	return values;
}

template<int tile_sz>
__device__ float PairWiseGaussian(cg::thread_block_tile<tile_sz> group, Surface* srf, point* cfg, gaussianRelationshipStruct *rs, int freedom,int source_index,int pwc) {
	int tid = group.thread_rank();
	int step = group.size();
	float values;
	values = 0.0f; //Since it's size blockDim, we can have each of them treat it as the starting value
	int i = tid; //Change when at nAngle
	int end = srf->nAngleRelationships;
	//Determines our bound on the array
	if (pwc == 0) {
		end = srf->nRelationships;
	}
	else if (pwc == 1 && srf->pwChoices[0].gaussian) {
		i += srf->nRelationships; 
		end += srf->nRelationships;
	}
	
	if (srf->pwChoices[i].total) {
		for (i; i < end; i += step) {
			float score = 1.0;
			for (int j = 0; j < 3; j++) {
				float dist = fabsf(cfg[rs[i].SourceIndex].freedom[j] - cfg[rs[i].TargetIndex].freedom[j]); //squared root of squared is bascially abs for 1-D shapes, this is much faster
				score *= gaussian(dist, rs[i].mean[j], rs[i].deviation[j]);
			}
			values -= score;
		}
	}
	else {
		for (i; i < end; i += step)
		{
			// Look up source index from relationship and retrieve object using that index.
			float dist = fabsf(cfg[rs[i].SourceIndex].freedom[freedom] - cfg[rs[i].TargetIndex].freedom[freedom]); //squared root of squared is bascially abs for 1-D shapes, this is much faster
			float score = gaussian(dist, rs[i].mean[source_index], rs[i].deviation[source_index]);
			values -= score;
		}
	}
	group.sync();
	values = reduce<tile_sz>(group, values);
	return values;
}


template<int tile_sz>
__device__ float FocalPointCosts(cg::thread_block_tile<tile_sz> group, Surface *srf, point* cfg)
{
	int tid = group.thread_rank();
	int step = group.size();
	double values;
	values = 0.0f; //Since it's size blockDim, we can have each of them treat it as the starting value
	for (int i = tid; i < srf->nObjs; i += step)
	{
		float phi_fi = phi(srf->focalX, srf->focalY, cfg[i].freedom[0], cfg[i].freedom[1], cfg[i].freedom[4]);
		// Old implementation of grouping, all objects that belong to the seat category are used in the focal point calculation
		// For now we default to all objects, focal point grouping will come later
		//int s_i = s(r.c[i]);

		// sum += s_i * cos(phi_fi);
		values += cosf(phi_fi);
	}
	group.sync();
	values = reduce<tile_sz>(group, values);
	//printf("tid = %d, value = %f\n", tid, values[tid]);
	//printf("Clearance costs error: %f\n", error);
	return values;
}

template<int tile_sz>
__device__ float SymmetryCosts(cg::thread_block_tile<tile_sz> group, Surface *srf, point* cfg)
{
	int tid = group.thread_rank();
	int step = group.size();
	float values = 0.0;
	for (int i = tid; i < srf->nObjs; i += step)
	{
		float maxVal = 0;

		float ux = cosf(srf->focalRot);
		float uy = sinf(srf->focalRot);
		float s = 2 * (srf->focalX * ux + srf->focalY * uy - (cfg[i].freedom[0] * ux + cfg[i].freedom[1] * uy));  // s = 2 * (f * u - v * u)

																												  // r is the reflection of g across the symmetry axis defined by p.
		float rx_i = cfg[i].freedom[0] + s * ux;
		float ry_i = cfg[i].freedom[1] + s * uy;
		float rRot_i = 2 * srf->focalRot - cfg[i].freedom[4];
		if (rRot_i < -PI)
			rRot_i += 2 * PI;

		for (int j = 0; j < srf->nObjs; j++)
		{
			// Types should be the same, this probably works great with their limited amount of types but will probably not work that great for us. Perhaps define a group?
			int gamma_ij = 1;
			float dp = Distance(cfg[j].freedom[0], cfg[j].freedom[1], rx_i, ry_i);
			float dt = cfg[j].freedom[4] - rRot_i;
			if (dt > PI)
				dt -= 2 * PI;

			float val = gamma_ij * (5 - sqrt(dp) - 0.4 * fabs(dt));
			maxVal = fmaxf(maxVal, val);
		}

		values -= maxVal;
	}

	group.sync();
	values = reduce<tile_sz>(group, values);
	//printf("Clearance costs error: %f\n", error);
	return values;
}

//Calculates the bounding box for a given configuration based on it's rotation, and returns the min and max points
__inline__ __device__ void calculateBoundingBox(point* cfg, int i, BoundingBox* rect) {
	float cos_t = fabsf(cosf(cfg[i].freedom[4])); //only want the percent of effect, so we do a fabs
	float sin_t = fabsf(sinf(cfg[i].freedom[4]));
	float dx = (cfg[i].width * cos_t + cfg[i].length * sin_t) / 2.0;
	float dy = (cfg[i].width * sin_t + cfg[i].length * cos_t) / 2.0;
	float dz = (cfg[i].height)/2.0;
	rect->minPoint.x = cfg[i].freedom[0] - dx;
	rect->minPoint.y = cfg[i].freedom[1] - dy;
	rect->maxPoint.x = cfg[i].freedom[0] + dx;
	rect->maxPoint.y = cfg[i].freedom[1] + dy;
	rect->minPoint.z = cfg[i].freedom[2] - dz;
	rect->maxPoint.z = cfg[i].freedom[2] + dz;

}

//Calculates the bounding box for a given set of four vertices and a rotation
__inline__ __device__ void calculateBoundingBox(vertex* verts, int i, float theta, BoundingBox* rect) {
	float cos_t = fabsf(cosf(theta)); //only want the percent of effect, so we do a fabs
	float sin_t = fabsf(sinf(theta));
	//At this point, we aren't concerned with the doing the calculations on the convex hull, but on a fitted box 
	vertex min;
	min.x = verts[i].x;
	min.y = verts[i].y;
	min.z = verts[i].z;
	vertex max;
	max.x = verts[i].x;
	max.y = verts[i].y;
	max.z = verts[i].z;
#pragma unroll
	for (int j = 1; j < 4; j++) { //Can probably do this better, but it's simple enough
		min.x = (min.x > verts[i + j].x) ? verts[i + j].x : min.x;
		min.y = (min.y > verts[i + j].y) ? verts[i + j].y : min.y;
		min.z = (min.z > verts[i + j].z) ? verts[i + j].z : min.z;
		max.x = (max.x < verts[i + j].x) ? verts[i + j].x : max.x;
		max.y = (max.y < verts[i + j].y) ? verts[i + j].y : max.y;
		max.z = (max.z < verts[i + j].z) ? verts[i + j].z : max.z;
	}
	//printf("Min is (%f,%f) and Max is (%f,%f)\n", min.x, min.y, max.x, max.y);
	vertex cent;
	cent.x = (min.x + max.x) / 2;
	cent.y = (min.y + max.y) / 2;
	cent.z = (min.z + max.z) / 2;
	vertex size;
	size.x = max.x - min.x;
	size.y = max.y - min.y;
	size.z = max.z - min.z;
	float dx = (size.x * cos_t + size.y * sin_t) / 2.0; //Slightly easier than doing an affine and then calculating the min and max points
	float dy = (size.x * sin_t + size.y * cos_t) / 2.0;
	//dz is not affected by our theta calculation, so no cos_t, so we don't need to redo the calcuation
	//because we rotate around the projection, we need to do it this way
	rect->minPoint.x = cent.x - dx;
	rect->minPoint.y = cent.y - dy;
	rect->minPoint.z = cent.z - size.z;
	rect->maxPoint.x = cent.x + dx;
	rect->maxPoint.y = cent.y + dy;
	rect->maxPoint.z = cent.z + size.z;
}

//From Merrell et al: We use the projection of the item (found in our in bounding box) with a line segment (or disk, but for now a line segment)
//Since our item is defined as the min and max points on the bounding box, we define the mink sum as adding those projections together
__inline__ __device__ void MinkowskiSum(vertex *vertices, int i, float theta, BoundingBox* in, BoundingBox* out) {
	float dx = vertices[i].x * cos(theta) - vertices[i].y * sin(theta); //Our rotation matrix written out
	float dy = vertices[i].x * sin(theta) + vertices[i].y * cos(theta);
	out->minPoint.x = in->minPoint.x + dx;
	out->maxPoint.x = in->maxPoint.x + dx;
	out->minPoint.y = in->minPoint.y + dy;
	out->maxPoint.y = in->maxPoint.y + dy;
}

//From Merrell et al: We use the projection of the item (found in our in bounding box) with a line segment (or disk, but for now a line segment)
//Since our item is defined as the min and max points on the bounding box, we define the mink sum as adding those projections together
//We can also do this as the bounding box is the clearance box and the center is the line segment (which is what was sort of being done, but not really)
__inline__ __device__ void MinkowskiSum(vertex *line, BoundingBox* in, BoundingBox* out) {
	float dx = line->x; //Our rotation matrix written out
	float dy = line->y;
	float dz = line->z;
	out->minPoint.x = in->minPoint.x + dx;
	out->maxPoint.x = in->maxPoint.x + dx;
	out->minPoint.y = in->minPoint.y + dy;
	out->maxPoint.y = in->maxPoint.y + dy;
	out->minPoint.z = in->minPoint.z + dz;
	out->maxPoint.z = in->maxPoint.z + dz;
}


//Going to move the calculations from two vertices to a bounding box
__inline__ __device__  float calculateIntersectionArea(BoundingBox* rect1, BoundingBox* rect2) {
	// printf("Clearance rectangle 1: Min X: %f Y: %f Max X: %f Y: %f\n", rect1Min.x, rect1Min.y, rect1Max.x, rect1Max.y);
	// printf("Clearance rectangle 2: Min X: %f Y: %f Max X: %f Y: %f\n", rect2Min.x, rect2Min.y, rect2Max.x, rect2Max.y);
	// for each two rectangles, find out their intersection. Increase the error using the area
	float x5 = fmaxf(rect1->minPoint.x, rect2->minPoint.x);
	float y5 = fmaxf(rect1->minPoint.y, rect2->minPoint.y);
	float x6 = fminf(rect1->maxPoint.x, rect2->maxPoint.x);
	float y6 = fminf(rect1->maxPoint.y, rect2->maxPoint.y);
	float z5 = fmaxf(rect1->minPoint.z, rect2->minPoint.z);
	float z6 = fminf(rect1->maxPoint.z, rect2->maxPoint.z);
	//If one of these is exactly a zero area, then we have a complete overlap
	//printf("Clearance Total is (%f - %f,%f- %f,%f -%f)\n", x6 , x5, y6 , y5, z6 , z5);
	return fmaxf(0.0, x6 - x5)*fmaxf(0.0, y6 - y5) * fmaxf(0.0, z6 - z5); //This assume that my difference is positive, which it should be for it to be overlapping
}
//We are assigning, which is highly parallizable. So we shouldn't need the overhead of the function call
//Creates the surface area complement rectangle to figure out how much we are off the room
__inline__ __device__ void createComplementRectangle(BoundingBox* srfRect, BoundingBox* rectangles) {
	// 0 is min value, 1 is max value
	rectangles[0].minPoint.x = -FLT_MAX;
	rectangles[0].minPoint.y = -FLT_MAX;
	rectangles[0].maxPoint.x = FLT_MAX;
	rectangles[0].maxPoint.y = srfRect->minPoint.y;
	rectangles[0].minPoint.z = -FLT_MAX;
	rectangles[0].maxPoint.z = FLT_MAX;

	rectangles[1].minPoint.x = -FLT_MAX;
	rectangles[1].minPoint.y = srfRect->minPoint.y;
	rectangles[1].maxPoint.x = srfRect->minPoint.x;
	rectangles[1].maxPoint.y = srfRect->maxPoint.y;
	rectangles[1].minPoint.z = -FLT_MAX;
	rectangles[1].maxPoint.z = FLT_MAX;

	rectangles[2].minPoint.x = -FLT_MAX;
	rectangles[2].minPoint.y = srfRect->maxPoint.y;
	rectangles[2].maxPoint.x = FLT_MAX;
	rectangles[2].maxPoint.y = FLT_MAX;
	rectangles[2].minPoint.z = -FLT_MAX;
	rectangles[2].maxPoint.z = FLT_MAX;

	rectangles[3].minPoint.x = srfRect->maxPoint.x;
	rectangles[3].minPoint.y = srfRect->minPoint.y;
	rectangles[3].maxPoint.x = FLT_MAX;
	rectangles[3].maxPoint.y = srfRect->maxPoint.y;
	rectangles[3].minPoint.z = -FLT_MAX;
	rectangles[3].maxPoint.z = FLT_MAX;
}


// Clearance costs is calculated by determining any intersections between clearances and offlimits. Clearances may overlap with other clearances
template<int tile_sz>
__device__ float ClearanceCosts(cg::thread_block_tile<tile_sz> group, Surface *srf, point* cfg, vertex *vertices, rectangle *clearances, rectangle *offlimits)
{
	int tid = group.thread_rank();
	int step = group.size();
	float values = 0.0f;
	//As this is 2D, we can break things up better
	for (int i = tid; i < srf->nClearances; i += step) {
		BoundingBox rect1; //The starting bounding box
		calculateBoundingBox(cfg, clearances[i].SourceIndex, &rect1);
		MinkowskiSum(vertices, clearances[i].point1Index,cfg[clearances[i].SourceIndex].freedom[4], &rect1, &rect1);
		//vertex rect1Min = minValue(vertices, clearances[i].point1Index, cfg[clearances[i].SourceIndex].x, cfg[clearances[i].SourceIndex].y);
		//vertex rect1Max = maxValue(vertices, clearances[i].point1Index, cfg[clearances[i].SourceIndex].x, cfg[clearances[i].SourceIndex].y);
		for (int j = 0; j < srf->nObjs; j++) {
			BoundingBox rect2; //The starting bounding box
			calculateBoundingBox(cfg, j, &rect2);
			float area = calculateIntersectionArea(&rect1, &rect2);
			values += area; //Clearence penalty should be positive
		}
	}
	group.sync();
	values = reduce<tile_sz>(group, values);
	//printf("Clearance costs error: %f\n", error);
	return values;
}

// Both clearance as offlimits may not lie outside of the surface area
//I do not believe this is the best way to do this. Instead, it should be part of the propose function
template<int tile_sz>
__device__ float SurfaceAreaCosts(cg::thread_block_tile<tile_sz> group, Surface *srf, point* cfg, vertex *vertices, rectangle *clearances, rectangle *offlimits, vertex *surfaceRectangle) {
	//printf("Surface cost calculation\n");

	int tid = group.thread_rank();
	int step = group.size();
	float values = 0.0f; //Since it's size blockDim, we can have each of them treat it as the starting value
						 // Describe the complement of surfaceRectangle as four rectangles (using their min and max values)
	BoundingBox rect;
	BoundingBox complementRectangle[4];
	calculateBoundingBox(surfaceRectangle, 0, 0, &rect);

	//This gives us the clearence area outside our surface area
	createComplementRectangle(&rect, complementRectangle);
	BoundingBox rect1;
	for (int i = tid; i < srf->nClearances; i += step) {
		// Determine max and min vectors of clearance rectangles
		// rectangle #1
		vertex item;
		item.x = cfg[clearances[i].SourceIndex].freedom[0];
		item.y = cfg[clearances[i].SourceIndex].freedom[1];
		calculateBoundingBox(vertices, clearances[i].point1Index, cfg[clearances[i].SourceIndex].freedom[4], &rect1);
		MinkowskiSum(&item, &rect1, &rect1);
		for (int j = 0; j < 4; j++) {
			values += calculateIntersectionArea(&rect1, &complementRectangle[j]);
		}
	}
	//This is meant to get all the objects that do not have a clearence. It also double-counts all the clearence objects
	for (int i = tid; i < srf->nObjs; i += step) {
		// Determine max and min vectors of off limit rectangles
		// rectangle #1
		//offlimits is the size of cfg
		calculateBoundingBox(cfg, i, &rect1);
		float area = 0.0f;
		for (int j = 0; j < 4; j++) {
			area += calculateIntersectionArea(&rect1, &complementRectangle[j]);
		}
		values += area;
		//printf("Clearance rectangle %d:(%f,%f),(%f,%f), error is %f\n", i, rect1.minPoint.x, rect1.minPoint.y, rect1.maxPoint.x, rect1.maxPoint.y,area);
	}

	group.sync();
	values = reduce<tile_sz>(group, values);
	//printf("Clearance costs error: %f\n", error);
	return values;
}

template<int tile_sz>
__device__ float OffLimitsCosts(cg::thread_block_tile<tile_sz> group, Surface *srf, point* cfg, vertex *vertices, rectangle *offlimits) {
	int tid = group.thread_rank();
	int step = group.size();
	float values = 0.0f; //Since it's size blockDim, we can have each of them treat it as the starting value
	for (int i = tid; i < srf->nObjs; i += step) {
		BoundingBox rect1;
		calculateBoundingBox(cfg, i, &rect1);
		for (int j = i + 1; j < srf->nObjs; j++) {
			BoundingBox rect2;
			calculateBoundingBox(cfg, j, &rect2);
			float area = calculateIntersectionArea(&rect1, &rect2);
			/*printf("Area for (%f,%f,%f,%f) and (%f,%f,%f,%f) is %f\n",
				rect1.minPoint.x, rect1.minPoint.y, rect1.maxPoint.x, rect1.maxPoint.y,
				rect2.minPoint.x, rect2.minPoint.y, rect2.maxPoint.x, rect2.maxPoint.y, area);*/
			values += area;
		}
	}
	group.sync();
	values = reduce<tile_sz>(group, values);
	//printf("Clearance costs error: %f\n", error);
	return values;
}