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
/*This code makes several assumptions on the input of our data.
 * 1)Our warp size is fixed, and the total number of threads is a multiple of the warp size (which is our working group size)
 */



#include "Kernel.h"
#include "CostFunctions.h"
#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif





__global__ void initRNG(curandState *const rngStates, const unsigned int seed)
{
	// Determine thread ID
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("tid: %d\n", tid);
	//printf("seed: %d\n", seed);
	// Initialise the RNG
	curand_init(seed + tid, tid, 0, &rngStates[tid]);
}

//Given the prevalance in graphics, I'm really surprised this isn't a standard function in cuda
__inline__ __device__ float clamp(float value, float min, float max) {
	return fminf(max, fmaxf(value, min));
}

__device__ void CopyCosts(resultCosts* copyFrom, resultCosts* copyTo)
{
	copyTo->PairWiseCosts = copyFrom->PairWiseCosts;
	copyTo->VisualBalanceCosts = copyFrom->VisualBalanceCosts;
	copyTo->FocalPointCosts = copyFrom->FocalPointCosts;
	copyTo->SymmetryCosts = copyFrom->SymmetryCosts;;
	copyTo->ClearanceCosts = copyFrom->ClearanceCosts;
	copyTo->OffLimitsCosts = copyFrom->OffLimitsCosts;
	copyTo->SurfaceAreaCosts = copyFrom->SurfaceAreaCosts;
	copyTo->AlignmentCosts = copyFrom->AlignmentCosts;
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

__inline__ __device__ float generateRandomFloatInRange(curandState *rngStates, unsigned int tid, float max, float min)
{
	curandState localState = rngStates[tid];
	float p_rand = curand_uniform(&localState);
	rngStates[tid] = localState;
	p_rand *= (max - min);
	p_rand += min;
	return p_rand; //The only difference between float and int is that we do not trucate the float in this one
}

/******************************************************************************************************************************************************************************************************************************/
//This is the start of our cost functions
//This cost function does everything but our pairwise costs, which differ between the different scene synth implimentations. This is what is kept similar between them
//or globally only for Merrell's algorithm
//Our if statements run per tile_sz (warp), so there shouldn't be no-op problems
template<int tile_sz>
__device__ void Costs(cg::thread_block_tile<tile_sz> group, Surface *srf, resultCosts* costs, point* cfg, vertex *vertices, rectangle *clearances, rectangle *offlimits, vertex *surfaceRectangle) {
	int gid = group.thread_rank();
	float visualBalanceCosts = 0;
	if (srf->WeightVisualBalance > 0.0f)
		visualBalanceCosts = srf->WeightVisualBalance * VisualBalanceCosts<tile_sz>(group, srf, cfg);
	// printf("Visual balance costs with weight %f\n", visualBalanceCosts);
	float focalPointCosts = 0;
	if (srf->WeightFocalPoint > 0.0f)
		focalPointCosts = srf->WeightFocalPoint * FocalPointCosts<tile_sz>(group, srf, cfg);
	// printf("Focal point costs with weight %f\n", focalPointCosts);
	float symmertryCosts = 0;
	if (srf->WeightSymmetry > 0.0f)
		symmertryCosts = srf->WeightSymmetry * SymmetryCosts<tile_sz>(group, srf, cfg);
	// printf("Symmertry costs with weight %f\n", symmertryCosts);
	float offlimitsCosts = 0;
	if (srf->WeightOffLimits > 0.0f)
		offlimitsCosts = srf->WeightOffLimits * OffLimitsCosts<tile_sz>(group, srf, cfg, vertices, offlimits);
	//printf("OffLimits costs with weight %f\n", offlimitsCosts);
	float clearanceCosts = 0;
	if (srf->WeightClearance > 0.0f)
		clearanceCosts = srf->WeightClearance * ClearanceCosts<tile_sz>(group, srf, cfg, vertices, clearances, offlimits);
	float surfaceAreaCosts = 0;
	if (srf->WeightSurfaceArea > 0.0f)
		surfaceAreaCosts = srf->WeightSurfaceArea * SurfaceAreaCosts<tile_sz>(group, srf, cfg, vertices, clearances, offlimits, surfaceRectangle);
	//printf("Surface area costs with weight %f\n", surfaceAreaCosts);
	float alignmentCosts = 0;
	if (srf->WeightAlignment > 0.0f)
		alignmentCosts = srf->WeightAlignment * AlignmentCosts<tile_sz>(group, srf, cfg);


	if (gid == 0) {
		costs->VisualBalanceCosts = visualBalanceCosts;
		costs->FocalPointCosts = focalPointCosts;
		costs->SymmetryCosts = symmertryCosts;
		costs->ClearanceCosts = clearanceCosts;
		costs->OffLimitsCosts = offlimitsCosts;
		costs->SurfaceAreaCosts = surfaceAreaCosts;
		costs->AlignmentCosts = alignmentCosts;
	}
	group.sync();
}
//This contains the euclidean pairwise cost that was originally in Merell et al.
template<int tile_sz>
__device__ void Costs(cg::thread_block_tile<tile_sz> group, Surface *srf, resultCosts* costs, point* cfg, relationshipStruct *rs, vertex *vertices, rectangle *clearances, rectangle *offlimits, vertex *surfaceRectangle)
{
	int gid = group.thread_rank();
	float pairWiseCosts = 0;
	if (srf->WeightPairWise > 0.0f) {
		pairWiseCosts = srf->WeightPairWise * PairWiseEuclidean<tile_sz>(group, srf, cfg, rs);
	}
	Costs<tile_sz>(group, srf, costs,  cfg, vertices, clearances,offlimits, surfaceRectangle);
	if (gid == 0) {
		costs->PairWiseCosts = pairWiseCosts;
		costs->totalCosts = pairWiseCosts + costs->VisualBalanceCosts + costs->FocalPointCosts + costs->SymmetryCosts + costs->ClearanceCosts + costs->OffLimitsCosts + costs->SurfaceAreaCosts + costs->AlignmentCosts;
	}
	group.sync();
}
//Gaussian that is in Fisher et al.
template<int tile_sz>
__device__ void Costs(cg::thread_block_tile<tile_sz> group, 
							  Surface *srf, resultCosts* costs, point* cfg, 
							  guassianRelationshipStruct* rs, vertex *vertices, rectangle *clearances, 
							  rectangle *offlimits, vertex *surfaceRectangle)
{
	int gid = group.thread_rank();
	float pairWiseCosts = 0;
	if (srf->WeightPairWise > 0.0f) {
		pairWiseCosts = srf->WeightPairWise * PairWiseGuassian<tile_sz>(group, srf, cfg, rs);
	}
	Costs<tile_sz>(group, srf, costs, cfg, vertices, clearances, offlimits, surfaceRectangle);
	if (gid == 0) {
		costs->PairWiseCosts = pairWiseCosts;
		costs->totalCosts = pairWiseCosts + costs->VisualBalanceCosts + costs->FocalPointCosts + costs->SymmetryCosts + costs->ClearanceCosts + costs->OffLimitsCosts + costs->SurfaceAreaCosts + costs->AlignmentCosts;

	}
	group.sync();
}
//Gaussian that is in Fisher et al. with angle tolerances
template<int tile_sz>
__device__ void Costs(cg::thread_block_tile<tile_sz> group,
	Surface *srf, resultCosts* costs, point* cfg,
	guassianRelationshipStruct* rs, guassianRelationshipStruct* ra, vertex *vertices, rectangle *clearances,
	rectangle *offlimits, vertex *surfaceRectangle)
{
	int gid = group.thread_rank();
	float pairWiseCosts = 0;
	if (srf->WeightPairWise > 0.0f) {
		pairWiseCosts = srf->WeightPairWise * (PairWiseGuassian<tile_sz>(group, srf, cfg, rs)+ PairWiseGuassianAngle(group,srf,cfg,ra));
	}
	Costs<tile_sz>(group, srf, costs, cfg, vertices, clearances, offlimits, surfaceRectangle);
	if (gid == 0) {
		costs->PairWiseCosts = pairWiseCosts;
		costs->totalCosts = pairWiseCosts + costs->VisualBalanceCosts + costs->FocalPointCosts + costs->SymmetryCosts + costs->ClearanceCosts + costs->OffLimitsCosts + costs->SurfaceAreaCosts + costs->AlignmentCosts;
	}
	group.sync();
}
//Gaussian that is in Fisher et al. with angle tolerances
template<int tile_sz>
__device__ void Costs(cg::thread_block_tile<tile_sz> group,
	Surface *srf, resultCosts* costs, point* cfg,
	relationshipStruct* rs, guassianRelationshipStruct* ra, vertex *vertices, rectangle *clearances,
	rectangle *offlimits, vertex *surfaceRectangle)
{
	int gid = group.thread_rank();
	float pairWiseCosts = 0;
	if (srf->WeightPairWise > 0.0f) {
		pairWiseCosts = srf->WeightPairWise * (PairWiseEuclidean<tile_sz>(group, srf, cfg, rs) + PairWiseGaussian1D(group, srf, cfg, ra,4));
	}
	Costs<tile_sz>(group, srf, costs, cfg, vertices, clearances, offlimits, surfaceRectangle);
	if (gid == 0) {
		costs->PairWiseCosts = pairWiseCosts;
		costs->totalCosts = costs->PairWiseCosts + costs->VisualBalanceCosts + costs->FocalPointCosts + costs->SymmetryCosts + costs->ClearanceCosts + costs->OffLimitsCosts + costs->SurfaceAreaCosts + costs->AlignmentCosts;
		/*printf("(p=%f,v=%f,f=%f,s=%f,c=%f,o=%f,sa=%f,a=%f) = %f\n", costs->PairWiseCosts,
			costs->VisualBalanceCosts,
			costs->FocalPointCosts,
			costs->SymmetryCosts,
			costs->ClearanceCosts,
			costs->OffLimitsCosts,
			costs->SurfaceAreaCosts,
			costs->AlignmentCosts,
			costs->totalCosts);*/
	}
	group.sync();
}
//End of cost functions
/**************************************************************************************************************************************************************************************************************************/


template<int tile_sz>
__device__ void propose(cg::thread_block_tile<tile_sz> group, Surface *srf, point* cfg, vertex * surfaceRectangle, curandState *rngStates, unsigned int tid)
{
	int gid = group.thread_rank();
	/*for (int j = 0; j < srf->nObjs; j++)
	{
		printf("Star values inside proposition jndex %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", j, cfgStar[j].x, cfgStar[j].y, cfgStar[j].z, cfgStar[j].rotX, cfgStar[j].freedom[4], cfgStar[j].rotZ);
	}*/
	int p = generateRandomIntInRange(rngStates, tid, 2, 0);

	//Get everyone on the same page
	p = group.shfl(p, 0); //broadcast out to p
	//group.sync(); shlf_sync is broadcast and so should sync
	// Determine width and length of surface rectangle
	BoundingBox srfRect;
	calculateBoundingBox(surfaceRectangle, 0, 0, &srfRect);
	float width  = srfRect.maxPoint.x - srfRect.minPoint.x;
	float length = srfRect.maxPoint.y - srfRect.minPoint.y;
	// Dividing the width by 2 makes sure that it stays withing a 95% percentile range that is usable, dividing it by 4 makes sure that it stretches the half of the length/width or lower (and that inside a 95% interval).
	float stdXAxis = width / 16;
	float stdYAxis = length / 16;

	// printf("Selected mode: %d\n", p);
	// Translate location using normal distribution
	if (p == 0)
	{
		//bool found = false;
		int obj = -1;
		obj = generateRandomIntInRange(rngStates, tid, srf->nObjs - 1, 0);
		if (!group.any(!cfg[obj].frozen)) {
			return;
		}
		int mask = group.ballot(!cfg[obj].frozen);
		int leader = __ffs(mask);
		obj = group.shfl(obj, leader);
		//Everybody choses a direction and change. The first one we don't have to clamp is the winner
		int direction = generateRandomIntInRange(rngStates, tid, 2, 0);
		if (gid == 0) {
			//printf("Selected object #: %d\n", obj);
			float dx = curand_normal(&rngStates[tid]);
			cfg[obj].freedom[0] += dx * stdXAxis;
			//printf("dx: %f\n", dx);
			float dy = curand_normal(&rngStates[tid]);
			cfg[obj].freedom[1] += dy * stdYAxis;
			// printf("Before translation, obj %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", obj, cfgStar[obj].freedom[0], cfgStar[obj].y, cfgStar[obj].z, cfgStar[obj].rotX, cfgStar[obj].freedom[4], cfgStar[obj].rotZ);
			//Clamp should be taken care of by surface area
			//cfg[obj].freedom[0] = clamp(cfg[obj].freedom[0], srfRect.minPoint.x, srfRect.maxPoint.x); //xClamps
			//cfg[obj].freedom[1] = clamp(cfg[obj].freedom[1], srfRect.minPoint.y, srfRect.maxPoint.y); //yClamps
		}
		// printf("After rotation, obj %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", obj, cfgStar[obj].x, cfgStar[obj].y, cfgStar[obj].z, cfgStar[obj].rotX, cfgStar[obj].freedom[4], cfgStar[obj].rotZ);
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
			// printf("Before rotation, obj %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", obj, cfgStar[obj].x, cfgStar[obj].y, cfgStar[obj].z, cfgStar[obj].rotX, cfgStar[obj].freedom[4], cfgStar[obj].rotZ);
			float dRot = curand_normal(&rngStates[tid]);
			cfg[obj].freedom[4] += dRot * S_SIGMA_T;
			cfg[obj].freedom[4]  = (cfg[obj].freedom[4]  <    0) ? cfg[obj].freedom[4] + 2 * PI : cfg[obj].freedom[4];
			cfg[obj].freedom[4] =  (cfg[obj].freedom[4] >= 2*PI) ? cfg[obj].freedom[4] - 2 * PI : cfg[obj].freedom[4];
		}
		// printf("After rotation, obj %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", obj, cfgStar[obj].x, cfgStar[obj].y, cfgStar[obj].z, cfgStar[obj].rotX, cfgStar[obj].freedom[4], cfgStar[obj].rotZ);
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
		mask = group.ballot(!cfg[obj2].frozen && obj2 != obj1);
		leader = __ffs(mask);
		obj2 = group.shfl(obj2, leader);

		/*if (obj1 == obj2) {
			return; //No point at this step
		}*/
		// printf("First selected object #: %d\n", obj1);
		// printf("Second selected object #: %d\n", obj2);

		// printf("Values, obj %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", obj1, cfgStar[obj1].freedom[0], cfgStar[obj1].y, cfgStar[obj1].z, cfgStar[obj1].rotX, cfgStar[obj1].freedom[4], cfgStar[obj1].rotZ);
		// printf("Values of, obj %d. X, Y, Z: %f, %f, %f rotation: %f, %f, %f\n", obj2, cfgStar[obj2].freedom[0], cfgStar[obj2].y, cfgStar[obj2].z, cfgStar[obj2].rotX, cfgStar[obj2].freedom[4], cfgStar[obj2].rotZ);
		if(gid < 5){
			float temp = cfg[obj1].freedom[gid];
			cfg[obj1].freedom[gid] = cfg[obj2].freedom[gid];
			cfg[obj2].freedom[gid] = temp;
		}
	}
	//Already sync after propose in the next function, so no need to here
}

__device__ bool Accept(float costStar, float costCur, curandState *rngStates, unsigned int tid, float beta)
{
	//printf("Costs of Star, Cur, and beta difference: %f, %f, %f\n", costStar,costCur,(costStar - costCur));
	//printf("(float) exp(-BETA * (costStar - costCur)): %f\n", (float)expf(beta * (costCur - costStar)));
	float randomNumber = curand_uniform(&rngStates[tid]);
	//printf("Random number: %f\n", randomNumber);
	//In reality, it should be -beta, because that is the boltsmann dist. However, writing it exactly like that  means that we favor higher star costs (as a negitive difference means e^x > 1)
	//instead of lower star costs. We also want the effect of it being harder to be greater than one when beta is lower
	return  randomNumber < fminf(1.0f, expf(-beta * (costStar - costCur)));
}

template<int tile_sz>
__device__ void Copy(cg::thread_block_tile<tile_sz> group, point* cfg1, point* cfg2, Surface* srf)
{
	int tid = group.thread_rank();
	int step = group.size();
	for (unsigned int i = tid; i < srf->nObjs; i += step)
	{
		cfg1[i].freedom[0] = cfg2[i].freedom[0];
		cfg1[i].freedom[1] = cfg2[i].freedom[1];
		cfg1[i].freedom[2] = cfg2[i].freedom[2];
		cfg1[i].freedom[3] = cfg2[i].freedom[3];
		cfg1[i].freedom[4] = cfg2[i].freedom[4];
		cfg1[i].frozen = cfg2[i].frozen;
		cfg1[i].length = cfg2[i].length;
		cfg1[i].width = cfg2[i].width;
		cfg1[i].height = cfg2[i].height;
	}
	group.sync();
}

//Lowers the temperature (beta) randomly between a set value
//In general temperature prediction is extremely difficult, and so
//this function should be changed accordingly to meet those needs
template<int tile_sz>
__device__ float generateBeta(cg::thread_block_tile<tile_sz> group, curandState *rngStates, float old_beta) {
	//Merrell et al. uses replica exchange to exchange temperature constants between warps.
	//Here, we use Simulated tempering to move the range states down from their starting state, starting on a random hot state and 
	//moving to a random cooler state.
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float beta = generateRandomFloatInRange(rngStates, tid, old_beta, old_beta /2); //We get a beta value between hot (accept if it is better or cold). 
	//if (group.thread_rank() == 0) {
	//	printf("Beta is %f from %f\n", beta, old_beta);
	//}
	beta = group.shfl(beta, 0);//shf calls shfl_sync, which as a broadcast should sync
	return beta;
}

template<int tile_sz>
__device__ void acceptKernel(cg::thread_block_tile<tile_sz> group,
	point* cfgBest,
	resultCosts* bestCosts,
	point* cfgStar,
	Surface *srf,
	resultCosts* starCosts,
	curandState *rngStates,
	float beta) {
	int gtid = group.thread_rank();//The thread id in the working group
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	bool accept;
	if (gtid == 0) {
		accept = Accept(starCosts->totalCosts, bestCosts->totalCosts, rngStates, tid, beta);
		//if(accept)
		//	printf("Star accepted, cost goes from %f -> %f with beta %f\n", bestCosts->totalCosts, starCosts->totalCosts,beta);
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
		//CopyCosts(bestCosts, starCosts); //We re-run the costs, so no need to copy it
	}
}
/*****************************************************************************************************************************************************************************/
//Our group kernal for euclidean distance
template<int tile_sz>
__device__ void groupKernel(cg::thread_block_tile<tile_sz> group,
	point* cfgBest,
	resultCosts* bestCosts,
	point* cfgStar,
	resultCosts* starCosts,
	relationshipStruct *rs,
	rectangle *clearances, rectangle *offlimits,
	vertex *vertices, vertex *surfaceRectangle, Surface *srf,
	int iterations, curandState *rngStates)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	int step = group.size();//The working group block size
	float beta = generateBeta(group,rngStates,generateRandomFloatInRange(rngStates, tid, 20, 10)); //This is a high value, and should give us a good starting range
	Copy<tile_sz>(group, cfgStar, cfgBest, srf);
	Costs<tile_sz>(group, srf, bestCosts, cfgBest, rs, vertices, clearances, offlimits, surfaceRectangle); //possible race condition here
	CopyCosts(bestCosts, starCosts);
	//printf("Threadblock: %d, Best costs before: %f\n", blockIdx.x, bestCosts->totalCosts);
	int modifier = iterations / 10;
	for (int i = 0; i < iterations; i++)
	{
		
		propose<tile_sz>(group, srf, cfgStar, surfaceRectangle, rngStates, tid);
		group.sync();
		Costs<tile_sz>(group, srf, starCosts, cfgStar, rs, vertices, clearances, offlimits, surfaceRectangle);
		acceptKernel<tile_sz>(group, cfgBest, bestCosts, cfgStar, srf, starCosts, rngStates,beta);
		if((i+1)% modifier == 0) //get about 10 cooling downs
			beta = generateBeta(group, rngStates, beta);//Every so often, we possibly cool the system
	}
	group.sync();
}
//Our group kernel for guassian2D distance
template<int tile_sz>
__device__ void groupKernel(cg::thread_block_tile<tile_sz> group,
	point* cfgBest,
	resultCosts* bestCosts,
	point* cfgStar,
	resultCosts* starCosts,
	guassianRelationshipStruct* rs,
	rectangle *clearances, rectangle *offlimits,
	vertex *vertices, vertex *surfaceRectangle, Surface *srf,
	int iterations, curandState *rngStates)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	int step = group.size();//The working group block size
	float beta = generateBeta(group, rngStates, generateRandomFloatInRange(rngStates, tid, 20, 10)); //This is a high value, and should give us a good starting range
	Copy<tile_sz>(group, cfgStar, cfgBest, srf);
	Costs<tile_sz>(group, srf, bestCosts, cfgBest, rs, vertices, clearances, offlimits, surfaceRectangle); //possible race condition here
	CopyCosts(bestCosts, starCosts);
	//printf("Threadblock: %d, Best costs before: %f\n", blockIdx.x, bestCosts->totalCosts);
	int modifier = iterations / 10;
	for (int i = 0; i < iterations; i++)
	{

		propose<tile_sz>(group, srf, cfgStar, surfaceRectangle, rngStates, tid);
		group.sync();
		Costs<tile_sz>(group, srf, starCosts, cfgStar, rs, vertices, clearances, offlimits, surfaceRectangle);
		acceptKernel<tile_sz>(group, cfgBest, bestCosts, cfgStar, srf, starCosts, rngStates, beta);
		if ((i + 1) % modifier == 0) //get about 10 cooling downs
			beta = generateBeta(group, rngStates, beta);//Every so often, we possibly cool the system
	}
	group.sync();
}
//Our group kernel for guassian2D distance with angle
template<int tile_sz>
__device__ void groupKernel(cg::thread_block_tile<tile_sz> group,
	point* cfgBest,
	resultCosts* bestCosts,
	point* cfgStar,
	resultCosts* starCosts,
	guassianRelationshipStruct* rs,
	guassianRelationshipStruct* ra,
	rectangle *clearances, rectangle *offlimits,
	vertex *vertices, vertex *surfaceRectangle, Surface *srf,
	int iterations, curandState *rngStates)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	int step = group.size();//The working group block size
	float beta = generateBeta(group, rngStates, generateRandomFloatInRange(rngStates, tid, 20, 10)); //This is a high value, and should give us a good starting range
	Copy<tile_sz>(group, cfgStar, cfgBest, srf);
	Costs<tile_sz>(group, srf, bestCosts, cfgBest, rs,ra, vertices, clearances, offlimits, surfaceRectangle); //possible race condition here
	CopyCosts(bestCosts, starCosts);
	//printf("Threadblock: %d, Best costs before: %f\n", blockIdx.x, bestCosts->totalCosts);
	int modifier = iterations / 10;
	for (int i = 0; i < iterations; i++)
	{

		propose<tile_sz>(group, srf, cfgStar, surfaceRectangle, rngStates, tid);
		group.sync();
		Costs<tile_sz>(group, srf, starCosts, cfgStar, rs, ra, vertices, clearances, offlimits, surfaceRectangle);
		acceptKernel<tile_sz>(group, cfgBest, bestCosts, cfgStar, srf, starCosts, rngStates, beta);
		if ((i + 1) % modifier == 0) //get about 10 cooling downs
			beta = generateBeta(group, rngStates, beta);//Every so often, we possibly cool the system
	}
	group.sync();
}

//Our group kernel for  distance with guassian angle
template<int tile_sz>
__device__ void groupKernel(cg::thread_block_tile<tile_sz> group,
	point* cfgBest,
	resultCosts* bestCosts,
	point* cfgStar,
	resultCosts* starCosts,
	relationshipStruct* rs,
	guassianRelationshipStruct* ra,
	rectangle *clearances, rectangle *offlimits,
	vertex *vertices, vertex *surfaceRectangle, Surface *srf,
	int iterations, curandState *rngStates)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	int step = group.size();//The working group block size
	float beta = generateBeta(group, rngStates, generateRandomFloatInRange(rngStates, tid, 20, 10)); //This is a high value, and should give us a good starting range
	Copy<tile_sz>(group, cfgStar, cfgBest, srf);
	Costs<tile_sz>(group, srf, bestCosts, cfgBest, rs, ra, vertices, clearances, offlimits, surfaceRectangle); //possible race condition here
	CopyCosts(bestCosts, starCosts);
	//printf("Threadblock: %d, Best costs before: %f\n", blockIdx.x, bestCosts->totalCosts);
	int modifier = iterations / 10;
	for (int i = 0; i < iterations; i++)
	{

		propose<tile_sz>(group, srf, cfgStar, surfaceRectangle, rngStates, tid);
		group.sync();
		Costs<tile_sz>(group, srf, starCosts, cfgStar, rs, ra, vertices, clearances, offlimits, surfaceRectangle);
		acceptKernel<tile_sz>(group, cfgBest, bestCosts, cfgStar, srf, starCosts, rngStates, beta);
		if ((i + 1) % modifier == 0) //get about 10 cooling downs
			beta = generateBeta(group, rngStates, beta);//Every so often, we possibly cool the system
	}
	group.sync();
}
/**************************************************************************************************************************************************************************/
//Helper function to copy the information from global memory into a shared array
template<int tile_sz>
__device__ void copyToSharedMemory(cg::thread_block_tile<tile_sz> group,
	point *p,
	Surface *srf,
	point* configuration) {
	int gid = group.thread_rank();
	for (unsigned int i = gid; i < srf->nObjs; i += WARP_SIZE)
	{
		// BlockId counts from 0, so to properly multiply
		int index = blockIdx.x * srf->nObjs + i;
		configuration[i].freedom[0] = p[index].freedom[0];
		configuration[i].freedom[1] = p[index].freedom[1];
		configuration[i].freedom[2] = p[index].freedom[2];
		configuration[i].freedom[3] = p[index].freedom[3];
		configuration[i].freedom[4] = p[index].freedom[4];
		configuration[i].frozen = p[index].frozen;
		configuration[i].length = p[index].length;
		configuration[i].width = p[index].width;
		configuration[i].height = p[index].height;
	}

}
//Helper function to copy the information from shared to global.
__device__ void copyToGlobalMemory(
	point *p,
	Surface *srf,
	resultCosts* resultCostsArray,
	point* configuration,
	resultCosts* costs,
	int lowest_cost) {
	//Copy current config back into the global memory
	// Copy best config (now set to input config) to result of this block
	for (unsigned int i = threadIdx.x; i < srf->nObjs; i += blockDim.x)
	{
		// BlockId counts from 0, so to properly multiply
		int index = blockIdx.x * srf->nObjs + i;
		p[index].freedom[0] = configuration[lowest_cost * srf->nObjs + i].freedom[0];
		p[index].freedom[1] = configuration[lowest_cost * srf->nObjs + i].freedom[1];
		p[index].freedom[2] = configuration[lowest_cost * srf->nObjs + i].freedom[2];
		p[index].freedom[3] = configuration[lowest_cost * srf->nObjs + i].freedom[3];
		p[index].freedom[4] = configuration[lowest_cost * srf->nObjs + i].freedom[4];
		p[index].frozen = configuration[lowest_cost * srf->nObjs + i].frozen;
		p[index].length = configuration[lowest_cost * srf->nObjs + i].length;
		p[index].width  = configuration[lowest_cost * srf->nObjs + i].width;
		p[index].height = configuration[lowest_cost * srf->nObjs + i].height;
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
	resultCostsArray[blockIdx.x].AlignmentCosts = costs[lowest_cost].AlignmentCosts;

}

//This function figures out the lowest cost of our search
//It can be written as a reduction problem, and definitely should
__device__ int lowestIndex(resultCosts* best_costs, int active_warps) {
	int best_cost = 0;
	for (int i = 1; i < active_warps; i++) {
		if (best_costs[i].totalCosts < best_costs[best_cost].totalCosts) {
			best_cost = i;
		}
	}
	return best_cost;
}
/*****************************************************************************************************************************************************************************************/
//Begin kernal function
// result is a [,] array with 1 dimension equal to the amount of blocks used and the other dimension equal to the amount of objects
// rs is an array with the length equal to the amount of relationships
// cfg is an array with the length equal to the amount of objects
//This isn't really worth it to break up 
// Surface is a basic struct
__global__ void Kernel(resultCosts* resultCostsArray,
	point *p, relationshipStruct *rs,
	rectangle *clearances, rectangle *offlimits,
	vertex *vertices, vertex *surfaceRectangle, Surface *srf,
	gpuConfig *gpuCfg, curandState *rngStates) {

	extern __shared__ int all_shared_memory[];
	int jumper = blockDim.x / WARP_SIZE;
	point* configurations = (point*)&all_shared_memory;
	resultCosts* costs = (resultCosts*)&configurations[2 * jumper * srf->nObjs]; 
	__syncthreads();
   //create the working groups
	int rank = threadIdx.x / WARP_SIZE;
	point* best_conf = &configurations[rank * srf->nObjs];
	point* star_conf = &configurations[srf->nObjs * (jumper + rank)];
	resultCosts* best_cost = &costs[rank];
	resultCosts* star_cost = &costs[jumper + rank];
	auto tile_warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block()); //Broken up by our warp size, which is our static shared memory size!

	//This is the actual work done
	copyToSharedMemory<WARP_SIZE>(tile_warp, p, srf, best_conf);
	groupKernel<WARP_SIZE>(tile_warp,best_conf,best_cost,star_conf,star_cost, rs, clearances, offlimits, vertices, surfaceRectangle, srf, gpuCfg->iterations, rngStates);
	__syncthreads();
	int lowest_cost = lowestIndex(best_cost, jumper);
	copyToGlobalMemory(p, srf, resultCostsArray, configurations, costs, lowest_cost);
	//__syncthreads();
}
//Gaussian Kernel
__global__ void Kernel(resultCosts* resultCostsArray,
	point *p, guassianRelationshipStruct* rs,
	rectangle *clearances, rectangle *offlimits,
	vertex *vertices, vertex *surfaceRectangle, Surface *srf,
	gpuConfig *gpuCfg, curandState *rngStates) {

	extern __shared__ int all_shared_memory[];
	int jumper = blockDim.x / WARP_SIZE;
	point* configurations = (point*)&all_shared_memory;
	resultCosts* costs = (resultCosts*)&configurations[2 * jumper * srf->nObjs];
	__syncthreads();
	//create the working groups
	int rank = threadIdx.x / WARP_SIZE;
	point* best_conf = &configurations[rank * srf->nObjs];
	point* star_conf = &configurations[srf->nObjs * (jumper + rank)];
	resultCosts* best_cost = &costs[rank];
	resultCosts* star_cost = &costs[jumper + rank];
	auto tile_warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block()); //Broken up by our warp size, which is our static shared memory size!

																			  //This is the actual work done
	copyToSharedMemory<WARP_SIZE>(tile_warp, p, srf, best_conf);
	groupKernel<WARP_SIZE>(tile_warp, best_conf, best_cost, star_conf, star_cost, rs, clearances, offlimits, vertices, surfaceRectangle, srf, gpuCfg->iterations, rngStates);
	__syncthreads();
	int lowest_cost = lowestIndex(best_cost, jumper);
	copyToGlobalMemory(p, srf, resultCostsArray, configurations, costs, lowest_cost);
	//__syncthreads();
}
//Gaussian Kernal with Angle
//Gaussian Kernel
__global__ void Kernel(resultCosts* resultCostsArray,
	point *p, guassianRelationshipStruct* rs,
	guassianRelationshipStruct* ra,
	rectangle *clearances, rectangle *offlimits,
	vertex *vertices, vertex *surfaceRectangle, Surface *srf,
	gpuConfig *gpuCfg, curandState *rngStates) {

	extern __shared__ int all_shared_memory[];
	int jumper = blockDim.x / WARP_SIZE;
	point* configurations = (point*)&all_shared_memory;
	resultCosts* costs = (resultCosts*)&configurations[2 * jumper * srf->nObjs];
	__syncthreads();
	//create the working groups
	int rank = threadIdx.x / WARP_SIZE;
	point* best_conf = &configurations[rank * srf->nObjs];
	point* star_conf = &configurations[srf->nObjs * (jumper + rank)];
	resultCosts* best_cost = &costs[rank];
	resultCosts* star_cost = &costs[jumper + rank];
	auto tile_warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block()); //Broken up by our warp size, which is our static shared memory size!
	copyToSharedMemory<WARP_SIZE>(tile_warp, p, srf, best_conf);
	groupKernel<WARP_SIZE>(tile_warp, best_conf, best_cost, star_conf, star_cost, rs,ra,clearances, offlimits, vertices, surfaceRectangle, srf, gpuCfg->iterations, rngStates);
	__syncthreads();
	int lowest_cost = lowestIndex(best_cost, jumper);
	copyToGlobalMemory(p, srf, resultCostsArray, configurations, costs, lowest_cost);
	//__syncthreads();
}

__global__ void Kernel(resultCosts* resultCostsArray,
	point *p, relationshipStruct* rs,
	guassianRelationshipStruct* ra,
	rectangle *clearances, rectangle *offlimits,
	vertex *vertices, vertex *surfaceRectangle, Surface *srf,
	gpuConfig *gpuCfg, curandState *rngStates) {

	extern __shared__ int all_shared_memory[];
	int jumper = blockDim.x / WARP_SIZE;
	point* configurations = (point*)&all_shared_memory;
	resultCosts* costs = (resultCosts*)&configurations[2 * jumper * srf->nObjs];
	__syncthreads();
	//create the working groups
	int rank = threadIdx.x / WARP_SIZE;
	point* best_conf = &configurations[rank * srf->nObjs];
	point* star_conf = &configurations[srf->nObjs * (jumper + rank)];
	resultCosts* best_cost = &costs[rank];
	resultCosts* star_cost = &costs[jumper + rank];
	auto tile_warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block()); //Broken up by our warp size, which is our static shared memory size!
	copyToSharedMemory<WARP_SIZE>(tile_warp, p, srf, best_conf);
	groupKernel<WARP_SIZE>(tile_warp, best_conf, best_cost, star_conf, star_cost, rs, ra, clearances, offlimits, vertices, surfaceRectangle, srf, gpuCfg->iterations, rngStates);
	__syncthreads();
	int lowest_cost = lowestIndex(best_cost, jumper);
	copyToGlobalMemory(p, srf, resultCostsArray, configurations, costs, lowest_cost);
	//__syncthreads();
}
//End Kernel functions
/***********************************************************************************************************************************************************************************/
//This is the kernal wrapper for everything. Sadly, it may not be worth it to break it up like some of our other functions
extern "C" __declspec(dllexport) result* KernelWrapper(relationshipStruct *rss, point *previouscfgs, rectangle *clearances, rectangle *offlimits, vertex *vertices, vertex *surfaceRectangle, Surface *srf, gpuConfig *gpuCfg)
{
	// Create pointer for on gpu
	// Determine memory size of object to transfer
	// Malloc on GPU size
	// Cpy memory from cpu to gpu

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

	relationshipStruct *gpuRS;
	int rsSize = sizeof(relationshipStruct) * srf->nRelationships;
	checkCudaErrors(cudaMalloc(&gpuRS, rsSize));
	checkCudaErrors(cudaMemcpy(gpuRS, rss, rsSize, cudaMemcpyHostToDevice));

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
	int share_size = gpuCfg->blockxDim / WARP_SIZE * 2 * srf->nObjs * sizeof(point) + gpuCfg->blockxDim / WARP_SIZE * 2 * sizeof(resultCosts);
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
		resultPointer[i].costs.AlignmentCosts = outResultCosts[i].AlignmentCosts;
		resultPointer[i].points = &(outPointArray[i * srf->nObjs]);
	}
	return resultPointer;
}

//This is the kernal wrapper for everything. Sadly, it may not be worth it to break it up like some of our other functions
extern "C" __declspec(dllexport) result* KernelGaussianWrapper(guassianRelationshipStruct* rss, point *previouscfgs, rectangle *clearances, rectangle *offlimits, vertex *vertices, vertex *surfaceRectangle, Surface *srf, gpuConfig *gpuCfg)
{
	// Create pointer for on gpu
	// Determine memory size of object to transfer
	// Malloc on GPU size
	// Cpy memory from cpu to gpu

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

	guassianRelationshipStruct* gpuRS;
	int rsSize = sizeof(guassianRelationshipStruct) * srf->nRelationships;
	checkCudaErrors(cudaMalloc(&gpuRS, rsSize));
	checkCudaErrors(cudaMemcpy(gpuRS, rss, rsSize, cudaMemcpyHostToDevice));

	// Output
	point *gpuPointArray;
	int pointArraySize = srf->nObjs * sizeof(point) * gpuCfg->gridxDim;
	point *outPointArray = (point *)malloc(pointArraySize);
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
	initRNG << <gpuCfg->gridxDim, gpuCfg->blockxDim >> > (d_rngStates, time(NULL));

	// Commented for possible later usage
	// dim3 dimGrid(gpuCfg->gridxDim, gpuCfg->gridyDim);
	// dim3 dimBlock(gpuCfg->blockxDim, gpuCfg->blockyDim, gpuCfg->blockzDim);

	// Block 1 dimensional, amount of threads available, configurable
	// Grid 1 dimension, amount of suggestions to be made.
	//we make the dynamic memory 3 times because we have at least 3 arrays that use it in one function
	int share_size = gpuCfg->blockxDim / WARP_SIZE * 2 * srf->nObjs * sizeof(point) + gpuCfg->blockxDim / WARP_SIZE * 2 * sizeof(resultCosts);
	Kernel << <gpuCfg->gridxDim, gpuCfg->blockxDim, share_size >> >(gpuResultCosts, gpuPointArray, gpuRS, gpuClearances, gpuOfflimits, gpuVertices, gpuSurfaceRectangle, gpuSRF, gpuGpuConfig, d_rngStates);
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
		resultPointer[i].costs.AlignmentCosts = outResultCosts[i].AlignmentCosts;
		resultPointer[i].points = &(outPointArray[i * srf->nObjs]);
	}
	return resultPointer;
}

//This is the kernal wrapper for everything. Sadly, it may not be worth it to break it up like some of our other functions
extern "C" __declspec(dllexport) result* KernelGaussianWrapperAngle(guassianRelationshipStruct* rss, guassianRelationshipStruct* gas, point *previouscfgs, rectangle *clearances, rectangle *offlimits, vertex *vertices, vertex *surfaceRectangle, Surface *srf, gpuConfig *gpuCfg)
{

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
	guassianRelationshipStruct* gpuRS;
	int rsSize = sizeof(guassianRelationshipStruct) * srf->nRelationships;
	checkCudaErrors(cudaMalloc(&gpuRS, rsSize));
	checkCudaErrors(cudaMemcpy(gpuRS, rss, rsSize, cudaMemcpyHostToDevice));
	guassianRelationshipStruct* gpuRA;
	int raSize = sizeof(guassianRelationshipStruct) * srf->nAngleRelationships;
	checkCudaErrors(cudaMalloc(&gpuRA, raSize));
	checkCudaErrors(cudaMemcpy(gpuRA, gas, raSize, cudaMemcpyHostToDevice));
	point *gpuPointArray;
	int pointArraySize = srf->nObjs * sizeof(point) * gpuCfg->gridxDim;
	point *outPointArray = (point*)malloc(pointArraySize);
	checkCudaErrors(cudaMalloc((void**)&gpuPointArray, pointArraySize));
	checkCudaErrors(cudaMemcpy(gpuPointArray, previouscfgs, pointArraySize, cudaMemcpyHostToDevice));
	resultCosts *gpuResultCosts;
	int resultCostsSize = sizeof(resultCosts) * gpuCfg->gridxDim;
	resultCosts *outResultCosts = (resultCosts *)malloc(resultCostsSize);
	checkCudaErrors(cudaMalloc((void**)&gpuResultCosts, resultCostsSize));
	curandState *d_rngStates = 0;
	checkCudaErrors(cudaMalloc((void **)&d_rngStates, gpuCfg->gridxDim * gpuCfg->blockxDim * sizeof(curandState)));

	// Initialise random number generator
	initRNG << <gpuCfg->gridxDim, gpuCfg->blockxDim >> > (d_rngStates, time(NULL));
	int share_size = gpuCfg->blockxDim / WARP_SIZE * 2 * srf->nObjs * sizeof(point) + gpuCfg->blockxDim / WARP_SIZE * 2 * sizeof(resultCosts);
	Kernel << <gpuCfg->gridxDim, gpuCfg->blockxDim, share_size >> >(gpuResultCosts, gpuPointArray, gpuRS, gpuRA, gpuClearances, gpuOfflimits, gpuVertices, gpuSurfaceRectangle, gpuSRF, gpuGpuConfig, d_rngStates);
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
		resultPointer[i].costs.AlignmentCosts = outResultCosts[i].AlignmentCosts;
		resultPointer[i].points = &(outPointArray[i * srf->nObjs]);
	}
	return resultPointer;
}

//This is the kernal wrapper for everything. Sadly, it may not be worth it to break it up like some of our other functions
extern "C" __declspec(dllexport) result* KernelWrapperGaussianAngle(relationshipStruct *rss, guassianRelationshipStruct *gas, point *previouscfgs, rectangle *clearances, rectangle *offlimits, vertex *vertices, vertex *surfaceRectangle, Surface *srf, gpuConfig *gpuCfg)
{

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
	relationshipStruct* gpuRS;
	int rsSize = sizeof(relationshipStruct) * srf->nRelationships;
	checkCudaErrors(cudaMalloc(&gpuRS, rsSize));
	checkCudaErrors(cudaMemcpy(gpuRS, rss, rsSize, cudaMemcpyHostToDevice));
	guassianRelationshipStruct* gpuRA;
	int raSize = sizeof(guassianRelationshipStruct) * srf->nAngleRelationships;
	checkCudaErrors(cudaMalloc(&gpuRA, raSize));
	checkCudaErrors(cudaMemcpy(gpuRA, gas, raSize, cudaMemcpyHostToDevice));
	point *gpuPointArray;
	int pointArraySize = srf->nObjs * sizeof(point) * gpuCfg->gridxDim;
	point *outPointArray = (point*)malloc(pointArraySize);
	checkCudaErrors(cudaMalloc((void**)&gpuPointArray, pointArraySize));
	checkCudaErrors(cudaMemcpy(gpuPointArray, previouscfgs, pointArraySize, cudaMemcpyHostToDevice));
	resultCosts *gpuResultCosts;
	int resultCostsSize = sizeof(resultCosts) * gpuCfg->gridxDim;
	resultCosts *outResultCosts = (resultCosts *)malloc(resultCostsSize);
	checkCudaErrors(cudaMalloc((void**)&gpuResultCosts, resultCostsSize));
	curandState *d_rngStates = 0;
	checkCudaErrors(cudaMalloc((void **)&d_rngStates, gpuCfg->gridxDim * gpuCfg->blockxDim * sizeof(curandState)));

	// Initialise random number generator
	initRNG << <gpuCfg->gridxDim, gpuCfg->blockxDim >> > (d_rngStates, time(NULL));
	int share_size = gpuCfg->blockxDim / WARP_SIZE * 2 * srf->nObjs * sizeof(point) + gpuCfg->blockxDim / WARP_SIZE * 2 * sizeof(resultCosts);
	Kernel << <gpuCfg->gridxDim, gpuCfg->blockxDim, share_size >> >(gpuResultCosts, gpuPointArray, gpuRS, gpuRA, gpuClearances, gpuOfflimits, gpuVertices, gpuSurfaceRectangle, gpuSRF, gpuGpuConfig, d_rngStates);
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
		resultPointer[i].costs.AlignmentCosts = outResultCosts[i].AlignmentCosts;
		resultPointer[i].points = &(outPointArray[i * srf->nObjs]);
	}
	return resultPointer;
}

//This ends our wrappers
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
	const int NClearances = 3*N;
	Surface srf;
	srf.nObjs = N;
	srf.nRelationships = NRel;
	srf.nAngleRelationships = 0;
	srf.nClearances = NClearances;
	srf.WeightFocalPoint = 1.0f;
	srf.WeightPairWise = 1.0f;
	srf.WeightVisualBalance = 1.0f;
	srf.WeightSymmetry = 1.0f;
	srf.WeightClearance = 1.0f;
	srf.WeightSurfaceArea = 1.0f;
	srf.WeightOffLimits = 1.0f;
	srf.WeightAlignment = 1.0f;
	srf.centroidX = 5.0;
	srf.centroidY = 5.0;
	srf.focalX = 5.0;
	srf.focalY = 5.0;
	srf.focalRot = 0.0;

	const int dimensions = 4;

	gpuConfig gpuCfg;

	gpuCfg.gridxDim = dimensions;
	gpuCfg.gridyDim = 0;
	gpuCfg.blockxDim = 4*WARP_SIZE;
	gpuCfg.blockyDim = 0;
	gpuCfg.blockzDim = 0;
	gpuCfg.iterations = 1000;//a 10th of what they claimed in the paper

	vertex surfaceRectangle[4];
	surfaceRectangle[0].x =  10;
	surfaceRectangle[0].y =  10;
	surfaceRectangle[0].z =   0;

	surfaceRectangle[1].x =  10;
	surfaceRectangle[1].y =   0;
	surfaceRectangle[1].z =   0;

	surfaceRectangle[2].x =	  0;
	surfaceRectangle[2].y =   0;
	surfaceRectangle[2].z =   0;

	surfaceRectangle[3].x =   0;
	surfaceRectangle[3].y =	 10;
	surfaceRectangle[3].z =   0;

	const int vertices = (N + NClearances) * 4;
	vertex vtx[vertices];
	//clearences were points assuming the object was at 0. Now they are vectors considered from the bounding box of the object
	//which is how it is written in the paper
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
		offlimits[i].SourceIndex = i;
	}

	point cfg[N*dimensions];
	for (int i = 0; i < dimensions; i++) {
		for (unsigned int j = 0; j < N; j++)
		{
			// BlockId counts from 0, so to properly multiply
			int index = i * N + j;

			cfg[index].freedom[0] = -6.4340348243713379;
			cfg[index].freedom[1] = -2.12361741065979;
			cfg[index].freedom[2] = 0.0;
			cfg[index].freedom[3] = 0.0;
			cfg[index].freedom[4] = 5.5179219245910645;
			cfg[index].frozen = false;
			cfg[index].length = 1.7677083015441895;
			cfg[index].width = 2.2480521202087402;
			cfg[index].height = 1.0002f;
		}
	}

	// Create relationship
	//relationshipStruct rss[1];
	//rss[0].TargetRange.targetRangeStart = 2.0;
	//rss[0].TargetRange.targetRangeEnd = 4.0;
	//rss[0].AngleRange.targetRangeStart = 0.01*PI;
	//rss[0].AngleRange.targetRangeEnd = PI;
	//rss[0].DegreesOfAtrraction = 2;
	//rss[0].SourceIndex = 0;
	//rss[0].TargetIndex = 1;

	guassianRelationshipStruct rss[1];
	rss->mean[0] = 0;
	rss->deviation[0] = 1;
	rss->mean[1] = 0;
	rss->deviation[1] = 1;
	rss->mean[2] = 0;
	rss->deviation[2] = 0;
	rss->SourceIndex = 0;
	rss->TargetIndex = 1;

	//for (int i = 0; i < NRel; i++) {
	//	rss[i].TargetRange.targetRangeStart = 0.0;
	//	rss[i].TargetRange.targetRangeEnd = 2.0;
	//	rss[i].Source.x = 0.0 + i;
	//	rss[i].Source.y = 0.0 + i;
	//	rss[i].Source.z = 0.0;
	//	rss[i].Source.rotX = 1.0;
	//	rss[i].Source.freedom[4] = 1.0;
	//	rss[i].Source.rotZ = 1.0;
	//	rss[i].Target.x = 3.0 + i;
	//	rss[i].Target.y = 3.0 + i;
	//	rss[i].Target.z = 0.0;
	//	rss[i].Target.rotX = 1.0;
	//	rss[i].Target.freedom[4] = 1.0;
	//	rss[i].Target.rotZ = 1.0;
	//	rss[i].DegreesOfAtrraction = 2.0;
	//}

	// Point test code:
	result *result = KernelGaussianWrapper(rss, cfg, clearances, offlimits, vtx, surfaceRectangle, &srf, &gpuCfg);
	printf("Results:\n");
	for (int i = 0; i < gpuCfg.gridxDim; i++)
	{
		printf("Result %d\n", i);
		for (int j = 0; j < srf.nObjs; j++) {
			printf("Point [%d] X,Y,Z: %f, %f, %f	Rotation: %f, %f\n",
				j,
				result[i].points[j].freedom[0],
				result[i].points[j].freedom[1],
				result[i].points[j].freedom[2],
				result[i].points[j].freedom[3],
				result[i].points[j].freedom[4]);
		}
		printf("Costs are %f+%f+%f+%f+%f+%f+%f+%f=%f\n",
			result[i].costs.FocalPointCosts,
			result[i].costs.PairWiseCosts,
			result[i].costs.SymmetryCosts,
			result[i].costs.VisualBalanceCosts,
			result[i].costs.ClearanceCosts,
			result[i].costs.OffLimitsCosts,
			result[i].costs.SurfaceAreaCosts,
			result[i].costs.AlignmentCosts,
			result[i].costs.totalCosts);
	}
	//system("PAUSE");
 	return EXIT_SUCCESS;
}