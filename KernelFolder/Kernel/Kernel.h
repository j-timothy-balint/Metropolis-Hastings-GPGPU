
struct vertex
{
	float x;
	float y;
	float z;
};

struct BoundingBox //added because several huristics use a min and max point, so this will keep things more cleaned
{
	vertex minPoint;
	vertex maxPoint;
};

//We sort group in place, and then can make assumptions about the objects in the group
struct Group
{
	int SourceIndex;
	int TargetIndex;
};

struct rectangle
{
	int point1Index;
	int point2Index;
	int point3Index;
	int point4Index;
	int SourceIndex;
};

struct targetRangeStruct {
	float targetRangeStart;
	float targetRangeEnd;
};

struct relationshipStruct
{
	targetRangeStruct TargetRange;
	int SourceIndex;
	int TargetIndex;
	int DegreesOfAtrraction;
};

//We have 5 degrees of freedom. max we will combine is 3 (x,y,z) 
struct gaussianRelationshipStruct
{
	int SourceIndex;
	int TargetIndex;
	float mean[3];
	float deviation[3];
};
//We've defined a few pairwise choices, guassian vs euclidean and single vs total. 
//These control the values sent into our cost function for pairwise
struct PairWiseChoices 
{
	bool total;
	bool gaussian;
};

struct Surface
{
	int nObjs; //Total number of objects we are solidifying
	int nRelationships;//Total number of relationships between objects
	int nAngleRelationships; //Total number of relationships for angles
	int nClearances;//Total number of clearances between objects
	int nGroups;//Total number of groups

	//Our choice commands
	PairWiseChoices pwChoices[2];

	// Weights
	float WeightFocalPoint;
	float WeightPairWise;
	float WeightVisualBalance;
	float WeightSymmetry;
	float WeightOffLimits;
	float WeightClearance;
	float WeightSurfaceArea;
	float WeightAlignment;

	// Centroid
	float centroidX;
	float centroidY;

	// Focal point
	float focalX;
	float focalY;
	float focalRot;
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
	//float x, y, z, rotX, rotY, rotZ; //moved these to degrees of freedom for our change
	float freedom[5]; //x,y,z,rotX,rotY
	bool frozen;

	float length;
	float width;
	float height;
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
	float AlignmentCosts;
};

struct result {
	point *points;
	resultCosts costs;
};



extern "C" __declspec(dllexport) result* KernelWrapper(relationshipStruct *rss, gaussianRelationshipStruct* gss, point *previouscfgs, rectangle *clearances, rectangle *offlimits, vertex *vertices, vertex *surfaceRectangle, Surface *srf, Group* groups, gpuConfig *gpuCfg);

__declspec(dllexport) void basicCudaDeviceInformation(int argc, char **argv);