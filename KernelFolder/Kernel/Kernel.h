
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

struct group
{
	int *groupItemIndices;
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
struct guassianRelationshipStruct
{
	int SourceIndex;
	int TargetIndex;
	float mean[3];
	float deviation[3];
};

struct Surface
{
	int nObjs; //Total number of objects we are solidifying
	int nRelationships;//Total number of relationships between objects
	int nAngleRelationships;
	int nClearances;//Total number of clearances between objects
	int nGroups;//Total number of groups

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



extern "C" __declspec(dllexport) result* KernelWrapper(relationshipStruct *rss, point *previouscfgs, rectangle *clearances, rectangle *offlimits, vertex *vertices, vertex *surfaceRectangle, Surface *srf, gpuConfig *gpuCfg);
extern "C" __declspec(dllexport) result* KernelGaussianWrapper(guassianRelationshipStruct *rss, point *previouscfgs, rectangle *clearances, rectangle *offlimits, vertex *vertices, vertex *surfaceRectangle, Surface *srf, gpuConfig *gpuCfg);
extern "C" __declspec(dllexport) result* KernelGaussianWrapperAngle(guassianRelationshipStruct *rss, guassianRelationshipStruct *gas, point *previouscfgs, rectangle *clearances, rectangle *offlimits, vertex *vertices, vertex *surfaceRectangle, Surface *srf, gpuConfig *gpuCfg);
extern "C" __declspec(dllexport) result* KernelWrapperGaussianAngle(relationshipStruct *rss, guassianRelationshipStruct *gas, point *previouscfgs, rectangle *clearances, rectangle *offlimits, vertex *vertices, vertex *surfaceRectangle, Surface *srf, gpuConfig *gpuCfg);

__declspec(dllexport) void basicCudaDeviceInformation(int argc, char **argv);