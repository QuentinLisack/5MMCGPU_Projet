__kernel void voronoi(
   __global unsigned int* points,
   __global unsigned int* original_positions,
   const unsigned int N,
   const unsigned int NumPoints)
{    
	
	int i = get_global_id(0);
    int j = get_global_id(1);

	unsigned int x, y, bestPoint, bestDist, tempDist, currPos;
	currPos = i + N*j;
	
	bestPoint = 1;
	x = original_positions[0]%N;
	y = (original_positions[0] - x)/N;
	bestDist = (x-i)*(x-i) + (y-j)*(y-j);
	
	for(int p = 1; p<NumPoints; p++){
		x = original_positions[p] % N;
		y = (original_positions[p] - x)/N;
		tempDist = (x-i)*(x-i) + (y-j)*(y-j);
		if(tempDist < bestDist){
			bestPoint = p+1;
			bestDist = tempDist;
		}
	}
	points[currPos] = bestPoint;
	
}
