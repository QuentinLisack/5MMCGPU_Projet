__kernel void voronoi(
   __global unsigned int* points,
   __global unsigned int* positions,
   __global unsigned int* original_positions,
   const unsigned int N,
   const unsigned int NumPoints,
   const unsigned int step)
{    

	int i = get_global_id(0);
    int j = get_global_id(1);
    //on regarde le point en i + N*j
    
    unsigned int pos_x2, pos_x1, pos_y1, pos_y2, lookedAtPos, tempPos1, tempPos2, currPos, bestPos, bestPoint, bestDist;
    float tempDist;
    bool isNotSeed, isChanged;
    
    
	currPos = i + N*j;
	isChanged = false;
	//test si c'est une graine
	isNotSeed = true;
	for(int i = 0; i < NumPoints; i++){
		if(currPos == original_positions[i]){
			isNotSeed = false;
		}
	}
	if(isNotSeed){
		tempPos1 = positions[currPos];
        pos_x1 = tempPos1 % N;
        pos_y1 = (tempPos1 - pos_x1)/N;
        bestDist = (pos_x1 - i)*(pos_x1 - i) + (pos_y1 - j)*(pos_y1 - j);
		for(int n = -1; n <= 1; n++){
			for(int m = -1; m <= 1; m++){
				//JFA
				if(i + n*step < N && i + n*step >= 0 && j + m*step < N && j + m*step >= 0){// si on est dans la grille
					lookedAtPos = ((i + n*step)) + N*((j + m*step));
					if(points[lookedAtPos] > 0 && (m != 0 || n != 0)){ //si le point qu'on regarde est initialisé, et est différent du point courant. Sinon, on ne fait rien.
						tempPos2 = positions[lookedAtPos];
					    pos_x2 = tempPos2 % N;
					    pos_y2 = (tempPos2 - pos_x2)/N;
					    if(points[currPos] == 0 && !isChanged){//si le point courant n'est pas initialisé, on le remplit avec le premier point qu'on croise
					    	isChanged = true;
					        bestPoint = points[lookedAtPos];
					        bestPos = tempPos2;
					        bestDist = (pos_x2 - i)*(pos_x2 - i) + (pos_y2 - j)*(pos_y2 - j);
					    } else {//sinon on compare avec ce qu'on a déjà
					        tempDist = (pos_x2 - i)*(pos_x2 - i) + (pos_y2 - j)*(pos_y2 - j);
					        if(bestDist > tempDist){
					        	isChanged = true;
					            bestPoint = points[lookedAtPos];
					            bestPos = tempPos2;
					            bestDist = (pos_x2 - i)*(pos_x2 - i) + (pos_y2 - j)*(pos_y2 - j);
					        }
					    }
					}
				}
			}
		}
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
	if(isChanged){
		points[currPos] = bestPoint;
		positions[currPos] = bestPos;
	}
}
