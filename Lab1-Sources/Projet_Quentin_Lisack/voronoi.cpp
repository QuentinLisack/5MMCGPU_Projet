#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"

#include "util.hpp" // utility library
#include <png++/png.hpp>

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <time.h> 

// pick up device type from compiler command line or from the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

#include <err_code.h>
#include "device_picker.hpp"

#define LENGTH (1024)    // taille de la grille de calcul pour le voronoi
#define NUMPOINTS (1000)  // nombre de points (<LENGTH)

int main(int argc, char *argv[]){
    std::vector<unsigned int> h_grid_p(LENGTH*LENGTH); // buffer which contains the number of the closest point
    std::vector<unsigned int> h_grid(LENGTH*LENGTH); // contains the coordinate of the closest point
    std::vector<unsigned int> h_points(NUMPOINTS); // contains the seeds
    std::vector<unsigned int> h_grid_seeds(LENGTH*LENGTH); // contains the seeds
    
    //sequential veriables
    std::vector<unsigned int> seq_points(LENGTH*LENGTH); // buffer which contains the number of the closest point
    std::vector<unsigned int> seq_positions(LENGTH*LENGTH); // contains the coordinate of the closest point
    std::vector<unsigned int> seq_original(NUMPOINTS); // contains the seeds
    
    cl::Buffer d_grid_points, d_grid_positions, d_grid_original, d_grid; // buffers for the kernel

	std::vector<png::rgb_pixel> col(NUMPOINTS);
    
    srand (time(NULL));
    double start_time;      // Starting time
    double run_time;        // Timing
    util::Timer timer;      // Timing
    
    //INITIALISATION DES POINTS
    for(int i = 0; i < NUMPOINTS; i++){
    	h_points[i] = rand() % (LENGTH*LENGTH);
    	seq_original[i] = h_points[i];
    	col[i] = png::rgb_pixel(rand() % 256, rand() % 256, rand() % 256);
    }
    
    //initialisation des matrices
    for(unsigned int i = 0; i<LENGTH; i++){
    	h_grid_p[i] = 0;
    	h_grid[i] = 0;
    	seq_points[i] = 0;
    	seq_positions[i] = 0;
    }
    for(unsigned int i = 0; i< NUMPOINTS; i++){
    	h_grid_p[h_points[i]] = i + 1;
    	h_grid[h_points[i]] = h_points[i];
    	seq_points[h_points[i]] = i + 1;
    	seq_positions[h_points[i]] = h_points[i];
    }
    
    png::rgb_pixel B(0, 0, 255);
    png::rgb_pixel G(0, 255, 0);
    png::rgb_pixel R(255, 0, 0);
    png::image<png::rgb_pixel> image(LENGTH, LENGTH);
    

    
        
        
        
// ------------------------------------------------------------------
// propagation 1
// ------------------------------------------------------------------
//basique : on effectue les n étapes dans le kernel, et on voit les problèmes de synchronisation.
// on utilise une "liste" pour stocker les graines
// plusieurs matrices utilisées
// barrière utilisée mais inefficace
		
	try
    {   
        // Create a context
    	cl::Context context(DEVICE);
        
        d_grid_points = cl::Buffer(context, h_grid_p.begin(), h_grid_p.end(), true);
        d_grid_positions = cl::Buffer(context, h_grid.begin(), h_grid.end(), true);
        d_grid_original = cl::Buffer(context, h_points.begin(), h_points.end(), true);

        timer.reset();
		
        // Create the compute program from the source buffer
        cl::Program program(context, util::loadProgram("voronoi1.cl"), true);

        // Get the command queue
    	cl::CommandQueue queue(context);

        // Create the compute kernel from the program
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, unsigned int, unsigned int> voronoi(program, "voronoi");
        
        start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

        cl::NDRange global(LENGTH, LENGTH);
        voronoi(cl::EnqueueArgs(queue, global), d_grid_points, d_grid_positions, d_grid_original, LENGTH, NUMPOINTS);

        queue.finish();

        run_time  = (static_cast<double>(timer.getTimeMilliseconds()) / 1000.0) - start_time;
        
        std::cout << "the kernel 1 ran in " << run_time << " seconds"<< std::endl;

        cl::copy(queue, d_grid_points, h_grid_p.begin(), h_grid_p.end());
        
    } catch (cl::Error err)
    {
        std::cout << "Exception\n";
        std::cerr << "ERROR: "
                  << err.what()
                  << "("
                  << err_code(err.err())
                  << ")"
                  << std::endl;
    }
    
    for (png::uint_32 y = 0; y < image.get_height(); ++y){
		for (png::uint_32 x = 0; x < image.get_width(); ++x){
			image[y][x] = R;
			for(int i = 0; i< NUMPOINTS; i++){
				if(h_grid_p[x + LENGTH * y] == i+1){
					image[y][x] = col[i];
				}
			}
		}
	}
	image.write("outputPar1.png");
    
// ------------------------------------------------------------------
// propagation 2
// ------------------------------------------------------------------
// on effectue une étape à la fois dans le kernel et on synchronise via le CPU : disparition des problèmes de synchronisation
// on utilise une "liste" pour stocker les graines
// plusieurs matrices utilisées
// barrière utilisée
    
    //réisnitialisation des matrices
    for(unsigned int i = 0; i<LENGTH; i++){
    	h_grid_p[i] = 0;
    	h_grid[i] = 0;
    }
    for(unsigned int i = 0; i< NUMPOINTS; i++){
    	h_grid_p[h_points[i]] = i + 1;
    	h_grid[h_points[i]] = h_points[i];
    }
    
    try{
    	cl::Context context(DEVICE);
    	
    	d_grid_points = cl::Buffer(context, h_grid_p.begin(), h_grid_p.end(), true);
        d_grid_positions = cl::Buffer(context, h_grid.begin(), h_grid.end(), true);
        d_grid_original = cl::Buffer(context, h_points.begin(), h_points.end(), true);
    	
    	timer.reset();
		
        // Create the compute program from the source buffer
        cl::Program program(context, util::loadProgram("voronoi2.cl"), true);

        // Get the command queue
    	cl::CommandQueue queue(context);

        // Create the compute kernel from the program
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, unsigned int, unsigned int, unsigned int> voronoi(program, "voronoi");
        
        start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

        cl::NDRange global(LENGTH, LENGTH);
        
        int step = LENGTH/2;
        
        while(step > 0){
		    voronoi(cl::EnqueueArgs(queue, global), d_grid_points, d_grid_positions, d_grid_original, LENGTH, NUMPOINTS, step);
		    queue.finish();
		    step = step/2;
		}

        run_time  = (static_cast<double>(timer.getTimeMilliseconds()) / 1000.0) - start_time;
        
        std::cout << "the kernel 2 ran in " << run_time << " seconds"<< std::endl;

        cl::copy(queue, d_grid_points, h_grid_p.begin(), h_grid_p.end());
    	
    } catch (cl::Error err)
    {
        std::cout << "Exception\n";
        std::cerr << "ERROR: "
                  << err.what()
                  << "("
                  << err_code(err.err())
                  << ")"
                  << std::endl;
    }
    
    for (png::uint_32 y = 0; y < image.get_height(); ++y){
		for (png::uint_32 x = 0; x < image.get_width(); ++x){
			image[y][x] = R;
			for(int i = 0; i< NUMPOINTS; i++){
				if(h_grid_p[x + LENGTH * y] == i+1){
					image[y][x] = col[i];
				}
			}
		}
	}
	image.write("outputPar2.png");
	
// ------------------------------------------------------------------
// propagation 3
// ------------------------------------------------------------------
// on effectue une étape à la fois dans le kernel et on synchronise via le CPU : disparition des problèmes de synchronisation
// on utilise une matrice booléenne pour stocker les graines
// plusieurs matrices utilisées
// pas de barrière
    
    //réisnitialisation des matrices
    for(unsigned int i = 0; i<LENGTH; i++){
    	h_grid_p[i] = 0;
    	h_grid[i] = 0;
    	h_grid_seeds[i] = 0;
    }
    for(unsigned int i = 0; i< NUMPOINTS; i++){
    	h_grid_p[h_points[i]] = i + 1;
    	h_grid[h_points[i]] = h_points[i];
    	h_grid_seeds[i] = 1;
    }
    
    try{
    	cl::Context context(DEVICE);
    	
    	d_grid_points = cl::Buffer(context, h_grid_p.begin(), h_grid_p.end(), true);
        d_grid_positions = cl::Buffer(context, h_grid.begin(), h_grid.end(), true);
        d_grid_original = cl::Buffer(context, h_grid_seeds.begin(), h_grid_seeds.end(), true);
    	
    	timer.reset();
		
        // Create the compute program from the source buffer
        cl::Program program(context, util::loadProgram("voronoi3.cl"), true);

        // Get the command queue
    	cl::CommandQueue queue(context);

        // Create the compute kernel from the program
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, unsigned int, unsigned int, unsigned int> voronoi(program, "voronoi");
        
        start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

        cl::NDRange global(LENGTH, LENGTH);
        
        int step = LENGTH/2;
        
        while(step > 0){
		    voronoi(cl::EnqueueArgs(queue, global), d_grid_points, d_grid_positions, d_grid_original, LENGTH, NUMPOINTS, step);
		    queue.finish();
		    step = step/2;
		}

        run_time  = (static_cast<double>(timer.getTimeMilliseconds()) / 1000.0) - start_time;
        
        std::cout << "the kernel 3 ran in " << run_time << " seconds"<< std::endl;

        cl::copy(queue, d_grid_points, h_grid_p.begin(), h_grid_p.end());
    	
    } catch (cl::Error err)
    {
        std::cout << "Exception\n";
        std::cerr << "ERROR: "
                  << err.what()
                  << "("
                  << err_code(err.err())
                  << ")"
                  << std::endl;
    }
    
    for (png::uint_32 y = 0; y < image.get_height(); ++y){
		for (png::uint_32 x = 0; x < image.get_width(); ++x){
			image[y][x] = R;
			for(int i = 0; i< NUMPOINTS; i++){
				if(h_grid_p[x + LENGTH * y] == i+1){
					image[y][x] = col[i];
				}
			}
		}
	}
	image.write("outputPar3.png");
	
// ------------------------------------------------------------------
// propagation 4
// ------------------------------------------------------------------
// on effectue une étape à la fois dans le kernel et on synchronise via le CPU : disparition des problèmes de synchronisation
// une seule matrice utilisée (données agrégées via un modulo N=LENGTH : x + N*y + N*N*point + N*N*N*seed)
    
    //réisnitialisation des matrices
    for(unsigned int i = 0; i<LENGTH; i++){
    	h_grid[i] = 0;
    }
    for(unsigned int i = 0; i< NUMPOINTS; i++){
    	h_grid[h_points[i]] = h_points[i] + LENGTH*LENGTH*(i + 1) + LENGTH*LENGTH*LENGTH;
    }
    
    try{
    	cl::Context context(DEVICE);
    	
        d_grid = cl::Buffer(context, h_grid.begin(), h_grid.end(), true);
    	
    	timer.reset();
		
        // Create the compute program from the source buffer
        cl::Program program(context, util::loadProgram("voronoi4.cl"), true);

        // Get the command queue
    	cl::CommandQueue queue(context);

        // Create the compute kernel from the program
        cl::make_kernel<cl::Buffer, unsigned int, unsigned int> voronoi(program, "voronoi");
        
        start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

        cl::NDRange global(LENGTH, LENGTH);
        
        int step = LENGTH/2;
        
        while(step > 0){
		    voronoi(cl::EnqueueArgs(queue, global), d_grid, LENGTH, step);
		    queue.finish();
		    step = step/2;
		}

        run_time  = (static_cast<double>(timer.getTimeMilliseconds()) / 1000.0) - start_time;
        
        std::cout << "the kernel 4 ran in " << run_time << " seconds"<< std::endl;

        cl::copy(queue, d_grid, h_grid.begin(), h_grid.end());
    	
    } catch (cl::Error err)
    {
        std::cout << "Exception\n";
        std::cerr << "ERROR: "
                  << err.what()
                  << "("
                  << err_code(err.err())
                  << ")"
                  << std::endl;
    }
    
    for(int i = 0; i < LENGTH * LENGTH; i++){
    	int temp = h_grid[i]%(LENGTH*LENGTH);
    	h_grid[i] = ((h_grid[i] - temp)%(LENGTH*LENGTH*LENGTH))/(LENGTH*LENGTH);
    }
    
    for (png::uint_32 y = 0; y < image.get_height(); ++y){
		for (png::uint_32 x = 0; x < image.get_width(); ++x){
			image[y][x] = R;
			for(int i = 0; i< NUMPOINTS; i++){
				if(h_grid_p[x + LENGTH * y] == i+1){
					image[y][x] = col[i];
				}
			}
		}
	}
	image.write("outputPar4.png");
	
// ------------------------------------------------------------------
// propagation 5
// ------------------------------------------------------------------
// version naive sur GPU : on teste pour chaque point toutes les graines.
    
    //réisnitialisation des matrices
    for(unsigned int i = 0; i<LENGTH; i++){
    	h_grid_p[i] = 0;
    }
    
    try{
    	cl::Context context(DEVICE);
    	
    	d_grid_points = cl::Buffer(context, h_grid_p.begin(), h_grid_p.end(), true);
        d_grid_original = cl::Buffer(context, h_points.begin(), h_points.end(), true);
    	
    	timer.reset();
		
        // Create the compute program from the source buffer
        cl::Program program(context, util::loadProgram("voronoi5.cl"), true);

        // Get the command queue
    	cl::CommandQueue queue(context);

        // Create the compute kernel from the program
        cl::make_kernel<cl::Buffer, cl::Buffer, unsigned int, unsigned int> voronoi(program, "voronoi");
        
        start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

        cl::NDRange global(LENGTH, LENGTH);
        
	    voronoi(cl::EnqueueArgs(queue, global), d_grid_points, d_grid_original, LENGTH, NUMPOINTS);
	    queue.finish();

        run_time  = (static_cast<double>(timer.getTimeMilliseconds()) / 1000.0) - start_time;
        
        std::cout << "the kernel 5 ran in " << run_time << " seconds"<< std::endl;

        cl::copy(queue, d_grid_points, h_grid_p.begin(), h_grid_p.end());
    	
    } catch (cl::Error err)
    {
        std::cout << "Exception\n";
        std::cerr << "ERROR: "
                  << err.what()
                  << "("
                  << err_code(err.err())
                  << ")"
                  << std::endl;
    }
    
    for (png::uint_32 y = 0; y < image.get_height(); ++y){
		for (png::uint_32 x = 0; x < image.get_width(); ++x){
			image[y][x] = R;
			for(int i = 0; i< NUMPOINTS; i++){
				if(h_grid_p[x + LENGTH * y] == i+1){
					image[y][x] = col[i];
				}
			}
		}
	}
	image.write("outputPar5.png");
    
// ------------------------------------------------------------------
// sequential naive 
// ------------------------------------------------------------------  

	timer.reset();

    unsigned int seqRes[LENGTH][LENGTH];
    for(unsigned int i = 0; i<LENGTH; i++){
    	for(unsigned int j = 0; j<LENGTH; j++){
    	
    		unsigned int x, y, bestPoint, bestDist, tempDist, currPos;
	
			bestPoint = 1;
			x = h_points[0]%LENGTH;
			y = (h_points[0] - x)/LENGTH;
			bestDist = (x-i)*(x-i) + (y-j)*(y-j);
	
			for(int p = 1; p<NUMPOINTS; p++){
				x = h_points[p] % LENGTH;
				y = (h_points[p] - x)/LENGTH;
				tempDist = (x-i)*(x-i) + (y-j)*(y-j);
				if(tempDist < bestDist){
					bestPoint = p+1;
					bestDist = tempDist;
				}
			}
    		seqRes[i][j] = bestPoint;
    	}
    }
    
    run_time  = (static_cast<double>(timer.getTimeMilliseconds()) / 1000.0) - start_time;
        
    std::cout << "the sequential naive ran in " << run_time << " seconds"<< std::endl;
    
// ------------------------------------------------------------------
// sequential jump flood 
// ------------------------------------------------------------------ 

	timer.reset();

	unsigned int NumPoints = NUMPOINTS;
    unsigned int N = LENGTH;
    
    std::vector<unsigned int> tempPosRes(LENGTH*LENGTH);
    std::vector<unsigned int> tempPointsRes(LENGTH*LENGTH);

	int step = N/2;
    
    unsigned int pos_x2, pos_x1, pos_y1, pos_y2, lookedAtPos, tempPos1, tempPos2, currPos, bestPos, bestPoint, bestDist;
    float tempDistbis;
    bool isNotSeed, isChanged;
    
    
    while(step >= 1){
    	
    	for(int i = 0; i<LENGTH*LENGTH; i++){
			tempPosRes[i] = 0;
			tempPointsRes[i] = 0;
		}
    	
		for(int i = 0; i<LENGTH; i++){
			for(int j = 0; j<LENGTH; j++){
				currPos = i + N*j;
				isChanged = false;
				//test si c'est une graine
		    	isNotSeed = true;
		    	for(int i = 0; i < NumPoints; i++){
					if(currPos == seq_original[i]){
						isNotSeed = false;
					}
				}
				if(isNotSeed){
					tempPos1 = seq_positions[i + N*j];
			        pos_x1 = tempPos1 % N;
			        pos_y1 = (tempPos1 - pos_x1)/N;
			        bestDist = (pos_x1 - i)*(pos_x1 - i) + (pos_y1 - j)*(pos_y1 - j);
					for(int n = -1; n <= 1; n++){
						for(int m = -1; m <= 1; m++){
							//JFA
							if(i + n*step < N && i + n*step >= 0 && j + m*step < N && j + m*step >= 0){// si on est dans la grille
								lookedAtPos = ((i + n*step)) + N*((j + m*step));
								if(seq_points[lookedAtPos] > 0 && (m != 0 || n != 0)){ //si le point qu'on regarde est initialisé, et est différent du point courant. Sinon, on ne fait rien.
									tempPos2 = seq_positions[lookedAtPos];
								    pos_x2 = tempPos2 % N;
								    pos_y2 = (tempPos2 - pos_x2)/N;
								    if(seq_points[currPos] == 0 && !isChanged){//si le point courant n'est pas initialisé, on le remplit avec le premier point qu'on croise
								    	isChanged = true;
								        bestPoint = seq_points[lookedAtPos];
								        bestPos = tempPos2;
								        bestDist = (pos_x2 - i)*(pos_x2 - i) + (pos_y2 - j)*(pos_y2 - j);
								    } else {//sinon on compare avec ce qu'on a déjà
								        tempDistbis = (pos_x2 - i)*(pos_x2 - i) + (pos_y2 - j)*(pos_y2 - j);
								        if(bestDist > tempDistbis){
								        	isChanged = true;
								            bestPoint = seq_points[lookedAtPos];
								            bestPos = tempPos2;
								            bestDist = (pos_x2 - i)*(pos_x2 - i) + (pos_y2 - j)*(pos_y2 - j);
								        }
								    }
								}
							}
						}
					}
				}
				if(isChanged){
					tempPointsRes[currPos] = bestPoint;
					tempPosRes[currPos] = bestPos;
				}
			}
		}
		step = step/2;
		for(int p = 0; p < LENGTH * LENGTH; p++){
			if(tempPointsRes[p] > 0){
				seq_points[p] = tempPointsRes[p];
				seq_positions[p] = tempPosRes[p];
			}
		}
    }
    
    run_time  = (static_cast<double>(timer.getTimeMilliseconds()) / 1000.0) - start_time;
        
    std::cout << "the sequential JFA ran in " << run_time << " seconds"<< std::endl;
    
// ------------------------------------------------------------------
// output the results in images
// ------------------------------------------------------------------
	
	for (png::uint_32 y = 0; y < image.get_height(); ++y){
		for (png::uint_32 x = 0; x < image.get_width(); ++x){
			image[y][x] = R;
			for(int i = 0; i< NUMPOINTS; i++){
				if(seqRes[x][y] == i+1){
					image[y][x] = col[i];
				}
			}
		}
	}
	image.write("outputSeqNaive.png");
	
	for (png::uint_32 y = 0; y < image.get_height(); ++y){
		for (png::uint_32 x = 0; x < image.get_width(); ++x){
			image[y][x] = R;
			for(int i = 0; i< NUMPOINTS; i++){
				if(seq_points[x + LENGTH * y] == i+1){
					image[y][x] = col[i];
				}
			}
		}
	}
	image.write("outputSeqJFA.png");
}


