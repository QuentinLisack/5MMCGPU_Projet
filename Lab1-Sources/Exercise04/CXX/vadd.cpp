/* ----------------------------------------------------------------
**
** Name:       vadd_cpp.cpp
**
** Purpose:    Elementwise addition of two vectors (c = a + b)
**
**                   c = a + b
**
** ----------------------------------------------------------------
*/

#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"

#include "util.hpp" // utility library

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <iostream>
#include <fstream>

// pick up device type from compiler command line or from the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

#include <err_code.h>

// ----------------------------------------------------------------

#define TOL    (0.001)   // tolerance used in floating point comparisons
#define LENGTH (1024)    // length of vectors a, b, and c

int main(void)
{
  std::vector<float> h_a(LENGTH);                // a vector
  std::vector<float> h_b(LENGTH);                // b vector
  std::vector<float> h_c(LENGTH);               // c = a + b, from compute device
  std::vector<float> h_d(LENGTH);               // d = c + e, from compute device 
  std::vector<float> h_e(LENGTH);               // e vector 
  std::vector<float> h_f(LENGTH);               // f = d + g, from compute device
  std::vector<float> h_g(LENGTH);               //g vector

  cl::Buffer d_a;                        // device memory used for the input  a vector
  cl::Buffer d_b;                        // device memory used for the input  b vector
  cl::Buffer d_c;                       // device memory used for the output c vector
  cl::Buffer d_e;                        // device memory used for the input  e vector
  cl::Buffer d_g;                        // device memory used for the input  g vector

  // Fill vectors a, b, e and g with random float values
  int count = LENGTH;
  for(int i = 0; i < count; i++)
  {
    h_a[i]  = rand() / (float)RAND_MAX;
    h_b[i]  = rand() / (float)RAND_MAX;
    h_e[i]  = rand() / (float)RAND_MAX;
    h_g[i]  = rand() / (float)RAND_MAX;
  }

  try
  {
    // Create a context
    cl::Context context(DEVICE);

    // Load in kernel source, creating a program object for the context
    cl::Program program(context, util::loadProgram("vadd.cl"), true);

    // Get the command queue
    cl::CommandQueue queue(context);

    // Create the kernel functor

    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int> vadd(program, "vadd");

    d_a   = cl::Buffer(context, h_a.begin(), h_a.end(), true);
    d_b   = cl::Buffer(context, h_b.begin(), h_b.end(), true);
    d_e   = cl::Buffer(context, h_e.begin(), h_e.end(), true);
    d_g   = cl::Buffer(context, h_g.begin(), h_g.end(), true);

    d_c  = cl::Buffer(context,  CL_MEM_READ_WRITE, sizeof(float) * LENGTH);

    util::Timer timer;

    vadd( cl::EnqueueArgs( queue, cl::NDRange(count)),
        d_a, d_b, d_c, count);
    
    cl::copy(queue, d_c, h_c.begin(), h_c.end());
        
    vadd( cl::EnqueueArgs( queue, cl::NDRange(count)),
        d_e, d_c, d_c, count);
        
    cl::copy(queue, d_c, h_d.begin(), h_d.end());
    
    vadd( cl::EnqueueArgs( queue, cl::NDRange(count)),
        d_g, d_c, d_c, count);
    
    queue.finish();
        
    cl::copy(queue, d_c, h_f.begin(), h_f.end());

    double rtime = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
    std::cout<<"The kernels ran in "<<rtime <<" seconds"<<std::endl;

    // Test the results
    int correct1 = 0;
    int correct2 = 0;
    int correct3 = 0;
    float tmp;
    for(int i = 0; i < count; i++) {
      tmp = h_a[i] + h_b[i]; // expected value for d_c[i]
      tmp -= h_c[i];         // compute errors
      if(tmp*tmp < TOL*TOL) {
        // correct if square deviation is less
        correct1++;  //  than tolerance squared
      }
      else {
        std::cout<<"tmp "<<tmp <<", h_a " << h_a[i]
          << ", h_b " << h_b[i] << ", h_c "<<h_c[i]<<std::endl;
      }
      tmp = h_c[i] + h_e[i]; // expected value for d_c[i]
      tmp -= h_d[i];         // compute errors
      if(tmp*tmp < TOL*TOL) {
        // correct if square deviation is less
        correct2++;  //  than tolerance squared
      }
      else {
        std::cout<<"tmp "<<tmp <<", h_c " << h_c[i]
          << ", h_e " << h_e[i] << ", h_d "<<h_d[i]<<std::endl;
      }
      tmp = h_d[i] + h_g[i]; // expected value for d_c[i]
      tmp -= h_f[i];         // compute errors
      if(tmp*tmp < TOL*TOL) {
        // correct if square deviation is less
        correct3++;  //  than tolerance squared
      }
      else {
        std::cout<<"tmp "<<tmp <<", h_d " << h_d[i]
          << ", h_g " << h_g[i] << ", h_f "<<h_f[i]<<std::endl;
      }
    }

    // summarize results
    std::cout<< "vector add to find C = A+B: " << correct1 <<" "
      << "out of "<<count<<"results were correct."<< std::endl;
    std::cout<< "vector add to find D = E+F: " << correct2 <<" "
      << "out of "<<count<<"results were correct."<< std::endl;
    std::cout<< "vector add to find F = D+G: " << correct3 <<" "
      << "out of "<<count<<"results were correct."<< std::endl;
  }
  catch (cl::Error err) {
    std::cout << "Exception\n";
    std::cerr << "ERROR: " << err.what()
      << "(" << err_code(err.err()) << ")"
      << std::endl;
  }
}
