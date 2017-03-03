/*
 **  PROGRAM: Approximation of pi
 **
 **  PURPOSE: This program will numerically compute the integral of
 **           4/(1+x*x)
 **
 **           from 0 to 1. The value of this integral is pi.
 **           The is the original sequential program. It uses the timer
 **           from the OpenMP runtime library
 **
 **  USAGE: ./pi
 **
 */

#include "util.hpp"

#include <iostream>
static long num_steps = 100000000;
double step;
extern double wtime();   // returns time since some fixed past point (wtime.c)

int main ()
{
  int i;

  double x, pi, sum = 0.0;


  step = 1.0/(double) num_steps;

  util::Timer timer;

  for (i=1;i<= num_steps; i++){
    x = (i-0.5)*step;
    sum = sum + 4.0/(1.0+x*x);
  }

  pi = step * sum;
  double run_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
  std::cout<<"pi with "<<num_steps<<" steps is "
    << pi <<" in "
    <<run_time<<" seconds"<<std::endl;
}

