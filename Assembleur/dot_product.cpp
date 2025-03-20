// ==================================================================
// Program: dot_product.cpp
// Date: July 2020
// Author: Jean-Michel Richer
// Email: jean-michel.richer@univ-angers.fr
// ==================================================================
// Description:
//	This program enables to test different implementations of the
// dot product of two vectors of floats
// ==================================================================
#include <iostream>
#include <string>
#include <stdint.h>
#include <getopt.h>
#include <xmmintrin.h>
#include "cpu_timer.h"
using namespace std;

// ------------------------------------------------------------------
//
// TYPES
//
// ------------------------------------------------------------------
typedef float f32;
typedef uint32_t u32;
typedef f32 (*DotProductFn)(f32 *x, f32 *y, u32 size);

// ------------------------------------------------------------------
//
// GLOBAL VARIABLES
//
// ------------------------------------------------------------------

// use 16 for SSE, 32 for AVX, 64 for AVX512
const int ALIGNMENT = 32;

// method to test
u32 method = 1;
// number of times the method is called
u32 zillions = 1000;
// size of each vector
u32 vector_size = 64;
// the two vectors used for the dot product
f32 *vector_x, *vector_y;



// ------------------------------------------------------------------
//
// IMPLEMENTATION OF METHODS
//
// ------------------------------------------------------------------


/**
 * Method that computes the dot product of two vectors
 * @param x pointer to first vector
 * @param y pointer to second vector
 * @param size size of the vectors
 * @return the dot product of x and y, i.e. x_0 * y_0 + ... + x_(n-1) * y_(n-1)
 */
f32 dp_ref(f32 *x, f32 *y, u32 size) {
	f32 sum = 0.0;
	
	for (u32 i = 0; i < size; ++i) {
		sum += x[i] * y[i];
	}
	
	return sum;
}

// ------------------------------------------------------------------
//
// METHODS
//
// ------------------------------------------------------------------

// register assembly methods here
extern "C" {

f32 dp_fpu(f32 *x, f32 *y, u32 size) ;

}

typedef struct {
	DotProductFn function;
	string name;
} Method ;

#define entry(name) { name, #name }

// define methods here

Method methods [ ] = {
	{ nullptr, "undefined" },
	entry(dp_ref),
	entry(dp_fpu),
	{ nullptr, "undefined" }
};	


/**
 * main subprogram
 */
int main(int argc, char *argv[]) {

	// get command line arguments
	// use getopt
	

	//
	// create vectors
	//
	
	vector_x = (f32 *) _mm_malloc( vector_size * sizeof(f32), ALIGNMENT);
	vector_y = (f32 *) _mm_malloc( vector_size * sizeof(f32), ALIGNMENT);
	
	//
	// initialize vectors
	//
	const u32 modulo = 17;	

	for (u32 i = 0; i < vector_size; ++i) {
		vector_x[i] = ((i % 3) == 0) ? -1.0 : 1.0;
		vector_y[i] = (f32) 1 + (i % modulo);
	}

	cout << "method=" << method << endl;
	cout << "method_name=" << methods[ method ].name << endl;
	
	//
	// Measure of the number of cycles of the treatment
	// 
	CPUTimer timer;
	
	timer.start();
	
	float total = 0;
	float result = 0;
	u32 z;
	
	result = methods[ method ].function( vector_x, vector_y, vector_size );
	total += result;
	cout << "result=" << std::fixed << result << endl;
		
	for (z = 2; z <= zillions; ++z) {
		result = methods[ method ].function( vector_x, vector_y, vector_size );
		total += result;
	}
	
	timer.stop();
	
	
	cout << "cycles=" << timer << endl;
	cout << "total=" << total << endl;
	
	return EXIT_SUCCESS;
}


