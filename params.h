#pragma once

#include <time.h> // time

#ifndef SHELLS
#define SHELLS 101 // discretization level
#endif

#ifndef PHOTONS
#define PHOTONS 10000000 // 32K photons original
#endif

#ifndef MU_A
#define MU_A 2.0f // Absorption Coefficient in 1/cm !!non-zero!!
#endif

#ifndef MU_S
#define MU_S 20.0f // Reduced Scattering Coefficient in 1/cm
#endif

#ifndef MICRONS_PER_SHELL
#define MICRONS_PER_SHELL 50 // Thickness of spherical shells in microns
#endif

#ifndef SEED
#define SEED (time(NULL)) // random seed
#endif


#define NUM_BLOCKS 56

#define NUM_THREADS_PER_BLOCK 320

#define NUM_THREADS 17920



typedef struct __align__(16)
{
	float x;		// Global x coordinate [cm]
	float y;		// Global y coordinate [cm]
	float z;		// Global z coordinate [cm]
	float u;		// (Global, normalized) x-direction
	float v;		// (Global, normalized) y-direction
	float w;		// (Global, normalized) z-direction
	float t;
	float weight;			// Photon weight
	unsigned int cascaron;				// Current layer
}FotonStruct;
