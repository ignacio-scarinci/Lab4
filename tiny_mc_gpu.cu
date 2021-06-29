#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "params.h"
#include "wtime.h"
#include <assert.h>

#include <curand_kernel.h>
#include <curand.h>

#include "helper_cuda.h"

#include <cuda_runtime.h>



#define BLOCK_SIZE 128

__device__ void LaunchPhoton(FotonStruct*);
__global__ void LaunchPhoton_Global(void);

__device__ __constant__ unsigned long int num_fotones_cd;
__device__ __constant__ float albedo_cd;
__device__ __constant__ float shells_per_mfp_cd;
__device__ __constant__ unsigned long shells_cd;

// const unsigned int DEFAULT_SEED = 777;

__global__ void init_rng(curandStatePhilox4_32_10_t *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    // Each thread gets same seed, a different sequence number, no offset
    curand_init(1234, id, 0, &state[id]);
}

__global__ void LaunchPhoton_Global(FotonStruct* pp)
{
	int id=threadIdx.x + blockIdx.x * blockDim.x;
  FotonStruct p;
	LaunchPhoton(&p);
  pp[id]=p;
}

__device__ void LaunchPhoton(FotonStruct* p)
{
	p->x  = 0.0f;
	p->y  = 0.0f;
	p->z  = 0.0f;
	p->u = 0.0f;
	p->v = 0.0f;
	p->w = 1.0f;

	p->cascaron = 0;
	p->weight = 1.0f;

}

__global__ void FOTON(float* heat, float* heat2, curandStatePhilox4_32_10_t *state, unsigned long long *fotones_simulados, FotonStruct* pp)
{

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curandStatePhilox4_32_10_t localState = state[id];

	float t;	//step length
  FotonStruct p;
  p  = pp[id];
  int ii = 0;
	while(ii == 0) //this is the main while loop
	{
 		p.t = -__logf(curand_uniform(&localState));
		p.x += p.u*t;
		p.y += p.u*t;
		p.z += p.w*t;
    p.cascaron = __float2uint_rz(sqrtf(p.x * p.x + p.y * p.y + p.z * p.z) * shells_per_mfp_cd);

    if (p.cascaron > shells_cd - 1){
			p.cascaron = shells_cd - 1;
		}
			// Drop weight (apparently only when the photon is scattered)
		p.weight *= albedo_cd ;

    atomicAdd(&heat[p.cascaron], (1.0f - albedo_cd) * p.weight);
    atomicAdd(&heat2[p.cascaron], (1.0f - albedo_cd) * (1.0f - albedo_cd) * p.weight * p.weight);

    if (p.weight < 0.001f){
      if(curand_uniform(&localState) > 0.1f)
      {
        if(atomicAdd(fotones_simulados,1ul) < (num_fotones_cd-NUM_THREADS))
			  {	// Ok to launch another photon
              LaunchPhoton(&p);//Launch a new photon
              continue;
			  }
        ii = 1;
      }
      p.weight *= 10.0f ;
    }

		float xi1, xi2;
		do {
		    xi1 = 2.0f * curand_uniform(&localState) -1.0f;
        xi2 = 2.0f * curand_uniform(&localState) -1.0f;
				t = xi1 * xi1 + xi2 * xi2;
		}while(1.0f < t);
		p.u = 2.0f * t - 1.0f;
		p.v = xi1 * sqrtf(__fdividef(1.0f - p.u * p.u, t));
		p.w = xi2 * sqrtf(__fdividef(1.0f - p.v * p.v, t));
    state[id] = localState;

	}//end main for loop!
}//end MCd



////////////////////
///   MAIN      ///
//////////////////

int main()
{

  printf("# Scattering = %8.3f/cm\n", MU_S);
  printf("# Absorption = %8.3f/cm\n", MU_A);
  printf("# Photons    = %8d\n#\n", PHOTONS);

  double start = wtime();

  // Paso al esapacio de memoria constante las constantes del problema
  const float albedo = MU_S / (MU_S + MU_A);
  const float shells_per_mfp = 1e4 / MICRONS_PER_SHELL / (MU_A + MU_S);
  const int cascaras = SHELLS;
  const unsigned long int n_fotones = PHOTONS;
  checkCudaCall( cudaMemcpyToSymbol(num_fotones_cd, &n_fotones, sizeof(unsigned long int)));
  checkCudaCall( cudaMemcpyToSymbol(albedo_cd, &albedo, sizeof(float)));
  checkCudaCall( cudaMemcpyToSymbol(shells_per_mfp_cd, &shells_per_mfp, sizeof(float)));
  checkCudaCall( cudaMemcpyToSymbol(shells_cd, &cascaras, sizeof(int)));

  unsigned int hilos_tot = NUM_THREADS_PER_BLOCK * NUM_BLOCKS;
  unsigned long long* fotones_simulados;
  fotones_simulados = 0;

  float* heat;
  float* heat2;
  FotonStruct* p_todos;
  cudaError_t cudastat;

  checkCudaCall(cudaMallocManaged(&fotones_simulados, sizeof(unsigned long long)));
  checkCudaCall(cudaMallocManaged(&heat, cascaras*sizeof(float)));
  checkCudaCall(cudaMallocManaged(&heat2, cascaras*sizeof(float)));
  checkCudaCall(cudaMallocManaged(&p_todos, hilos_tot*sizeof(FotonStruct)));

  curandStatePhilox4_32_10_t *devPHILOXStates;
  checkCudaCall(cudaMalloc((void **)&devPHILOXStates,
                           hilos_tot * sizeof(curandStatePhilox4_32_10_t)));

   dim3 dimBlock(NUM_THREADS_PER_BLOCK);
   dim3 dimGrid(NUM_BLOCKS);

   //inicializo el generado de números aleatorios
   init_rng<<<dimGrid, dimBlock>>>(devPHILOXStates);

   //Pongo un foton en cada hilo
   LaunchPhoton_Global<<<dimGrid,dimBlock>>>(p_todos);
   checkCudaCall( cudaDeviceSynchronize() ); // Espero a q terminen todos los hilos
   cudastat=cudaGetLastError(); // Chequeo si hubo algun error
   if(cudastat)printf("Código de error=%i, %s.\n",cudastat,cudaGetErrorString(cudastat));

   //run the kernel
   FOTON<<<dimGrid,dimBlock>>>(heat, heat2, devPHILOXStates, fotones_simulados, p_todos);
   checkCudaCall( cudaDeviceSynchronize() ); // Wait for all threads to finish
   cudastat=cudaGetLastError(); // Check if there was an error
   if(cudastat)printf("Código de error=%i, %s.\n",cudastat,cudaGetErrorString(cudastat));
   // stop timer
   double end = wtime();
   assert(start <= end);
   double elapsed = end - start;

   printf("# %lf seconds\n", elapsed);
   printf("# %lf K photons per second\n", 1e-3 * PHOTONS / elapsed);

   printf("# Radius\tHeat\n");
   printf("# [microns]\t[W/cm^3]\tError\n");
   float t = 4.0f * M_PI * powf(MICRONS_PER_SHELL, 3.0f) * PHOTONS / 1e12;
   for (unsigned int i = 0; i < SHELLS - 1; ++i) {
       printf("%6.0f\t%12.5f\t%12.5f\n", i * (float)MICRONS_PER_SHELL,
              heat[i] / t / (i * i + i + 1.0 / 3.0),
              sqrt(heat2[i] - heat[i] * heat[i] / PHOTONS) / t / (i * i + i + 1.0f / 3.0f));
   }
   printf("# extra\t%12.5f\n", heat[SHELLS - 1] / PHOTONS);

  return 0;

}
