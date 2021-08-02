#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "params.h"
#include "wtime.h"
#include <assert.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>
#include <cstdio>
#include "helper_cuda.h"
#include <cuda_runtime.h>

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

__device__ __constant__ unsigned long int num_fotones_cd;
__device__ __constant__ float albedo_cd;
__device__ __constant__ float shells_per_mfp_cd;
__device__ __constant__ int shells_cd;
__device__ __constant__ int hilos_tot_cd;

__global__ void init_rng(curandStatePhilox4_32_10_t *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    // Cada hilo tiene la misma semilla, pero diferente secuenciador
		clock_t clock();
		long long int clock64();

		curand_init(clock64(), id, 0, &state[id]);
}

__device__ void InicializoFoton(FotonStruct* p)
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

__global__ void simulador(float* heat, float* heat2, curandStatePhilox4_32_10_t *state, unsigned long long *fotones_simulados)
{
  __shared__ float heat_b[SHELLS];
 	__shared__ float heat2_b[SHELLS];
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curandStatePhilox4_32_10_t localState = state[id];
	float t;
  FotonStruct p;

   for(int i=threadIdx.x;i<SHELLS; i += blockDim.x)
 			{
	  		heat_b[i] =0.0f;
				heat2_b[i]=0.0f;
			}
	 __syncthreads();


	InicializoFoton(&p);
  int termino = 0;
	while(termino == 0)
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

    atomicAdd(&heat_b[p.cascaron], (1.0f - albedo_cd) * p.weight);
    atomicAdd(&heat2_b[p.cascaron], (1.0f - albedo_cd) * (1.0f - albedo_cd) * p.weight * p.weight);

    if (p.weight < 0.001f){
      if(curand_uniform(&localState) > 0.1f)
      {
        if(atomicAdd(fotones_simulados,1ul) < (num_fotones_cd-hilos_tot_cd))
			  {	// Fquedan fotones por lanzar
              InicializoFoton(&p);//Lanzo foton
              continue;
			  }
        termino = 1;
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

	}
	  __syncthreads();
	  for(int i=threadIdx.x; i<SHELLS;i+=blockDim.x){
			atomicAdd(&heat[i],heat_b[i]);
			atomicAdd(&heat2[i],heat2_b[i]);
		}
		__syncthreads();

}



////////////////////
///   MAIN      ///
//////////////////

int main(int argc, char *argv[])
{
	unsigned long PHOTONS;
	unsigned int NUM_BLOCKS;
	unsigned int THREADS_POR_BLOCK;
	sscanf(argv[1], "%lu", &PHOTONS);
	sscanf(argv[2], "%iu", &NUM_BLOCKS);
	sscanf(argv[3], "%iu", &THREADS_POR_BLOCK);

  printf("# Scattering = %8.3f/cm\n", MU_S);
  printf("# Absorption = %8.3f/cm\n", MU_A);
  printf("# Photons    = %8lu\n#\n", PHOTONS);


  // Paso al esapacio de memoria constante las constantes del problema
  const float albedo = MU_S / (MU_S + MU_A);
  const float shells_per_mfp = 1e4 / MICRONS_PER_SHELL / (MU_A + MU_S);
  const int cascaras = SHELLS;
  const unsigned long int n_fotones = PHOTONS;
	unsigned int hilos_tot = THREADS_POR_BLOCK * NUM_BLOCKS;
  checkCudaCall( cudaMemcpyToSymbol(num_fotones_cd, &n_fotones, sizeof(unsigned long int)));
  checkCudaCall( cudaMemcpyToSymbol(albedo_cd, &albedo, sizeof(float)));
  checkCudaCall( cudaMemcpyToSymbol(shells_per_mfp_cd, &shells_per_mfp, sizeof(float)));
  checkCudaCall( cudaMemcpyToSymbol(shells_cd, &cascaras, sizeof(int)));
	checkCudaCall( cudaMemcpyToSymbol(hilos_tot_cd, &hilos_tot, sizeof(int)));

  unsigned long long* fotones_simulados;
  fotones_simulados = 0;

	float* heat;
  float* heat2;

  checkCudaCall(cudaMallocManaged(&fotones_simulados, sizeof(unsigned long long)));
  checkCudaCall(cudaMallocManaged(&heat, cascaras*sizeof(float)));
  checkCudaCall(cudaMallocManaged(&heat2, cascaras*sizeof(float)));

  checkCudaCall(cudaMemset(heat, 0, cascaras));
  checkCudaCall(cudaMemset(heat2, 0, cascaras));


  curandStatePhilox4_32_10_t *devPHILOXStates;
  checkCudaCall(cudaMalloc((void **)&devPHILOXStates,
                           hilos_tot * sizeof(curandStatePhilox4_32_10_t)));


	 double start = wtime();
   dim3 dimBlock(THREADS_POR_BLOCK);
   dim3 dimGrid(NUM_BLOCKS);
	 //inicializo el generado de números aleatorios
	 init_rng<<<dimGrid, dimBlock>>>(devPHILOXStates);
	 checkCudaCall( cudaDeviceSynchronize() );


	simulador<<<dimGrid,dimBlock>>>(heat, heat2, devPHILOXStates, fotones_simulados); //, p_todos);
	checkCudaCall( cudaDeviceSynchronize() );

   // stop timer
   double end = wtime();
   assert(start <= end);
   double elapsed = end - start;


   printf("# %lf seconds\n", elapsed);
   printf("# %lf K photons per second\n", 1e-3 * PHOTONS / elapsed);
	 printf("# Fotones simulados    = %llu\n#\n", fotones_simulados[0]);
   printf("# Radius\tHeat\n");
   printf("# [microns]\t[W/cm^3]\tError\n");
   float t = 4.0f * M_PI * powf(MICRONS_PER_SHELL, 3.0f) * PHOTONS / 1e12;
   for (unsigned int i = 0; i < SHELLS - 1; ++i) {
       printf("%6.0f\t%12.5f\t%12.5f\n", i * (float)MICRONS_PER_SHELL,
              heat[i] / t / (i * i + i + 1.0 / 3.0),
              sqrt(heat2[i] - heat[i] * heat[i] / PHOTONS) / t / (i * i + i + 1.0f / 3.0f));
   }
   printf("# extra\t%12.5f\n", heat[SHELLS - 1] / PHOTONS);

   checkCudaCall( cudaFree(heat));
   checkCudaCall( cudaFree(heat2));
	 checkCudaCall( cudaFree(fotones_simulados));



  return 0;

}
