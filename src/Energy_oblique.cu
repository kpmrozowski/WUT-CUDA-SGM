#include "Energy.hpp"

namespace sgm {

template <int X_DIRECTION, int Y_DIRECTION, unsigned int MAX_DISPARITY>
__global__ void aggregate_oblique_path_kernel(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp)
{
	;
}

template <unsigned int MAX_DISPARITY>
void compute_energy_upL2downR(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream)
{
	const int gdim = 1;
	const int bdim = width;
	aggregate_oblique_path_kernel<1, 1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		dest, cost_in, width, height, p1, p2, min_disp);
}

template <unsigned int MAX_DISPARITY>
void compute_energy_upR2downL(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream)
{
	const int gdim = 1;
	const int bdim = width;
	aggregate_oblique_path_kernel<-1, 1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		dest, cost_in, width, height, p1, p2, min_disp);
}

template <unsigned int MAX_DISPARITY>
void compute_energy_downR2upL(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream)
{
	const int gdim = 1;
	const int bdim = width;
	aggregate_oblique_path_kernel<-1, -1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		dest, cost_in, width, height, p1, p2, min_disp);
}

template <unsigned int MAX_DISPARITY>
void compute_energy_downL2upR(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream)
{
	const int gdim = 1;
	const int bdim = width;
	aggregate_oblique_path_kernel<1, -1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		dest, cost_in, width, height, p1, p2, min_disp);
}


template void compute_energy_upL2downR<64u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_upL2downR<128u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_upL2downR<256u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_upR2downL<64u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_upR2downL<128u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_upR2downL<256u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_downR2upL<64u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_downR2upL<128u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_downR2upL<256u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_downL2upR<64u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_downL2upR<128u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_downL2upR<256u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

}