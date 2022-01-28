#include "Energy.hpp"

namespace sgm {

template <int DIRECTION, unsigned int MAX_DISPARITY>
__global__ void aggregate_horizontal_path_kernel(
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
void compute_energy_L2R(
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
	aggregate_horizontal_path_kernel<1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		dest, cost_in, width, height, p1, p2, min_disp);
}

template <unsigned int MAX_DISPARITY>
void compute_energy_R2L(
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
	aggregate_horizontal_path_kernel<-1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		dest, cost_in, width, height, p1, p2, min_disp);
}


template void compute_energy_L2R<64u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_L2R<128u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_L2R<256u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_R2L<64u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_R2L<128u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_R2L<256u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

}