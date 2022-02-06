#include <cstdio>
#include <Parameters.hpp>
#include "Median_filter.hpp"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

namespace sgm {
    
const int MEDIAN_BLOCK_SIZE = MEDIAN_WINDOW_WIDTH * MEDIAN_WINDOW_HEIGHT;
const int padX = MEDIAN_WINDOW_WIDTH / 2;
const int padY = MEDIAN_WINDOW_HEIGHT / 2;

template <typename T>
__global__ void median_filter_kernel(
    T *dest, 
    const T *src, 
    int width, 
    int height
)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	T filterVector[MEDIAN_BLOCK_SIZE];
	if ( !( (padX <= x && x < width - padX) && (padY <= y && y < height - padY) ) )
        return;
	for (int win_y = 0; win_y < MEDIAN_WINDOW_HEIGHT; ++win_y) { 
		for (int win_x = 0; win_x < MEDIAN_WINDOW_WIDTH; ++win_x) {
			filterVector[win_x + win_y * MEDIAN_WINDOW_WIDTH] = 
                src[x + win_x - padX + (y + win_y - padY) * width];
		}
	}
	for (int i = 0; i < MEDIAN_BLOCK_SIZE; ++i) {
		for (int j = i + 1; j < MEDIAN_BLOCK_SIZE; ++j) {
			if (filterVector[i] > filterVector[j]) {
				char tmp = filterVector[i];
				filterVector[i] = filterVector[j];
				filterVector[j] = tmp;
			}
		}
	}
	dest[x + y * width] = filterVector[MEDIAN_BLOCK_SIZE / 2];
}

template <typename T>
void compute_median_filter(
	T *dest,
	const T *src,
	int width,
	int height)
{
	printf("My median filter\n");
    int num_threads = 1024;
	int block_x_dim = nextPowerOf2(width);
	int block_y_dim = num_threads / block_x_dim;
	int grid_dim = height / block_y_dim;
	if (grid_dim * block_y_dim < height) {
		grid_dim += 1;
	}
	const dim3 bdim(block_x_dim, block_y_dim, 1);
	const dim3 gdim(1, grid_dim, 1);
    median_filter_kernel<<<gdim, bdim, 0>>>(dest, src, width, height);
#ifdef DEBUG
    T lookup[height * width];
    T lookup_in[height * width];
    printf("dest.size()=%zd\n", sizeof(lookup)/sizeof(lookup[0]));
    cudaMemcpy(lookup, dest, sizeof(T) * height * width, cudaMemcpyDeviceToHost);
    cudaMemcpy(lookup_in, src, sizeof(T) * height * width, cudaMemcpyDeviceToHost);
    for (int y = 0; y < height; ++y) {
        printf("y:%3.3d", y);
        for (int x = 0; x < width; ++x) {
            T val = lookup[x + y * width];
            T val2 = lookup_in[x + y * width];
            // printf("cost[%d][300][%d]=%d\t", x, d, val);
            // printf("%4.1d", val);
            printf("%3.1d", val - val2);
        }
        printf("\n");
    }
#endif
}

void MedianFilter::compute(
	output_type *dest_left,
	const output_type *src,
	int width,
	int height)
{
	compute_median_filter<output_type>(
		dest_left, src, width, height);
}

} // namespace sgm
