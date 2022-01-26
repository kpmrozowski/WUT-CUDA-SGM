#include <cstdio>
#include <Parameters.hpp>
#include "Matching_cost.hpp"
#include <exception>
#include <iostream>
 #include "device_launch_parameters.h"

namespace sgm {

static constexpr int CESUS_DATA_LEN = CESUS_WINDOW_WIDTH * CESUS_WINDOW_HEIGHT - 1;

__global__ void matching_cost_kernel(
	cost_type *dest,
	const feature_type *ctL,
	const feature_type *ctR,
	int width,
	int height)
{
	const int x = threadIdx.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	const int d = blockIdx.z;
    
    if (x < width && y < height && 0 <= x - d) {
        feature_type pxL = ctL[x + y * width];
        feature_type pxR = ctR[x + y * width - d];
        feature_type diff = pxL ^ pxR;
        cost_type distance = 0;   // hamming distance
        for(int i = 0; i < CESUS_DATA_LEN; ++i) {
            distance += (diff >> i) & 1;
        }
        dest[x + y * width + d * width * height] = distance;
        return;
    }
    else if (x < width && y < height) {
        dest[x + y * width + d * width * height] = CESUS_DATA_LEN;
        return;
    }
}

int nextPowerOf2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

template <size_t MAX_DISPARITY>
void enqueue_matching_cost(
	cost_type *dest,
	const feature_type *ctL,
	const feature_type *ctR,
	int width,
	int height)
{
	printf("My matching cost\n");
    int num_threads = 1024;
	int block_x_dim = nextPowerOf2(width);
	int block_y_dim = num_threads / block_x_dim;
	int grid_dim = height / block_y_dim;
	if (grid_dim * block_y_dim < height) {
		grid_dim += 1;
	}
	const dim3 bdim(block_x_dim, block_y_dim);
	const dim3 gdim(1, grid_dim, MAX_DISPARITY);

	matching_cost_kernel<<<gdim, bdim, 0>>>(dest, ctL, ctR, width, height);
	
#ifdef DEBUG
    cost_type lookup[height * width * 32];
    printf("dest.size()=%zd\n", sizeof(lookup)/sizeof(lookup[0]));
    cudaMemcpy(lookup, dest, sizeof(cost_type) * height * width * 32, cudaMemcpyDeviceToHost);
    cost_type max = 0;
    for (int x = 10; x < width - 10; ++x) {
        for (int y = 10; y < height - 10; ++y) {
            for (int d = 0; d < 32; ++d) {
                cost_type val = lookup[x + y * width + d * width * height];
                if (max < val && val < 100) max = val;
            }
        }
    }
    printf("max=%d\n", max);
    for (int x = 10; x < width - 10; ++x) {
        printf("x:%3.2d", x);
        for (int d = 0; d < 32; ++d) {
            cost_type val = lookup[x + 300 * width + d * width * height];
            // printf("cost[%d][300][%d]=%d\t", x, d, val);
            printf("%3.1d", val);
        }
        printf("\n");
    }
#endif
}


template <size_t MAX_DISPARITY>
MatchingCost<MAX_DISPARITY>::MatchingCost()
	: m_cost_cube()
{ }

template <size_t MAX_DISPARITY>
void MatchingCost<MAX_DISPARITY>::enqueue(
	const feature_type *ctL,
	const feature_type *ctR,
	int width,
	int height)
{
	if(m_cost_cube.size() != static_cast<size_t>(width * height * MAX_DISPARITY)){
		m_cost_cube = DeviceBuffer<cost_type>(width * height * MAX_DISPARITY);
	}
	enqueue_matching_cost<MAX_DISPARITY>(
		m_cost_cube.data(), ctL, ctR, width, height);
}

template class MatchingCost< 64>;
template class MatchingCost<128>;
template class MatchingCost<256>;

}