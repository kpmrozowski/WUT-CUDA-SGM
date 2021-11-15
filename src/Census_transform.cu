#include <cstdio>
#include "Census_transform.hpp"

namespace sgm {

namespace {

static constexpr int CESUS_WINDOW_WIDTH  = 5;
static constexpr int CESUS_WINDOW_HEIGHT = 5;

/**
 * @see https://www.spiedigitallibrary.org/journals/optical-engineering/volume-55/issue-06/063107/Improved-census-transform-for-noise-robust-stereo-matching/10.1117/1.OE.55.6.063107.full?SSO=1
 */
template <typename T>
__global__ void census_transform_kernel(
	feature_type *dest,
	const T *src,
	int width,
	int height)
{
	using pixel_type = T;
	const int padX = CESUS_WINDOW_WIDTH / 2;
	const int padY = CESUS_WINDOW_HEIGHT / 2;
	const int x = threadIdx.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	feature_type f = 1;
	if (padX <= x && x < width - padX && padY <= y && y < height - padY) {
		const T center = src[x + y * width];
        int bitPos0 = 0;
		int bitPos1 = CESUS_WINDOW_WIDTH * CESUS_WINDOW_HEIGHT - 1;
		for (int win_y = 0; win_y < CESUS_WINDOW_HEIGHT; ++win_y, ++bitPos0, --bitPos1) {
			for (int win_x = 0; win_x < CESUS_WINDOW_WIDTH; ++win_x) {
				const T srcVal0 = src[ x + win_x + (y + win_y) * width ];
				const T srcVal1 = src[ x - win_x + (y - win_y) * width ];
				f |= (center < srcVal0) << bitPos0;
				f |= (center < srcVal1) << bitPos1;
			}
		}
		dest[x + y * width] = f;
	}
	// else if ( (0 <= x && x < padX || width - padX <= x && x < width) 
	// 	   || (0 <= y && y < padY || height -padY <= y && y < height) ) {
	// 		dest[x + y * width] = 0;
	// }
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

template <typename T>
void enqueue_census_transform(
	feature_type *dest,
	const T *src,
	int width,
	int height,
	cudaStream_t stream)
{
	printf("My cesus transform\n");
    int num_threads = 1024;
	int block_x_dim = nextPowerOf2(width);
	int block_y_dim = num_threads / block_x_dim;
	int grid_dim = height / block_y_dim;
	if (grid_dim * block_y_dim < height) {
		grid_dim += 1;
	}
	const dim3 bdim(block_x_dim, block_y_dim);
	const dim3 gdim(1, grid_dim, 1);

	
	census_transform_kernel<<<gdim, bdim, 0, stream>>>(dest, src, width, height);
#ifdef DEBUG
	feature_type lookup[height * width];
	printf("dest.size()=%d\n", sizeof(lookup)/sizeof(lookup[0]));
	cudaMemcpy(lookup, dest, sizeof(feature_type) * height * width, cudaMemcpyDeviceToHost);
	printf("dest[100][100]=%d\n", lookup[300 + 300 * width]);
#endif
}

}


template <typename T>
CensusTransform<T>::CensusTransform()
	: m_feature_buffer()
{ }

template <typename T>
void CensusTransform<T>::enqueue(
	const input_type *src,
	int width,
	int height,
	cudaStream_t stream)
{
	if(m_feature_buffer.size() != static_cast<size_t>(width * height)){
		m_feature_buffer = DeviceBuffer<feature_type>(width * height);
	}
	enqueue_census_transform(
		m_feature_buffer.data(), src, width, height, stream);
}

template class CensusTransform<uint8_t>;
template class CensusTransform<uint16_t>;

}
