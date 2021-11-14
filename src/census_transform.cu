#include <cstdio>
#include "census_transform.hpp"

namespace sgm {

namespace {

template <typename T>
__global__ void census_transform_kernel(
	feature_type *dest,
	const T *src,
	int width,
	int height,
	int pitch)
{}

template <typename T>
void enqueue_census_transform(
	feature_type *dest,
	const T *src,
	int width,
	int height,
	int pitch,
	cudaStream_t stream)
{
	printf("My cesus transform\n");
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
	int pitch,
	cudaStream_t stream)
{
	if(m_feature_buffer.size() != static_cast<size_t>(width * height)){
		m_feature_buffer = DeviceBuffer<feature_type>(width * height);
	}
	enqueue_census_transform(
		m_feature_buffer.data(), src, width, height, pitch, stream);
}

template class CensusTransform<uint8_t>;
template class CensusTransform<uint16_t>;

}
