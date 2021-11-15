#ifndef SGM_CENSUS_TRANSFORM_HPP
#define SGM_CENSUS_TRANSFORM_HPP

#include <Device_buffer.hpp>
#include <types.hpp>

namespace sgm {

template <typename T>
class CensusTransform {

public:
	using input_type = T;

private:
	DeviceBuffer<feature_type> m_feature_buffer;

public:
	CensusTransform();

	const feature_type *get_output() const {
		return m_feature_buffer.data();
	}
	
	void enqueue(
		const input_type *src,
		int width,
		int height,
		cudaStream_t stream);

};

}

#endif  // SGM_CENSUS_TRANSFORM_HPP
