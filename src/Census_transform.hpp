#ifndef SGM_CENSUS_TRANSFORM_HPP
#define SGM_CENSUS_TRANSFORM_HPP

#include <Device_buffer.hpp>
#include <types.hpp>

namespace sgm {

template <typename T>
class CensusTransform {

private:
	DeviceBuffer<feature_type> m_feature_buffer;

public:
	CensusTransform();

	const feature_type *get_output() const {
		return m_feature_buffer.data();
	}
	
	void compute(
		const T *src,
		int width,
		int height);

};

}

#endif  // SGM_CENSUS_TRANSFORM_HPP
