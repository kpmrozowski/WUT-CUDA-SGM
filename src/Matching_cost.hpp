#ifndef MATCHING_COST_HPP
#define MATCHING_COST_HPP

#include <Device_buffer.hpp>
#include <types.hpp>

namespace sgm {

template <size_t MAX_DISPARITY>
class MatchingCost {

private:
	DeviceBuffer<cost_type> m_cost_cube;

public:
	MatchingCost();

	const cost_type *get_output() const {
		return m_cost_cube.data();
	}
	
	void compute(
        const feature_type *ctL,
        const feature_type *ctR,
		int width,
		int height);

};

}

#endif // MATCHING_COST_HPP