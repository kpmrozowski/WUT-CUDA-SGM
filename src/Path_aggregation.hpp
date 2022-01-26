#ifndef SGM_PATH_AGGREGATION_HPP
#define SGM_PATH_AGGREGATION_HPP

#include <Device_buffer.hpp>
#include <Parameters.hpp>
#include <types.hpp>

namespace sgm {

template <size_t MAX_DISPARITY>
class PathAggregation {

private:
	static const unsigned int MAX_NUM_PATHS = 8;

	DeviceBuffer<cost_type> m_path_cost_buffer;
	
public:
	PathAggregation();
	~PathAggregation();

	const cost_type *get_output() const {
		return m_path_cost_buffer.data();
	}

	void enqueue(
		const feature_type *left,
		const feature_type *right,
		int width,
		int height,
		PathType path_type,
		unsigned int p1,
		unsigned int p2,
		int min_disp,
		cudaStream_t stream);

};

}

#endif // SGM_PATH_AGGREGATION_HPP
