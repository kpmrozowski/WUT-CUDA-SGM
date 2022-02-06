#ifndef SGM_WINNER_TAKES_ALL_HPP
#define SGM_WINNER_TAKES_ALL_HPP

#include <Device_buffer.hpp>
#include <Parameters.hpp>
#include <types.hpp>

namespace sgm {

class WinnerTakesAll {

private:
	DeviceBuffer<output_type> m_disparities;
	
public:
	WinnerTakesAll();

	const output_type *get_output() const {
		return m_disparities.data();
	}

	void compute(
		const cost_sum_type *cost_in,
		int width,
		int height,
	    int min_disparity,
	    int max_disparity,
		cudaStream_t stream);

};

void choose_disparities(
	output_type *dest_left,
	const cost_sum_type *cost_in,
	int width,
	int height,
	int min_disparity,
	int max_disparity,
	cudaStream_t stream);

void find2largest(
	output_type *dest,
	const cost_sum_type *cost_in,
	int width,
	int height,
	int min_disparity,
	int max_disparity,
	cudaStream_t stream);

}

#endif // SGM_WINNER_TAKES_ALL_HPP