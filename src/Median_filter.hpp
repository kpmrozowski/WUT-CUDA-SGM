#ifndef SGM_MEDIAN_FILTER_HPP
#define SGM_MEDIAN_FILTER_HPP

#include <Device_buffer.hpp>
#include <types.hpp>

namespace sgm {

class MedianFilter {

public:
	void compute(
	    output_type *dest_left,
		const output_type *src,
		int width,
		int height);

};

}

#endif  // SGM_MEDIAN_FILTER_HPP
