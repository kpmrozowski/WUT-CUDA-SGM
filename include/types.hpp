#pragma once

#ifndef SGM_TYPES_HPP
#define SGM_TYPES_HPP

#include <cstdint>
#include <driver_types.h>

namespace sgm {

using feature_type = uint64_t;
using cost_type = uint8_t;
using cost_sum_type = uint16_t;
using output_type = uint16_t;

}

#endif