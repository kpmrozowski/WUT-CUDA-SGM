#ifndef SGM_STEREO_SGM_HPP
#define SGM_STEREO_SGM_HPP

#pragma once
#include "Device_buffer.hpp"
#include <Parameters.hpp>
#include "IEngine_SGM.hpp"


namespace sgm {

class StereoSGM {
public:
	static const int SUBPIXEL_SHIFT = 4;
	static const int SUBPIXEL_SCALE = (1 << SUBPIXEL_SHIFT);

	std::unique_ptr<IEngine_SGM> engine;
	
	DeviceBuffer<uint8_t> d_src_left;
	DeviceBuffer<uint8_t> d_src_right;
	DeviceBuffer<uint16_t> d_left_disp;
	DeviceBuffer<uint16_t> d_right_disp;
	DeviceBuffer<uint16_t> d_tmp_left_disp;
	DeviceBuffer<uint16_t> d_tmp_right_disp;

	/**
	* @param width Image's width.
	* @param height Image's height.
	* @param disparity_size Must be 64, 128 or 256.
	* @param depth_bits Image's bits per pixel. Must be 8 or 16.
	* @attention depth_bits must be set to 16 when subpixel is enabled.
	*/
	StereoSGM(int width, int height, int disparity_size, int depth_bits, const Parameters& param = Parameters());
	
	~StereoSGM();

	/**
	* Execute stereo semi global matching.
	* @param dst          Output pointer. User must allocate enough memory.
	* @param left_pixels  A pointer stored input left image.
	* @param right_pixels A pointer stored input right image.
	*/
	void execute(void* dst, const void* left_pixels, const void* right_pixels);

private:
	StereoSGM(const StereoSGM&);
	StereoSGM& operator=(const StereoSGM&);
    
	void cuda_resource_allocate_all();
	void cuda_resource_free_all();
	int width_;
	int height_;
	int disparity_size_;
	int depth_bits_;
	Parameters param_;


};

}

#endif // SGM_STEREO_SGM_HPP
