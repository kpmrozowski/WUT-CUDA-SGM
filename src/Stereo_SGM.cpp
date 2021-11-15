#include <stdio.h>
#include <memory>

#include "Stereo_SGM.hpp"

#include "internal.h"
#include "Device_buffer.hpp"
#include "Engine_SGM.hpp"

namespace sgm {

	StereoSGM::StereoSGM(int width, int height, int disparity_size, int depth_bits, const Parameters& param) :
		width_(width),
		height_(height),
		disparity_size_(disparity_size),
		depth_bits_(depth_bits),
		param_(param)
	{
		cuda_resource_allocate_all();
		cuda_resource_fillZero();
	}

	StereoSGM::~StereoSGM() {
	}
    
	void StereoSGM::cuda_resource_allocate_all()
	{
		d_src_left.allocate(depth_bits_ / 8 * width_ * height_);
		d_src_right.allocate(depth_bits_ / 8 * width_ * height_);

		d_left_disp.allocate(width_ * height_);
		d_right_disp.allocate(width_ * height_);
        
        d_tmp_left_disp.allocate(width_ * height_);
		d_tmp_right_disp.allocate(width_ * height_);
	}
	void StereoSGM::cuda_resource_fillZero()
	{
        d_left_disp.fillZero();
		d_right_disp.fillZero();
		d_tmp_left_disp.fillZero();
		d_tmp_right_disp.fillZero();
	}

	void StereoSGM::cuda_resource_free_all()
	{
		d_src_left.destroy();
		d_src_right.destroy();
        d_left_disp.destroy();
		d_right_disp.destroy();
		d_tmp_left_disp.destroy();
		d_tmp_right_disp.destroy();
	}
	
	void StereoSGM::execute(void* dst, const void* left_pixels, const void* right_pixels)
    {
		if (depth_bits_ == 8 && disparity_size_ == 64)
			engine = std::make_unique< Engine_SGM_Impl<uint8_t, 64> >();
		else if (depth_bits_ == 8 && disparity_size_ == 128)
			engine = std::make_unique< Engine_SGM_Impl<uint8_t, 128> >();
		else if (depth_bits_ == 8 && disparity_size_ == 256)
			engine = std::make_unique< Engine_SGM_Impl<uint8_t, 256> >();
		else if (depth_bits_ == 16 && disparity_size_ == 64)
			engine = std::make_unique< Engine_SGM_Impl<uint16_t, 64> >();
		else if (depth_bits_ == 16 && disparity_size_ == 128)
			engine = std::make_unique< Engine_SGM_Impl<uint16_t, 128> >();
		else if (depth_bits_ == 16 && disparity_size_ == 256)
			engine = std::make_unique< Engine_SGM_Impl<uint16_t, 256> >();
		else
			throw std::logic_error("depth bits must be 8 or 16, and disparity size must be 64 or 128");
		
		const void *d_input_left, *d_input_right;
		CudaSafeCall(cudaMemcpy(d_src_left.data(), left_pixels, d_src_left.size(), cudaMemcpyHostToDevice));
		CudaSafeCall(cudaMemcpy(d_src_right.data(), right_pixels, d_src_right.size(), cudaMemcpyHostToDevice));
		d_input_left = d_src_left.data();
		d_input_right = d_src_right.data();
		
		void* d_tmp_left_disp_data = d_tmp_left_disp.data();
		void* d_tmp_right_disp_data = d_tmp_right_disp.data();
		// const void *d_src_left_data = d_src_left.data();
		// const void *d_src_right_data = d_src_right.data();

		// engine->execute();
		engine->execute(
			(uint16_t*)d_tmp_left_disp_data, (uint16_t*)d_tmp_right_disp_data,
			d_input_left, d_input_right, 
			width_, height_, param_);
		cuda_resource_free_all();
	}
}

