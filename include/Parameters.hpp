#pragma once
namespace sgm {

static constexpr int CESUS_WINDOW_WIDTH  = 5; // 9
static constexpr int CESUS_WINDOW_HEIGHT = 5; // 7
static constexpr int MEDIAN_WINDOW_WIDTH  = 3;
static constexpr int MEDIAN_WINDOW_HEIGHT = 5;

struct Parameters
{
	int P1;
	int P2;
	bool subpixel;
    int num_paths;
	int min_disp;
	int LR_max_diff;
    
	/**
	* @param P1 Penalty on the disparity change by plus or minus 1 between nieghbor pixels.
	* @param P2 Penalty on the disparity change by more than 1 between neighbor pixels.
	* @param subpixel Disparity value has 4 fractional bits if subpixel option is enabled.
	* @param num_paths Number of scanlines used in cost aggregation.
	* @param min_disp Minimum possible disparity value.
	* @param LR_max_diff Acceptable difference pixels which is used in LR check consistency. LR check consistency will be disabled if this value is set to negative.
	*/
	Parameters(int P1 = 10, int P2 = 120, bool subpixel = false, int num_paths = 8, int min_disp = 0, int LR_max_diff = 1)
		: P1(P1)
		, P2(P2)
		, subpixel(subpixel)
		, num_paths(num_paths)
		, min_disp(min_disp)
		, LR_max_diff(LR_max_diff)
	{ }
};

int nextPowerOf2(int n);

}
