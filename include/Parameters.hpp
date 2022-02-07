#pragma once
namespace sgm {

static constexpr int CESUS_WINDOW_WIDTH  = 5; // 9
static constexpr int CESUS_WINDOW_HEIGHT = 5; // 7
static constexpr int MEDIAN_WINDOW_WIDTH  = 3;
static constexpr int MEDIAN_WINDOW_HEIGHT = 3;

struct Parameters
{
	int P1;
	int P2;
    int num_paths;
	int min_disp;
    
	/**
	* @param P1 Penalty on the disparity change by plus or minus 1 between nieghbor pixels.
	* @param P2 Penalty on the disparity change by more than 1 between neighbor pixels.
	* @param num_paths Number of scanlines used in cost aggregation.
	* @param min_disp Minimum possible disparity value.
	*/
	Parameters(int P1 = 10, int P2 = 120, int num_paths = 8, int min_disp = 0)
		: P1(P1)
		, P2(P2)
		, num_paths(num_paths)
		, min_disp(min_disp)
	{ }
};

int nextPowerOf2(int n);

}
