// Downsample: select the central point of each grid
#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include "Functions.h"

using namespace std;
using namespace cv;

std::vector<RadarPoint> Downsample(std::vector<RadarPoint> kstrong_pointset, double grid_r, double grid_f, double Dmax) {

	grid_r = grid_r / grid_f;
	int N = static_cast<int>(floor((Dmax - grid_r / 2.0) / grid_r));
	double begin = -(grid_r / 2.0 + static_cast<double>(N) * grid_r);

	// Downsampling for each grid
	std::vector<RadarPoint> downsample_pointset;
	for (int i = 0; i < 2 * N + 1; i++) {
		double begin_y = begin + static_cast<double>(i) * grid_r;

		for (int j = 0; j < 2 * N + 1; j++) {
			double begin_x = begin + static_cast<double>(j) * grid_r;

			// Find points in each grid
			std::vector<RadarPoint> grid_pointset;
			for (int k = 0; k < kstrong_pointset.size(); k++) {
				RadarPoint point = kstrong_pointset[k];
				if (point.coordinate(0) >= begin_x && point.coordinate(0) <= begin_x + grid_r && point.coordinate(1) >= begin_y && point.coordinate(1) <= begin_y + grid_r) {
					grid_pointset.push_back(point);
				}
			}

			// Find the centermost point in each grid
			RadarPoint center_point;
			double center_x = begin_x + grid_r / 2.0;
			double center_y = begin_y + grid_r / 2.0;
			double dist = DBL_MAX;
			if (grid_pointset.size() > 0) {
				for (int k = 0; k < grid_pointset.size(); k++) {
					RadarPoint point = grid_pointset[k];
					double dist_new = hypot(point.coordinate(0) - center_x, point.coordinate(1) - center_y);

					if (dist_new < dist) {
						dist = dist_new;
						center_point = point;
					}
				}
				downsample_pointset.push_back(center_point);
			}
		}
	}

	return downsample_pointset;
}