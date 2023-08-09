// Save transformed Cartesian images
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include "Functions.h"

using namespace std;
using namespace cv;

void SaveCartesianImage(Scan scan, double Dmax, double range_res, std::string save_path) {

	int N = static_cast<int>(floor(Dmax / range_res));

	cv::Mat img_scan = cv::Mat::zeros(2 * N, 2 * N, CV_64FC1);

	for (int i = 0; i < scan.means.size(); i++) {
		int col = N + static_cast<int>(floor(scan.means[i][0] / range_res));
		int row = N + static_cast<int>(floor(scan.means[i][1] / range_res));
		img_scan.at<double>(row, col) = 255;
	}
	
	imwrite(save_path + to_string(static_cast<long long>(scan.timespan(0))) + "_Cartesian.png", img_scan);
}