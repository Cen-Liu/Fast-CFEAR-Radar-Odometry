#define _USE_MATH_DEFINES
#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include "Functions.h"

using namespace std;
using namespace cv;


std::vector<RadarPoint> MotionCompensation(cv::Mat img_polar_oxford, std::vector<RadarPoint> kstrong_pointset, std::vector<double> velocity) {
	
	std::vector<RadarPoint> kstrong_mc_pointset;
	
	double time_bgn = FormatInfo(img_polar_oxford.row(0).colRange(0, 8));
	double time_end = FormatInfo(img_polar_oxford.row(399).colRange(0, 8));
	double time_mid = (time_bgn + time_end) / 2.0;

	for (int i = 0; i < kstrong_pointset.size(); i++) {
		RadarPoint point = kstrong_pointset[i];
		double time_diff = abs(point.timestamp - time_mid) / 1000000;

		// Translation vector
		Eigen::Vector3d t(velocity[0] * time_diff, velocity[1] * time_diff, 0.0);

		// Rotation matrix
		double theta = velocity[2] * time_diff;
		Eigen::Matrix3d R = (Eigen::Matrix3d() << cos(theta), -sin(theta), 0.0,
			sin(theta), cos(theta), 0.0,
			0.0, 0.0, 1.0).finished();

		// Coordinate
		Eigen::Vector3d coordinate(0.0, 0.0, 0.0);
		if (point.timestamp <= time_mid)
			coordinate = R * point.coordinate + t; // Before middle time
		else
			coordinate = R.transpose() * (point.coordinate - t); // After middle time

		// Azimuth
		double azimuth = atan(abs(coordinate(1) / coordinate(0)));
		if (coordinate(0) < 0 && coordinate(1) > 0) {
			azimuth = M_PI - azimuth;
		}
		else if (coordinate(0) < 0 && coordinate(1) < 0) {
			azimuth = M_PI + azimuth;
		}
		else if (coordinate(0) > 0 && coordinate(1) < 0) {
			azimuth = 2 * M_PI - azimuth;
		}

		RadarPoint point_mc;
		point_mc.timestamp = point.timestamp;
		point_mc.coordinate = coordinate;
		point_mc.azimuth = azimuth;
		point_mc.range = coordinate.norm();

		kstrong_mc_pointset.push_back(point_mc);
	}

	return kstrong_mc_pointset;
}