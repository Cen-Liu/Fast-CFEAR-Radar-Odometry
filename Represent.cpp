// Calculate full scan representation: means and normals
#define _USE_MATH_DEFINES
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "Functions.h"

using namespace std;
using namespace cv;
using namespace Eigen;

Scan Represent(cv::Mat img_polar_oxford, std::vector<RadarPoint> kstrong_pointset, std::vector<RadarPoint> downsample_pointset, int inlier_num, double condition_num, double grid_r) {

	Scan scan; // Full scan representation
	scan.timespan(0) = FormatInfo(img_polar_oxford.row(0).colRange(0, 8)); // Start time
	scan.timespan(2) = FormatInfo(img_polar_oxford.row(399).colRange(0, 8)); // End time
	scan.timespan(1) = (scan.timespan(0) + scan.timespan(1)) / 2.0; // Middle time

	for (int i = 0; i < downsample_pointset.size(); i++) {

		// Find non-zero points in each circle
		std::vector<RadarPoint> circle_pointset;
		for (int j = 0; j < kstrong_pointset.size(); j++) {
			double dist = (downsample_pointset[i].coordinate - kstrong_pointset[j].coordinate).norm();
			if (dist <= grid_r) {
				circle_pointset.push_back(kstrong_pointset[j]);
			}
		}

		// 1st Stage Filter: Exclude distributions without enough inliers
		if (circle_pointset.size() < inlier_num)
			continue;

		// Sample mean
		Eigen::Vector3d mean = Eigen::Vector3d::Zero();
		for (int k = 0; k < circle_pointset.size(); k++) {
			mean += circle_pointset[k].coordinate;
		}
		mean /= static_cast<double>(circle_pointset.size());

		// Sample covariance matrix
		double var_x = 0, var_y = 0, covar = 0;
		for (int k = 0; k < circle_pointset.size(); k++) {
			var_x += (circle_pointset[k].coordinate(0) - mean(0)) * (circle_pointset[k].coordinate(0) - mean(0));
			var_y += (circle_pointset[k].coordinate(1) - mean(1)) * (circle_pointset[k].coordinate(1) - mean(1));
			covar += (circle_pointset[k].coordinate(0) - mean(0)) * (circle_pointset[k].coordinate(1) - mean(1));
		}
		var_x /= (static_cast<double>(circle_pointset.size()) - 1.0);
		var_y /= (static_cast<double>(circle_pointset.size()) - 1.0);
		covar /= (static_cast<double>(circle_pointset.size()) - 1.0);

		cv::Mat A = cv::Mat::zeros(2, 2, CV_64FC1);
		A.at<double>(0, 0) = var_x;
		A.at<double>(0, 1) = covar;
		A.at<double>(1, 0) = covar;
		A.at<double>(1, 1) = var_y;

		// Eigenvalue decomposition
		cv::Mat eigen_val, eigen_vec;
		cv::eigen(A, eigen_val, eigen_vec);

		// 2nd Stage Filter: Exclude ill-defined distributions
		double kappa = eigen_val.at<double>(0) / eigen_val.at<double>(1);
		if (kappa > condition_num)
			continue;

		// Save valid means and normals
		scan.means.push_back(mean);
		scan.normals.push_back((Eigen::Vector3d() << eigen_vec.at<double>(1, 0), eigen_vec.at<double>(1, 1), 0.0).finished()); // The smallest eigenvector among the two eigenvectors

		// Range
		scan.ranges.push_back(mean.norm());

		// Azimuth
		double azimuth = atan(abs(mean(1) / mean(0)));
		if (mean(0) < 0 && mean(1) > 0) {
			azimuth = M_PI - azimuth;
		}
		else if (mean(0) < 0 && mean(1) < 0) {
			azimuth = M_PI + azimuth;
		}
		else if (mean(0) > 0 && mean(1) < 0) {
			azimuth = 2 * M_PI - azimuth;
		}
		scan.azimuths.push_back(azimuth);

		// Timestamp
		double timestamp;
		double diff_azi = DBL_MAX;
		for (int k = 0; k < kstrong_pointset.size(); k++) {
			double diff_azi_new = abs(azimuth - kstrong_pointset[k].azimuth);
			if (diff_azi_new < diff_azi) {
				diff_azi = diff_azi_new;
				timestamp = kstrong_pointset[k].timestamp;
			}
		}
		scan.timestamps.push_back(timestamp);

		// Initial pose
		scan.poses.push_back(Sophus::SE3d());
	}
	// Initial control pose
	scan.control_pose = Sophus::SE3d();

	return scan;
}