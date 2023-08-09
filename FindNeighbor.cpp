/*-------------------------------------------------------------------------
Find a valid neighbor of i-th point in current_frame from keyframe.
Valid: within a radius 'grid_r' and an angle 'yaw_max'.
Return -1 when there are no vaild neighbors in keyframe.
-------------------------------------------------------------------------*/
#define _USE_MATH_DEFINES
#include <iostream>
#include "Functions.h"

using namespace std;

int FindNeighbor(Scan current_frame, Scan keyframe, int i, double grid_r, double yaw_max) {

	const Eigen::Vector3d mu_ti(current_frame.means[i]);
	const Eigen::Vector3d n_ti(current_frame.normals[i]);

	double dist = DBL_MAX, angle = DBL_MAX;

	int j = -1;
	for (int p = 0; p < keyframe.means.size(); p++) {

		const Eigen::Vector3d mu_kl(keyframe.means[p]);

		const Eigen::Vector3d n_kl(keyframe.normals[p]);

		double dist_new = hypot(mu_ti(0) - mu_kl(0), mu_ti(1) - mu_kl(1));
		double angle_new = acos(n_ti(0) * n_kl(0) + n_ti(1) * n_kl(1)); // Range: [0, PI]
		if (angle_new > M_PI / 2)
			angle_new = M_PI - angle_new; // Range: [0, PI/2]

		if (dist_new <= grid_r && angle_new <= yaw_max && dist_new < dist) {
			dist = dist_new;
			j = p;
		}
	}

	return j;
}