#include "EssentialComputer.hpp"

#include <string>

#include <opencv2/calib3d.hpp>

EssentialComputer::EssentialComputer(std::vector<Param>&& params)
  : IConfigurable({"confidence", "threshold"}, std::move(params))
	,	m_prob{std::stof(getParam("confidence"))}
	,	m_threshold{std::stof(getParam("threshold"))}
	,	m_focal{1.0}
	,	m_pp{0., 0.} {
}

void EssentialComputer::setFocalLength(double focal) {
	m_focal = focal;
}

void EssentialComputer::setPP(const cv::Point2d pp) {
	m_pp = pp;
}

bool EssentialComputer::computePose(const Points& features1, const Points& features2,
	cv::Mat& r, cv::Mat& t) const {
	cv::Mat mask;

	//Compute essential matrix
	auto E = cv::findEssentialMat(features1, features2, m_focal, m_pp, cv::RANSAC, m_prob,
		m_threshold, mask);

	//Recover pose from essential matrix (with cheirality check)
	cv::recoverPose(E, features1, features2, r, t, m_focal, m_pp, mask);

	//Validate rotation matrix
	return validateRotationMatrix(r);
}

bool EssentialComputer::validateRotationMatrix(const cv::Mat& r) {
	return (std::fabs(cv::determinant(r) - 1.0) < 1e-07);
}
