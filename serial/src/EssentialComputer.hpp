#pragma once

#include "IConfigurable.hpp"

#include <vector>
#include <utility>

#include <opencv2/core/core.hpp>

class EssentialComputer : IConfigurable {
public:
	using Points = std::vector<cv::Point2f>;
  EssentialComputer(std::vector<Param>&& params);

	void setFocalLength(double focal);
	void setPP(const cv::Point2d pp);

	bool computePose(const Points& features1, const Points& features2, cv::Mat& r, cv::Mat& t) const;
private:
	static bool validateRotationMatrix(const cv::Mat& r);

	double m_prob, m_threshold;
	double m_focal;
	cv::Point2d m_pp;
};
