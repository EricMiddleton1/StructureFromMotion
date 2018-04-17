#pragma once

#include "IConfigurable.hpp"

#include <vector>
#include <utility>

#include <opencv2/features2d.hpp>

class ORBDetector : IConfigurable {
public:
	using KeyPoints = std::vector<cv::KeyPoint>;

	struct Features {
		KeyPoints keyPoints;
		cv::Mat descriptors;
	};

  ORBDetector(std::vector<Param>&& params);

	Features detectKeyPoints(const cv::Mat& frame);

	std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> matchFeatures(
		const Features& features1, const Features& features2);

	static void draw(cv::Mat& frame, const KeyPoints& keyPoints);
	static void draw(cv::Mat& frame, const std::vector<cv::Point2f>& keyPoints);

private:
  cv::Ptr<cv::Feature2D> m_extractor;
	cv::Ptr<cv::DescriptorMatcher> m_matcher;
};
