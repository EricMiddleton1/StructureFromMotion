#include "OrbDetector.hpp"

#include <opencv2/imgproc.hpp>

namespace SFM {
ORBDetector::ORBDetector(std::vector<Param>&& params)
  : IConfigurable({}, std::move(params))
  , m_extractor{cv::AKAZE::create()}
	,	m_matcher{cv::DescriptorMatcher::create("BruteForce-Hamming")} {
}

Features ORBDetector::detectKeyPoints(const cv::Mat& frame) const {
	Features result;

	m_extractor->detectAndCompute(frame, cv::noArray(), result.keyPoints, result.descriptors);

	return result;
}

std::pair<Points, Points> ORBDetector::matchFeatures(Features& features1,
  Features& features2) const {
	
	std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> result;
	std::vector<cv::DMatch> matches;

	m_matcher->match(features1.descriptors, features2.descriptors, matches);

	for(const auto& match : matches) {
		result.first.push_back(features1.keyPoints[match.queryIdx].pt);
		result.second.push_back(features2.keyPoints[match.trainIdx].pt);
	}

	return result;
}

cv::Ptr<cv::Feature2D> ORBDetector::getExtractor() {
  return m_extractor;
}

cv::Ptr<cv::DescriptorMatcher> ORBDetector::getMatcher() {
  return m_matcher;
}

void ORBDetector::draw(cv::Mat& frame, const KeyPoints& keyPoints) {
	for(const auto& keyPoint : keyPoints) {
		auto pos = keyPoint.pt;
		auto radius = keyPoint.size/2.f;

		cv::rectangle(frame, {pos.x + radius, pos.y + radius, 2*radius, 2*radius},
			cv::Scalar(0, 255, 0));
	}
}

void ORBDetector::draw(cv::Mat& frame, const std::vector<cv::Point2f>& keyPoints) {
	for(const auto& pt : keyPoints) {
		auto radius = 5;

		cv::rectangle(frame, {pt.x + radius, pt.y + radius, 2*radius, 2*radius},
			cv::Scalar(0, 255, 0));
	}
}
}
