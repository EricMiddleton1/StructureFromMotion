#include "ORBDetector.hpp"


ORBDetector::ORBDetector(std::vector<Param>&& params)
  : IConfigurable({}, std::move(params))
  , detector{cv::ORB::create()} {
}

std::vector<KeyPoint> ORBDetector::detectKeyPoints(const cv::Mat& frame) {

