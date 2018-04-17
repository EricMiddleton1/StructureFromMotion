#pragma once

#include "IConfigurable.hpp"

#include <features2d.hpp>

class OrbDetector : IConfigurable {
public:
  OrbDetector(std::vector<Param>&& params);

  std::vector<cv::KeyPoint> detectKeyPoints(const cv::Mat& frame);

private:
  cv::Ptr<ORB> extractor;
};
