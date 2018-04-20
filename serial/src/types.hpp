#pragma once

#include <opencv2/core/core.hpp>

namespace SFM {
  class Frame;

  using KeyPoints = std::vector<cv::KeyPoint>;
  using Points = std::vector<cv::Point2f>;

  struct Features {
    KeyPoints keyPoints;
    cv::Mat descriptors;
  };

  struct Pose {
    cv::Mat r, t;
  };
}
