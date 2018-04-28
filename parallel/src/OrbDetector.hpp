#pragma once

#include "IConfigurable.hpp"
#include "types.hpp"

#include <vector>
#include <utility>

#include <opencv2/features2d.hpp>

namespace SFM {
  class ORBDetector : IConfigurable {
  public:
    ORBDetector(std::vector<Param>&& params);

    Features detectKeyPoints(const cv::Mat& frame) const;

    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> matchFeatures(
      Features& features1, Features& features2) const;

    cv::Ptr<cv::Feature2D> getExtractor();
    cv::Ptr<cv::DescriptorMatcher> getMatcher();

    static void draw(cv::Mat& frame, const KeyPoints& keyPoints);
    static void draw(cv::Mat& frame, const std::vector<cv::Point2f>& keyPoints);

  private:
    cv::Ptr<cv::AKAZE> m_extractor;
    cv::Ptr<cv::DescriptorMatcher> m_matcher;
  };
}
