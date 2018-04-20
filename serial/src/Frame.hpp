#pragma once

#include <vector>
#include <utility>
#include <map>

#include <opencv2/features2d.hpp>

#include "OrbDetector.hpp"
#include "types.hpp"

namespace SFM {
  class Frame {
  public:
    Frame(ORBDetector& detector, const cv::Mat& image, double focal, const cv::Vec2d& pp,
      int minMatches = 10);

    bool compare(Frame& other);

    const cv::Mat& getImage() const;
    const Features& getFeatures() const;

    bool hasPose(const Frame& other) const;
    const Pose& getPose(const Frame& other) const;

    bool hasKeypoints(const Frame& other) const;
    const Points& getKeypoints(const Frame& frame) const;

  private:
    using KeyPoint_idx = size_t;

    Features extractFeatures(const cv::Mat& image) const;

    ORBDetector& m_detector;

    cv::Mat m_image;
    double m_focal;
    cv::Vec2d m_pp;
    int m_minMatches;

    Features m_features;

    //std::map<KeyPoint_idx, std::map<Frame*, KeyPoint_idx>> m_keypointMatches;
    std::map<const Frame*, Points> m_keypointMap;
    std::map<const Frame*, Pose> m_poseMap;
  };
}
