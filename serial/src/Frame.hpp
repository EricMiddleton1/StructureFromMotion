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

    cv::Mat T() const;
    void T(const cv::Mat& t);

    cv::Mat P() const;
    void P(const cv::Mat& p);

  private:
    using Keypoint_idx = size_t;

    Features extractFeatures(const cv::Mat& image) const;

    ORBDetector& m_detector;

    cv::Mat m_image;
    double m_focal;
    cv::Vec2d m_pp;
    int m_minMatches;

    Features m_features;

    std::map<const Frame*, Points> m_keypointMap;
    std::map<const Frame*, Pose> m_poseMap;
    std::map<Keypoint_idx, PointID> m_landmarkMap;

    //4x4 pose transformation matrix
    cv::Mat m_T;
    //3x4 projection matrix
    cv::Mat m_P;
  };
}
