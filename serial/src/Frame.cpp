#include "Frame.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

namespace SFM {
Frame::Frame(ORBDetector& detector, const cv::Mat& image, double focal,
  const cv::Vec2d& pp, int minMatches)
  : m_detector{detector}
  , m_image{image.clone()}
  , m_focal{focal}
  , m_pp{pp}
  , m_minMatches{minMatches}
  , m_features{extractFeatures(image)}
  , m_landmarkMap([](const cv::Point2f& p1, const cv::Point2f& p2) {
      return p1.x < p2.x;
    })
  , m_T{cv::Mat::eye(4, 4, CV_64F)}
  , m_P{cv::Mat::eye(3, 4, CV_64F)} {
}

const cv::Mat& Frame::getImage() const {
  return m_image;
}

const Features& Frame::getFeatures() const {
  return m_features;
}

bool Frame::compare(Frame& other) {
  std::vector<std::vector<cv::DMatch>> matches;
  std::vector<size_t> candidates1, candidates2;
  std::vector<cv::Point2f> points1, points2;
  cv::Mat mask;

  //Two nearest neighbor matches per feature
  m_detector.getMatcher()->knnMatch(m_features.descriptors, other.m_features.descriptors,
    matches, 2);

  //Apply ratio test, store passing features
  for(const auto& match : matches) {
    if(match[0].distance < 0.7*match[1].distance) {
      auto idx_1 = match[0].queryIdx, idx_2 = match[0].trainIdx;

      candidates1.push_back(idx_1);
      candidates2.push_back(idx_2);
      points1.push_back(m_features.keyPoints[idx_1].pt);
      points2.push_back(m_features.keyPoints[idx_2].pt);
    }
  }

  //Reject frame if not enough matches
  if(candidates1.size() < m_minMatches) {
    return false;
  }

  //Calculate essential matrix
  auto E = cv::findEssentialMat(points1, points2, m_focal, m_pp, cv::RANSAC, 0.999,
    1.0, mask);

  //Recover pose (rotation and translation) from essential matrix
  Pose pose;
  cv::recoverPose(E, points1, points2, pose.r, pose.t, m_focal, m_pp, mask);

  //Validate rotation matrix
  if(std::fabs(cv::determinant(pose.r) - 1.0) > 1e-07) {
    return false;
  }

  //Move pose into pose map
  //m_poseMap.emplace(&other, std::move(pose));
  m_poseMap[&other] = pose;

  //Move common keypoints into map
  //m_keypointMap.emplace(&other, std::move(points1));
  //other.m_keypointMap.emplace(this, std::move(points2));
  m_keypointMap[&other] = candidates1;
  other.m_keypointMap[this] = candidates2;

  return true;
}

Features Frame::extractFeatures(const cv::Mat& image) const {
  return m_detector.detectKeyPoints(image);
}

bool Frame::hasPose(const Frame& other) const {
  return m_poseMap.count(&other) > 0;
}

const Pose& Frame::getPose(const Frame& other) const {
  return m_poseMap.at(&other);
}

bool Frame::hasKeypoints(const Frame& other) const {
  return m_keypointMap.count(&other) > 0;
}

Points Frame::getKeypoints(const Frame& other) const {
  const auto& indicies = m_keypointMap.at(&other);

  Points p(indicies.size());
  for(size_t i = 0; i < indicies.size(); ++i) {
    p[i] = m_features.keyPoints[indicies[i]].pt;
  }

  return p;
}

bool Frame::hasLandmark(const cv::Point2f& point) const {
  return m_landmarkMap.count(point) > 0;
}

LandmarkID Frame::getLandmark(const cv::Point2f& point) const {
  return m_landmarkMap.at(point);
}

void Frame::addLandmark(const cv::Point2f& point, LandmarkID id) {
  m_landmarkMap[point] = id;
}

cv::Mat Frame::T() const {
  return m_T;
}

void Frame::T(const cv::Mat& t) {
  m_T = t;
}

cv::Mat Frame::P() const {
  return m_P;
}

void Frame::P(const cv::Mat& p) {
  m_P = p;
}

}
