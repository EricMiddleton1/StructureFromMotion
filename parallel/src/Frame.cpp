#include "Frame.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

namespace SFM {
Frame::Frame(size_t id, ORBDetector& detector, const cv::Mat& image, double focal,
  const cv::Vec2d& pp, int minMatches)
  : m_id{id}
  , m_detector{detector}
  , m_focal{focal}
  , m_pp{pp}
  , m_minMatches{minMatches}
  , m_features{extractFeatures(image)}
  , m_T{cv::Mat::eye(4, 4, CV_64F)}
  , m_P{cv::Mat::eye(3, 4, CV_64F)} {
}

Frame::Frame(size_t id, ORBDetector& detector, Features&& features, double focal,
  const cv::Vec2d& pp, int minMatches)
  : m_id{id}
  , m_detector{detector}
  , m_focal{focal}
  , m_pp{pp}
  , m_minMatches{minMatches}
  , m_features{std::move(features)}
  , m_T{cv::Mat::eye(4, 4, CV_64F)}
  , m_P{cv::Mat::eye(3, 4, CV_64F)} {
}

Frame::Frame(const Frame& other)
	:	m_id{other.m_id}
	,	m_detector{other.m_detector}
	,	m_focal{other.m_focal}
	,	m_pp{other.m_pp}
	,	m_minMatches{other.m_minMatches}
	,	m_features{other.m_features}
	,	m_landmarks{other.m_landmarks}
	,	m_keypointMap{other.m_keypointMap}
	,	m_poseMap{other.m_poseMap}
	,	m_landmarkMap{other.m_landmarkMap}
	,	m_T{other.m_T}
	,	m_P{other.m_P} {
}

Frame& Frame::operator=(const Frame& other) {
	m_id = other.m_id;
	m_detector = other.m_detector;
	m_focal = other.m_focal;
	m_pp = other.m_pp;
	m_minMatches = other.m_minMatches;
	m_features = other.m_features;
	m_landmarks = other.m_landmarks;
	m_keypointMap = other.m_keypointMap;
	m_poseMap = other.m_poseMap;
	m_landmarkMap = other.m_landmarkMap;
	m_T = other.m_T;
	m_P = other.m_P;

	return *this;
}

size_t Frame::id() const {
  return m_id;
}

const Features& Frame::getFeatures() const {
  return m_features;
}

const std::map<const Frame*, std::vector<KeypointID>>& Frame::keypointMap() const {
  return m_keypointMap;
}

const std::map<const Frame*, Pose>& Frame::poseMap() const {
  return m_poseMap;
}

bool Frame::compare(Frame& other) {
  std::vector<std::vector<cv::DMatch>> matches;
  std::vector<KeypointID> candidates1, candidates2;
  std::vector<cv::Point2f> points1, points2;
  std::vector<unsigned char> fundMask;
  cv::Mat essentialMask;

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
      points2.push_back(other.m_features.keyPoints[idx_2].pt);
    }
  }

  //Reject frame if not enough matches
  if(candidates1.size() < m_minMatches) {
    return false;
  }

  //Filter bad matches using fundamental matrix constraint
  cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, 3.0, 0.99, fundMask);

  //Construct list of inlier points for each frame
  std::vector<KeypointID> inliers1, inliers2;
  std::vector<cv::Point2f> inlierPts1, inlierPts2;
  for(size_t i = 0; i < points1.size(); ++i) {
    if(fundMask[i]) {
      inliers1.push_back(candidates1[i]);
      inliers2.push_back(candidates2[i]);
      
      inlierPts1.push_back(points1[i]);
      inlierPts2.push_back(points2[i]);
    }
  }

  //Calculate essential matrix
  auto E = cv::findEssentialMat(inlierPts2, inlierPts1, m_focal, m_pp, cv::RANSAC, 0.999,
    1.0, essentialMask);
  
  //Recover pose (rotation and translation) from essential matrix
  Pose pose;
  cv::recoverPose(E, inlierPts2, inlierPts1, pose.r, pose.t, m_focal, m_pp,
    essentialMask);

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
  m_keypointMap[&other] = inliers1;
  other.m_keypointMap[this] = inliers2;

  return true;
}

void Frame::addComparison(Frame& other, Pose&& pose, std::vector<KeypointID>&& keypoints1,
  std::vector<KeypointID>&& keypoints2) {
  
  m_poseMap.insert(std::make_pair<const Frame*, Pose>(&other, std::move(pose)));
  m_keypointMap.insert(std::make_pair<const Frame*, std::vector<KeypointID>>(&other,
    std::move(keypoints1)));
  other.m_keypointMap.insert(std::make_pair<const Frame*, std::vector<KeypointID>>(this,
    std::move(keypoints2)));
}

Features Frame::extractFeatures(const cv::Mat& image) const {
  return m_detector.detectKeyPoints(image);
}

bool Frame::hasPose(const Frame& other) const {
  return m_poseMap.count(&other) > 0;
}
/*
void Frame::setKeypoints(const Frame& other, std::vector<int>&& keypoints) {
	//m_keypointMap.emplace(std::make_pair<const Frame*, std::vector<int>>(&other,
		//std::move(keypoints)));
	m_keypointMap[&other] = keypoints;
}
*/
const Pose& Frame::getPose(const Frame& other) const {
  return m_poseMap.at(&other);
}
/*
void Frame::setPose(const Frame& other, Pose&& pose) {
	m_poseMap.insert(std::make_pair<const Frame*, Pose>(&other, std::move(pose)));
}
*/
bool Frame::hasKeypoints(const Frame& other) const {
  return m_keypointMap.count(&other) > 0;
}

size_t Frame::countMatchedKeypoints(const Frame& other) const {
  return m_keypointMap.at(&other).size();
}

const std::vector<KeypointID>& Frame::getKeypoints(const Frame& other) const {
  return m_keypointMap.at(&other);
}

cv::Point2f Frame::keypoint(KeypointID id) const {
  return m_features.keyPoints[id].pt;
}

bool Frame::hasLandmark(KeypointID id) const {
  return m_landmarkMap.count(id) > 0;
}

LandmarkID Frame::getLandmark(KeypointID id) const {
  return m_landmarkMap.at(id);
}

void Frame::addLandmark(KeypointID kpID, LandmarkID lID) {
  m_landmarkMap[kpID] = lID;
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

void Frame::writeFeatures(std::ostream& stream) const {
  stream << "feature_id,feature_x,feature_y\n";

  for(size_t i = 0; i < m_features.keyPoints.size(); ++i) {
    const auto& pt = m_features.keyPoints[i].pt;

    stream << i << ',' << pt.x << ',' << pt.y << '\n';
  }
}

void Frame::writeCovisibility(std::ostream& stream) const {
  stream << "other_frame,my_feature_id,other_feature_id\n";

  for(const auto& entry : m_keypointMap) {
    const auto& otherFrame = *(entry.first);
    const auto& myKeypoints = entry.second;
    const auto& otherKeypoints = otherFrame.m_keypointMap.at(this);
    
    for(size_t i = 0; i < myKeypoints.size(); ++i) {
      stream << otherFrame.id() << ',' << myKeypoints[i] << ','
        << otherKeypoints[i] << '\n';
    }
  }
}

void Frame::writeLandmarks(std::ostream& stream) const {
  stream << "feature_id,landmark_id\n";

  for(const auto& entry : m_landmarkMap) {
    stream << entry.first << ',' << entry.second << '\n';
  }
}

void Frame::writePose(std::ostream& stream) const {
  stream << m_T << '\n';
}

}
