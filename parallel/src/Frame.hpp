#pragma once

#include <vector>
#include <utility>
#include <map>
#include <functional>

#include <opencv2/features2d.hpp>

#include "OrbDetector.hpp"
#include "types.hpp"

namespace SFM {
  class Frame {
  public:
    //Construct from image. Extract image features inside constructor
    Frame(size_t id, ORBDetector& detector, const cv::Mat& image, double focal,
      const cv::Vec2d& pp, int minMatches = 10);

    //Construct from set of features
    Frame(size_t id, ORBDetector& detector, Features&& features, double focal,
      const cv::Vec2d& pp, int minMatches = 10);

		Frame(const Frame& other);

		Frame& operator=(const Frame& other);

    size_t id() const;

    bool compare(Frame& other);
    void addComparison(Frame& other, Pose&& pose, std::vector<KeypointID>&& keypoints1,
      std::vector<KeypointID>&& keypoints2);

    const std::map<const Frame*, std::vector<KeypointID>>& keypointMap() const;
    const std::map<const Frame*, Pose>& poseMap() const;

    const Features& getFeatures() const;

    bool hasPose(const Frame& other) const;
    const Pose& getPose(const Frame& other) const;
		void setPose(const Frame& other, Pose&& pose);

    bool hasKeypoints(const Frame& other) const;
    size_t countMatchedKeypoints(const Frame& other) const;
    const std::vector<KeypointID>& getKeypoints(const Frame& frame) const;
    cv::Point2f keypoint(KeypointID id) const;

		void setKeypoints(const Frame& other, std::vector<int>&& keypoints);

    bool hasLandmark(KeypointID id) const;
    LandmarkID getLandmark(KeypointID id) const;
    void addLandmark(KeypointID kpID, LandmarkID lID);

    cv::Mat T() const;
    void T(const cv::Mat& t);

    cv::Mat P() const;
    void P(const cv::Mat& p);

    void writeFeatures(std::ostream& stream) const;
    void writeCovisibility(std::ostream& stream) const;
    void writeLandmarks(std::ostream& stream) const;
    void writePose(std::ostream& stream) const;

  private:
    Features extractFeatures(const cv::Mat& image) const;

    size_t m_id;

    ORBDetector& m_detector;

    //cv::Mat m_image;
    double m_focal;
    cv::Vec2d m_pp;
    int m_minMatches;

    Features m_features;
    std::vector<LandmarkID> m_landmarks;

    std::map<const Frame*, std::vector<KeypointID>> m_keypointMap;//Maps common keypoints
                                                                  //to other frame
    std::map<const Frame*, Pose> m_poseMap;           //Maps pose between this frame and
                                                      //other frame
    std::map<KeypointID, LandmarkID> m_landmarkMap;   //Maps point to landmark

    //4x4 pose transformation matrix
    cv::Mat m_T;
    //3x4 projection matrix
    cv::Mat m_P;
  };
}
