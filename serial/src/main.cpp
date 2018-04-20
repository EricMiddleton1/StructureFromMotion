#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <string>
#include <iostream>
#include <stdexcept>
#include <thread>

#include <yaml-cpp/yaml.h>

#include "Config.hpp"
#include "DeviceManager.hpp"
#include "types.hpp"
#include "OrbDetector.hpp"
#include "Frame.hpp"

int main(void)
{
  //Load YAML configuration file
  Config config{"config.yml"};
  
  //Initialize objects based on YAML configuration parameters
  auto videoParams = config.getParams("video_device");
  auto videoName = std::find_if(videoParams.begin(), videoParams.end(),
    [](const auto& param) { return param.first == "name"; });
  if(videoName == videoParams.end()) {
    std::cerr << "[Error] Missing parameter video_device/name" << std::endl;
    return 1;
  }
  auto videoDevice =
    device_cast<VideoDevice>(DeviceManager::build(videoName->second,
  	std::move(videoParams)));
	
	SFM::ORBDetector orbDetector{config.getParams("feature_detector")};

  double focal = 1.0;
  cv::Point2d pp{0., 0.};

  std::vector<SFM::Frame> frames;

  std::cout << "[Info] Loading images and extracting features..." << std::endl;

  //Phase 1 - Load frames from disk and store in Frame datstructure
	cv::Mat image;
  auto startTime = cv::getTickCount();
  while(videoDevice->getFrame(image)) {
    //Save frame
    //This converts the frame to grayscale and
    //extracts and stores ORB features and descriptors
    frames.emplace_back(orbDetector, image, focal, pp);
  }

  auto endTime = cv::getTickCount();
	std::cout << "[Info] Loaded " << frames.size() << " frames ("
    << static_cast<float>(endTime - startTime)
		/ cv::getTickFrequency()*1000.f << "ms)" << std::endl;

  if(frames.empty()) {
    std::cerr << "[Error] Loaded 0 frames" << std::endl;
    return 1;
  }
  
  //Phase 2 - Loop through every pair of frames to track feature correspondences accross frames
  for(size_t i = 0; i < (frames.size()-1); ++i) {
    auto& frame1 = frames[i];

    std::cout << "[" << i << "]: ";

    for(size_t j = i+1; j < frames.size(); ++j) {
      if(frame1.compare(frames[j])) {
        std::cout << j << " ";
      }
    }
    std::cout << "\n" << std::endl;

    auto display = frame1.getImage().clone();
    cv::imshow("Current Frame", display);
    cv::waitKey(10);
  }

  
  //Camera Intrinsic Matrix
  cv::Mat k = cv::Mat::eye(3, 3, CV_64F);
  k.at<double>(0, 0) = focal;
  k.at<double>(1, 1) = focal;
  k.at<double>(0, 2) = pp.x;
  k.at<double>(1, 2) = pp.y;


  //Phase 3 - Recover motion between frames and triangle keypoints 2D->3D
  cv::Mat T = cv::Mat::eye(4, 4, CV_64F),
    P = cv::Mat::eye(3, 4, CV_64F);

  for(size_t i = 0; i < frames.size()-1; ++i) {
    auto& frame1 = frames[i];
    auto& frame2 = frames[i+1];

    if(frame1.hasKeypoints(frame2)) {
      const auto& pose = frame1.getPose(frame2);
      
      //Perform local transform
      cv::Mat T{cv::Mat::eye(4, 4, CV_64F)}, localR, localT;
      localR.copyTo(T(cv::Range(0, 3), cv::Range(0, 3)));
      localT.copyTo(T(cv::Range(0, 3), cv::Range(3, 4)));

      frame2.T(frame1.getTransformation()*T);

      //Create projection matrix
      cv::Mat r = frame2.T()(cv::Range(0, 3), cv::Range(0, 3));
      cv::Mat t = frame2.T()(cv::Range(0, 3), cv::Range(3, 4));
      cv::Mat P{3, 4, CV_64F};
      P(cv::Range(0, 3), cv::Range(0, 3)) = r.t();
      P(cv::Range(0, 3), cv::Range(3, 4)) = -r.t()*t;
      P = k*P;
      frame2.P(P);

      //Triangulate points
      cv::Mat points4D;
      cv::triangulatePoints(frame1.getProjectionMatrix(), frame2.getProjectionMatrix(),
        frame1.getKeypoints(frame2), frame2.getKeypoints(frame1), points4D);

      //Scale the triangulated points to match scale with existing 3D landmarks
      if(i > 0) {
        double scale = 0.;
        
        cv::Point3f frame1Pos{
          frame1.T().at<double>(0, 3),
          frame1.T().at<double>(1, 3),
          frame1.T().at<double>(2, 3)
        };

        std::vector<cv::Point3f> newPoints, existingPoints;



  return 0;
}
