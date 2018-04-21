#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

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
#include "PointCloud.hpp"

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

  //Essential datastructures: vector of frames, point cloud
  std::vector<SFM::Frame> frames;
  SFM::PointCloud pointCloud;

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
  
  std::cout << "[Info] Tracking features between all pairs of frames" << std::endl;
  //Phase 2 - Loop through every pair of frames to track feature correspondences accross frames
  for(size_t i = 0; i < (frames.size()-1); ++i) {
    auto& frame1 = frames[i];

    std::cout << "[" << i << "]: ";

    for(size_t j = i+1; j < i+2/*frames.size()*/; ++j) {
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

  std::cout << "[Info] Tracking motion between pairs of adjacent frames" << std::endl;
  //Phase 3 - Recover motion between frames and triangle keypoints 2D->3D
  cv::Mat T = cv::Mat::eye(4, 4, CV_64F),
    P = cv::Mat::eye(3, 4, CV_64F);

  for(size_t i = 0; i < frames.size()-1; ++i) {
    auto& frame1 = frames[i];
    auto& frame2 = frames[i+1];

    if(frame1.hasKeypoints(frame2)) {
      const auto& pose = frame1.getPose(frame2);
      
      //Perform local transform
      cv::Mat T{cv::Mat::eye(4, 4, CV_64F)}, localR = pose.r, localT = pose.t;
      localR.copyTo(T(cv::Range(0, 3), cv::Range(0, 3)));
      localT.copyTo(T(cv::Range(0, 3), cv::Range(3, 4)));

      frame2.T(frame1.T()*T);

      //Create projection matrix
      cv::Mat r = frame2.T()(cv::Range(0, 3), cv::Range(0, 3));
      cv::Mat t = frame2.T()(cv::Range(0, 3), cv::Range(3, 4));
      cv::Mat P(3, 4, CV_64F);
      
      P(cv::Range(0, 3), cv::Range(0, 3)) = r.t();
      P(cv::Range(0, 3), cv::Range(3, 4)) = -r.t()*t;
      P = k*P;
      frame2.P(P);

      //Triangulate points
      cv::Mat points4d;
      cv::triangulatePoints(frame1.P(), frame2.P(),
        frame1.getKeypoints(frame2), frame2.getKeypoints(frame1), points4d);

      //Scale the triangulated points to match scale with existing 3D landmarks
      if(i > 0) {
        double scale = 0.;
        size_t count = 0;
        
        cv::Point3f frame1Pos{
          frame1.T().at<double>(0, 3),
          frame1.T().at<double>(1, 3),
          frame1.T().at<double>(2, 3)
        };

        //std::cout << "[Info] Finding existing landmarks in frame " << i << std::endl;
        //Find existing 3D landmarks that are visible in current frame2
        std::vector<cv::Point3f> newPoints, existingPoints;
        auto frame1Keypoints = frame1.getKeypoints(frame2);
        for(size_t j = 0; j < frame1Keypoints.size(); ++j) {
          if(frame1.hasLandmark(frame1Keypoints[j])) {
            cv::Point3f pt3d;
            auto ptID = frame1.getLandmark(frame1Keypoints[j]);
            auto avgLandmark = pointCloud.getPoint(ptID) /
              static_cast<float>(pointCloud.getOrder(ptID)-1);

            pt3d.x = points4d.at<float>(0, j) / points4d.at<float>(3, j);
            pt3d.y = points4d.at<float>(1, j) / points4d.at<float>(3, j);
            pt3d.z = points4d.at<float>(2, j) / points4d.at<float>(3, j);

            newPoints.push_back(pt3d);
            existingPoints.push_back(avgLandmark);
          }
        }

        //std::cout << "[Info] Calculating scale factor for 3D landmarks" << std::endl;
        //std::cout << "[Info] Using " << newPoints.size() << " points" << std::endl;
        //Calculate average ratio of distance for all landmark/point matches
        //TODO: Consider using RANSAC here if outliers are a problem
        for(int j = 0; j < newPoints.size()-1; ++j) {
          for(int k = j+1; k < newPoints.size(); ++k) {
            //std::cout << "[Info] Scale for point (" << j << ", " << k << ") of "
              //<< newPoints.size() << std::endl;
            double s = cv::norm(existingPoints[j] - existingPoints[k]) /
              cv::norm(newPoints[j] - newPoints[k]);

            scale += s;
            ++count;
          }
        }
        //std::cout << "[Info] Scale total = " << scale << ", from " << count << " landmarks"
          //<< std::endl;
        //TODO: deal with possible division by zero
        scale /= count;

        //std::cout << "[Info] Frame " << i << " has relative scale " << scale <<
          //" and " << newPoints.size() << " landmark matches with previous frame" << std::endl;


        //Scale unit translation vector by scale factor and recalculate T, P
        localT *= scale;
        cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
        localR.copyTo(T(cv::Range(0, 3), cv::Range(0, 3)));
        localT.copyTo(T(cv::Range(0, 3), cv::Range(3, 4)));

        std::cout << "[Info] Frame " << i+1 << " position: " << localT << std::endl;
        
        //Update global frame2 position based on rescaled T
        //TODO: Consider also rescaling relative pose between current frame and previous frame
        //TODO NOTE: It would be useful to calculate this relative pose and scaling factor
        //between all covisible frames
        frame2.T(frame1.T()*T);

        //Make new projection matrix
        //TODO: Make sure this is correct
        cv::Mat P(3, 4, CV_64F);
        P(cv::Range(0, 3), cv::Range(0, 3)) = localR.t();
        P(cv::Range(0, 3), cv::Range(3, 4)) = -localR.t()*localT;
        P = k*P;

        //Update frame2 projection matrix with correct relative scaling
        frame2.P(P);

        //Re-triangulate points based on new projection matrix with correct relative scale
        cv::triangulatePoints(frame1.P(), frame2.P(), 
          frame1.getKeypoints(frame2), frame2.getKeypoints(frame1), points4d);
      }//End of block if(i > 0)

      //Loop through matched points and update point cloud, frame landmark maps
      const auto& f1Keypoints = frame1.getKeypoints(frame2);
      const auto& f2Keypoints = frame2.getKeypoints(frame1);
      for(size_t j = 0; j < f1Keypoints.size(); ++j) {
        cv::Point3f pt3d;

        pt3d.x = points4d.at<float>(0, j) / points4d.at<float>(3, j);
        pt3d.y = points4d.at<float>(1, j) / points4d.at<float>(3, j);
        pt3d.z = points4d.at<float>(2, j) / points4d.at<float>(3, j);

        if(frame1.hasLandmark(f1Keypoints[j])) {
          //Add existing landmark to frame2 landmark map
          auto id = frame1.getLandmark(f1Keypoints[j]);
          frame2.addLandmark(f2Keypoints[j], id);

          //Add new sighting of landmark to point cloud datastructure
          pointCloud.addSighting(id, pt3d);

          //std::cout << "[Info] Added new sighting of existing landmark " << id << std::endl;
        }
        else {
          //This is a new landmark

          //Add to point cloud datastructure
          auto id = pointCloud.addPoint(pt3d);

          //Add to landmark maps for frame1, frame2
          frame1.addLandmark(f1Keypoints[j], id);
          frame2.addLandmark(f2Keypoints[j], id);
          //std::cout << "[Info] Added new landmark " << id << std::endl;
        }
      }
    }
  }

  return 0;
}
