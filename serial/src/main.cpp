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
  
  //Loop through every pair of frames to track feature correspondences accross frames
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

  return 0;
}
