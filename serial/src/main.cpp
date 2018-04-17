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
#include "OrbDetector.hpp"
#include "EssentialComputer.hpp"

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
	
	auto orbDetector = std::make_unique<ORBDetector>(std::move(config.getParams("feature_detector")));

	auto essentialComputer = std::make_unique<EssentialComputer>(
		std::move(config.getParams("essential_computer")));

	auto startTime = cv::getTickCount();
	
	cv::Mat prevFrame;
	ORBDetector::Features prevFeatures;
	bool valid = false;

  while(cv::waitKey(1) == -1) {
    cv::Mat frame, display;

    //Try to grab frame from video device
    if(!videoDevice->getFrame(frame)) {
      std::cerr << "[Warning] Failed to fetch frame" << std::endl;
      continue;
    }

		display = frame.clone();

		//Detect ORB features
		auto features = orbDetector->detectKeyPoints(frame);

		if(valid) {
			//Match features between this frame and the previous frame
			auto matchedFeatures = orbDetector->matchFeatures(prevFeatures, features);
			
			orbDetector->draw(display, matchedFeatures.second);

			//Calculate essential matrix and extract pose
			cv::Mat r, t;
			if(essentialComputer->computePose(matchedFeatures.first, matchedFeatures.second,
				r, t)) {

				std::cout << "[Info] Recovered camera pose:\n" << t << std::endl;
			}
			else {
				std::cout << "[Warning] Could not recover pose" << std::endl;
			}
		}

		imshow("Features", display);

		prevFrame = frame.clone();
		prevFeatures = features;
		valid = true;

    //Stop loop stopwatch
		auto endTime = cv::getTickCount();
		std::cout << "[Info] Processed frame in " << static_cast<float>(endTime - startTime)
			/ cv::getTickFrequency()*1000.f << "ms" << std::endl;
    startTime = endTime;

  }

  return 0;
}
