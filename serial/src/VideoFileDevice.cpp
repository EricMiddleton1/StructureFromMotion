#include "VideoFileDevice.hpp"

DeviceRegistration VideoFileDevice::registration_{{"video file",
  [](std::vector<VideoDevice::Param>&& params) {
    return std::make_unique<VideoFileDevice>(std::move(params));
  }}};

VideoFileDevice::VideoFileDevice(std::vector<Param>&& params)
  : VideoDevice({"file"}, std::move(params))
  , m_frameCount{0}
  , m_maxFrameCount{std::stoi(getParam("max frames", "-1"))} {

  cap_.open(getParam("file"));
  if(!cap_.isOpened()) {
    throw std::runtime_error("VideoFileDevice: Failed to file " + getParam("file"));
  }
}

bool VideoFileDevice::getFrame(cv::Mat& out) {
  m_frameCount++;
  if(m_maxFrameCount != -1 && m_frameCount > m_maxFrameCount) {
    return false;
  }

  cap_ >> out;
  if(out.empty()) {
    return false;
  }
  else {
    if(paramExists("max_height")) {
      auto maxHeight = std::stoi(getParam("max_height"));
      if(out.size().height > maxHeight) {
        out = resize(out, maxHeight);
      }
    }
    
    if(paramExists("color_mode") && (getParam("color_mode") == "bw")) {
      cv::Mat gray;
      cv::cvtColor(out, gray, cv::COLOR_RGB2GRAY);
      out = gray;
    }

		display(out);

    return true;
  }
}
