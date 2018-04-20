#pragma once

#include <opencv2/videoio.hpp>
#include <opencv2/core/core.hpp>

#include "VideoDevice.hpp"
#include "DeviceRegistration.hpp"

class ImageFolderDevice : public VideoDevice {
public:
  ImageFolderDevice(std::vector<Param>&& params);

  bool getFrame(cv::Mat& out) override;
private:
  static DeviceRegistration registration_;
  
  std::vector<cv::String> m_images;
  size_t m_curImage;
};
