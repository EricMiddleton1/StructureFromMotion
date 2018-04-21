#include "ImageFolderDevice.hpp"

#include <iostream>

#include <opencv2/imgcodecs.hpp>

DeviceRegistration ImageFolderDevice::registration_{{"image_folder",
  [](std::vector<VideoDevice::Param>&& params) {
    return std::make_unique<ImageFolderDevice>(std::move(params));
  }}};

ImageFolderDevice::ImageFolderDevice(std::vector<Param>&& params)
  : VideoDevice({"folder_path"}, std::move(params))
  , m_curImage{0} {
  
  cv::glob(getParam("folder_path"), m_images, false);
  
  if(m_images.empty()) {
    throw std::runtime_error("ImageFolderDevice: Didn't find any images in folder '"
      + getParam("folder_path") + "'");
  }
}

bool ImageFolderDevice::getFrame(cv::Mat& out) {
  if(m_curImage >= m_images.size()) {
    return false;
  }
  
  out = cv::imread(m_images[m_curImage++]);

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
