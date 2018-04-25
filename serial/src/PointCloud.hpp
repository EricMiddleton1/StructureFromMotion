#pragma once

#include <vector>
#include <opencv2/core/core.hpp>
#include <iostream>

#include "Frame.hpp"
#include "types.hpp"

namespace SFM {
  class PointCloud {
  public:
    LandmarkID addPoint(const cv::Point3f& pt);
    void addSighting(LandmarkID pt, const cv::Point3f& newPos);

    cv::Point3f getPoint(LandmarkID pt) const;
    size_t getOrder(LandmarkID pt) const;

    void write(std::ostream& stream) const;

  private:
    std::vector<cv::Point3f> m_points;
    std::vector<size_t> m_orders;
  };
}
