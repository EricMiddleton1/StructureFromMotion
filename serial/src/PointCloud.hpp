#pragma once

#include <vector>
#include <opencv2/core/core.hpp>

#include "Frame.hpp"
#include "types.hpp"

namespace SFM {
  class PointCloud {
  public:
    PointID addPoint(const cv::Point3f& pt);
    void addSighting(PointID pt);

    cv::Point3f getPoint(PointID pt) const;
    size_t getOrder(PointID pt) const;

  private:
    std::vector<cv::Point3f> m_points;
    std::vector<size_t> m_orders;
  };
}
