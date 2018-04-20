#include "PointCloud.hpp"

namespace SFM {

PointID PointCloud::addPoint(const cv::Point3f& pt) {
  PointID id = m_points.size();

  m_points.push_back(pt);
  m_orders.push_back(2); //A new point must have been seen by at least 2 frames

  return id;
}

void PointCloud::addSighting(PointID pt) {
  m_orders[pt]++;
}

cv::Point3f PointCloud::getPoint(PointID pt) const {
  return m_points[pt];
}

size_t PointCloud::getOrder(PointID pt) const {
  retun m_orders[pt];
}

}
