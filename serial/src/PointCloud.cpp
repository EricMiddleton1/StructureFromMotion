#include "PointCloud.hpp"

namespace SFM {

LandmarkID PointCloud::addPoint(const cv::Point3f& pt) {
  LandmarkID id = m_points.size();

  m_points.push_back(pt);
  m_orders.push_back(2); //A new point must have been seen by at least 2 frames

  return id;
}

void PointCloud::addSighting(LandmarkID pt, const cv::Point3f& newPos) {
  m_points[pt] += newPos;
  m_orders[pt]++;
}

cv::Point3f PointCloud::getPoint(LandmarkID pt) const {
  //m_points array contains sum of 3D position accross all sightings
  return m_points[pt] / static_cast<float>(m_orders[pt]-1);
}

size_t PointCloud::getOrder(LandmarkID pt) const {
  return m_orders[pt];
}

}
