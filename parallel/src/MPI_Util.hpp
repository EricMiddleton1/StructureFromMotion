#pragma once

#include <opencv2/core/core.hpp>
#include "Frame.hpp"

namespace MPI_Util {
	//Send cv::Mat to other process
	void sendMat(const cv::Mat& mat, int destination);

	//Receive cv::Mat from other process
	cv::Mat recvMat(int from);

	//Send Frame datastructure to other process
	void sendFrames(const std::vector<SFM::Frame>& frames, int destination);

	//Receive Frame datastructure from other process
	std::vector<SFM::Frame> recvFrames(int from, SFM::ORBDetector& detector, int startID,
		double focal, const cv::Vec2d& pp);

	//Send covsibility info to root
	void sendCovisibility(const std::vector<SFM::Frame>& frames, int start, int end);

	//Receive covisibility info from nodes
	void recvCovisibility(std::vector<SFM::Frame>& frames, int comm_sz);
}
