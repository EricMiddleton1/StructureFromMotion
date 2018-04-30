#include "MPI_Util.hpp"

#include <opencv2/core/core.hpp>
#include "Frame.hpp"

#include <mpi.h>
#include <cstring>

namespace MPI_Util {

void sendMat(const cv::Mat& m, int destination) {
	cv::Mat mat = m;
	if(!mat.isContinuous()) {
		//Must be continuous for direct memory copy
		mat = mat.clone();
	}

	int rows = mat.rows, cols = mat.cols, type = mat.type(), channels = mat.channels();
	int size = rows*cols*channels; //Assuming 8 bits per channel

	char header[4*sizeof(int)];

	//Copy "header" information
	std::memcpy(header, &rows, sizeof(int));
	std::memcpy(header + sizeof(int), &cols, sizeof(int));
	std::memcpy(header + 2*sizeof(int), &type, sizeof(int));
	std::memcpy(header + 3*sizeof(int), &channels, sizeof(int));

	//Send header first
	MPI_Send(header, sizeof(header), MPI_CHAR, destination, 0, MPI_COMM_WORLD);

	char* buffer = new char[size];

	//Copy mat data into buffer
	std::memcpy(buffer, mat.data, size);

	//Send mat data
	MPI_Send(buffer, size, MPI_CHAR, destination, 0, MPI_COMM_WORLD);

	delete[] buffer;
}

cv::Mat recvMat(int from) {
	MPI_Status status;

	char header[4*sizeof(int)];
	MPI_Recv(header, sizeof(header), MPI_CHAR, from, 0, MPI_COMM_WORLD, &status);
	int rows, cols, type, channels, size;

	std::memcpy(&rows, header, sizeof(int));
	std::memcpy(&cols, header+sizeof(int), sizeof(int));
	std::memcpy(&type, header+2*sizeof(int), sizeof(int));
	std::memcpy(&channels, header+3*sizeof(int), sizeof(int));

	size = rows*cols*channels;

	char* buffer = new char[size];

	MPI_Recv(buffer, size, MPI_CHAR, from, 0, MPI_COMM_WORLD, &status);

	return {rows, cols, type, buffer};

	//TODO: Does the Mat object own the buffer (memory leak?)
}

void sendFrames(const std::vector<SFM::Frame>& frames, int destination) {
	MPI_Request request;

	//Collect information for header to send first
	int descriptorLength = frames[0].getFeatures().descriptors.cols;
	int descriptorType = frames[0].getFeatures().descriptors.type();
	int totalSize = 2*sizeof(int);

	for(const auto& frame : frames) {
		totalSize += sizeof(int) + frame.getFeatures().keyPoints.size() * 
			(sizeof(cv::KeyPoint) + descriptorLength);
	}

	int header[] = {totalSize, descriptorType};
	MPI_Isend(header, 2, MPI_INT, destination, 0, MPI_COMM_WORLD, &request);

	//Bulk data format:
	//  Total frame count (int)
	//  Descriptor length (int)
	//  For each frame:
	//    Number of features (int)
	//    Feature keypoint struct
	//    Feature matrix data
	
	char* buffer = new char[totalSize];
	int frameCount = frames.size();
	std::memcpy(buffer, &frameCount, sizeof(int));
	std::memcpy(buffer+sizeof(int), &descriptorLength, sizeof(int));

	int ptr = 2*sizeof(int);
	for(const auto& frame : frames) {
		const auto& features = frame.getFeatures();
		
		int kpCount = features.keyPoints.size();
		int kpSize = features.keyPoints.size() * sizeof(cv::KeyPoint);
		int descriptorSize = features.descriptors.rows * features.descriptors.cols;

		std::memcpy(buffer+ptr, &kpCount, sizeof(int));
		ptr += sizeof(int);
		
		std::memcpy(buffer+ptr, features.keyPoints.data(), kpSize);
		ptr += kpSize;

		std::memcpy(buffer+ptr, features.descriptors.data, descriptorSize);
		ptr += descriptorSize;
	}

	MPI_Isend(buffer, totalSize, MPI_CHAR, destination, 1, MPI_COMM_WORLD, &request);
}

std::vector<SFM::Frame> recvFrames(int from, SFM::ORBDetector& detector, int startID,
	double focal, const cv::Vec2d& pp) {
	std::vector<SFM::Frame> frames;
	MPI_Status status;

	int header[2];
	MPI_Recv(header, 2, MPI_INT, from, 0, MPI_COMM_WORLD, &status);

	int totalSize = header[0], descriptorType = header[1];

	char* buffer = new char[totalSize];
	MPI_Recv(buffer, totalSize, MPI_CHAR, from, 1, MPI_COMM_WORLD, &status);

	int frameCount, descriptorSize;
	std::memcpy(&frameCount, buffer, sizeof(int));
	std::memcpy(&descriptorSize, buffer+sizeof(int), sizeof(int));
	int ptr = 2*sizeof(int);

	while(ptr < totalSize) {
		SFM::Features features;

		int kpCount;
		std::memcpy(&kpCount, buffer+ptr, sizeof(int));
		ptr += sizeof(int);

		features.keyPoints.resize(kpCount);
		std::memcpy(features.keyPoints.data(), buffer+ptr, kpCount*sizeof(cv::KeyPoint));
		ptr += kpCount*sizeof(cv::KeyPoint);

		cv::Mat temp(kpCount, descriptorSize, descriptorType, buffer + ptr);
		features.descriptors = temp.clone();
		ptr += kpCount*descriptorSize;

		frames.emplace_back(startID + frames.size(), detector, std::move(features), focal,
			pp);
	}

	return frames;
}

void sendCovisibility(const std::vector<SFM::Frame>& frames, int start, int end) {
	int totalSize = 0, rSize = 0, tSize, rType = 0, tType = 0;

	//Data format:
	//For each frame:
	//	Frame ID
	//	Covisible Frame Count
	//  For each covisible frame:
	//		Covis frame ID
	//		R matrix
	//		T matrix
	//  	Keypoint count
	//  	keypoints (frame 1)
	//  	keypoints (frame 2)
	
	for(int i = start; i < end; ++i) {
		const auto& frame1 = frames[i];
		const auto& poseMap = frame1.poseMap();

		totalSize += 2*sizeof(int); //ID, Count

		for(const auto& pair : poseMap) {
			const auto& frame2 = *pair.first;
			const auto& pose = pair.second;

			const auto& keypoints = frame1.getKeypoints(frame2);

			totalSize += sizeof(int); //ID
			totalSize += pose.r.rows * pose.r.cols * pose.r.elemSize1(); //R
			totalSize += pose.t.rows * pose.t.cols * pose.t.elemSize1(); //T
			totalSize += sizeof(int); //Keypoint count
			totalSize += 2*sizeof(int)*keypoints.size(); //Keypoints arrays

			rSize = pose.r.rows * pose.r.cols * pose.r.elemSize1();
			rType = pose.r.type();
			tSize = pose.t.rows * pose.t.cols * pose.t.elemSize1();
			tType = pose.t.type();
		}
	}

	//Send header
	int header[] = {totalSize, rSize, rType, tSize, tType};
	MPI_Send(&header, 5, MPI_INT, 0, 0, MPI_COMM_WORLD);

	char* buffer = new char[totalSize];
	int ptr = 0;

	for(int i = start; i < end; ++i) {
		const auto& frame1 = frames[i];
		const auto& poseMap = frame1.poseMap();

		int id = frame1.id();
		int count = poseMap.size();
		
		std::memcpy(buffer + ptr, &id, sizeof(int));
		ptr += sizeof(int);
		std::memcpy(buffer + ptr, &count, sizeof(int));
		ptr += sizeof(int);


		for(const auto& pair : poseMap) {
			const auto& frame2 = *pair.first;
			const auto& pose = pair.second;

			int covisID = frame2.id();
			std::memcpy(buffer + ptr, &covisID, sizeof(int));
			ptr += sizeof(int);

			std::memcpy(buffer + ptr, pose.r.data, rSize);
			ptr += rSize;

			std::memcpy(buffer + ptr, pose.t.data, tSize);
			ptr += tSize;

			const auto& keypoints = frame1.getKeypoints(frame2);

			int kpCount = keypoints.size();
			std::memcpy(buffer + ptr, &kpCount, sizeof(int));
			ptr += sizeof(int);

			std::memcpy(buffer + ptr, keypoints.data(), kpCount*sizeof(int));
			ptr += kpCount * sizeof(int);

			const auto& otherKP = frame2.getKeypoints(frame1);
			std::memcpy(buffer + ptr, otherKP.data(), kpCount*sizeof(int));
			ptr += kpCount * sizeof(int);
		}
	}

	MPI_Send(buffer, totalSize, MPI_CHAR, 0, 0, MPI_COMM_WORLD);

	delete[] buffer;
}

void recvCovisibility(std::vector<SFM::Frame>& frames, int comm_sz) {
	MPI_Status status;

	int header[5];

	//Data format:
	//For each frame:
	//	Frame ID
	//	Covisible Frame Count
	//  For each covisible frame:
	//		Covis frame ID
	//		Pose
	//  	Keypoint count
	//  	keypoints (frame 1)
	//  	keypoints (frame 2)

	for(int i = 1; i < comm_sz; ++i) {
		MPI_Recv(header, 5, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
		int totalSize = header[0];
		int rSize = header[1];
		int rType = header[2];
		int tSize = header[3];
		int tType = header[4];

		char* buffer = new char[totalSize];
		int ptr = 0;
		MPI_Recv(buffer, totalSize, MPI_CHAR, i, 0, MPI_COMM_WORLD, &status);

		while(ptr < totalSize) {
			int frameID;
			std::memcpy(&frameID, buffer + ptr, sizeof(int));
			ptr += sizeof(int);
			auto& frame1 = frames[frameID];

			int covisCount;
			std::memcpy(&covisCount, buffer + ptr, sizeof(int));
			ptr += sizeof(int);

			for(int j = 0; j < covisCount; ++j) {
				int covisFrameID;
				std::memcpy(&covisFrameID, buffer + ptr, sizeof(int));
				ptr += sizeof(int);

				auto& frame2 = frames[covisFrameID];

				SFM::Pose pose{cv::Mat(3, 3, rType), cv::Mat(3, 1, tType)};
				std::memcpy(pose.r.data, buffer + ptr, rSize);
				ptr += rSize;
				std::memcpy(pose.t.data, buffer + ptr, tSize);
				ptr += tSize;

				int kpCount;
				std::memcpy(&kpCount, buffer + ptr, sizeof(int));
				ptr += sizeof(int);

				std::vector<int> kp1(kpCount), kp2(kpCount);
				std::memcpy(kp1.data(), buffer + ptr, kpCount * sizeof(int));
				ptr += kpCount * sizeof(int);
				std::memcpy(kp2.data(), buffer + ptr, kpCount * sizeof(int));
				ptr += kpCount * sizeof(int);

				frame1.addComparison(frame2, std::move(pose), std::move(kp1), std::move(kp2));
			}
		}

		delete[] buffer;
	}
}

}
