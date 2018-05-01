#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <string>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <thread>

#include <mpi.h>

#include <yaml-cpp/yaml.h>

#include "Config.hpp"
#include "DeviceManager.hpp"
#include "types.hpp"
#include "OrbDetector.hpp"
#include "Frame.hpp"
#include "PointCloud.hpp"
#include "MPI_Util.hpp"

using namespace SFM;

float cvTickToMs(int64_t start, int64_t end) {
	return static_cast<float>(end - start) / cv::getTickFrequency() * 1000.f;
}

int main(int argc, char* argv[])
{
	int my_rank, comm_sz;

	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	int cvThreads = 1;
	if(argc == 2) {
		cvThreads = std::stoi(argv[1]);
	}

	cv::setNumThreads(cvThreads);

  //Load YAML configuration file
  Config config{"config.yml"};
	SFM::ORBDetector orbDetector{config.getParams("feature_detector")};
	double focal = 484;
  cv::Point2d pp{311.9607, 219.6259};

	//Local set of images
	std::vector<cv::Mat> localImages;
	std::vector<Frame> localFrames;
	int totalImages;

  //Essential datastructures: vector of frames, point cloud
  std::vector<SFM::Frame> frames;
  SFM::PointCloud pointCloud;

	int64_t startTime, loadDone, loadTransferDone, featureDone, featureTransferDone,
		covisDone, covisTransferDone, motionDone;

  if(my_rank == 0) {
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

		startTime = cv::getTickCount();

		//Load images from disk
		cv::Mat img;
		while(videoDevice->getFrame(img)) {
			localImages.emplace_back(img.clone());
		}

		totalImages = localImages.size();

		loadDone = cv::getTickCount();
	}

	//Broadcast image count
	MPI_Bcast(&totalImages, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if(my_rank == 0) {
		//Scatter images to processors
		for(size_t i = 1; i < comm_sz; ++i) {
			size_t start = i*localImages.size()/comm_sz,
				end = (i+1)*localImages.size()/comm_sz;

			//MPI_Util::sendMats(localImages, start, end, i);
			for(int j = start; j < end; ++j) {
				MPI_Util::sendMat(localImages[j], i);
			}
		}

		//Only keep root's local set
		localImages.resize(localImages.size()/comm_sz);
	}
	else {
		int myImageCount = (my_rank+1)*totalImages/comm_sz - my_rank*totalImages/comm_sz;

		for(int i = 0; i < myImageCount; ++i) {
			localImages.push_back(MPI_Util::recvMat(0));
		}
	}


	size_t imageOffset = my_rank*totalImages/comm_sz;
	MPI_Barrier(MPI_COMM_WORLD);


	if(my_rank == 0) {
		loadTransferDone = cv::getTickCount();
	}

	//Extract features from local set
	for(size_t i = 0; i < localImages.size(); ++i) {
		localFrames.emplace_back(i+imageOffset, orbDetector, localImages[i], focal, pp);
	}
	
	MPI_Barrier(MPI_COMM_WORLD);

	if(my_rank == 0) {
		featureDone = cv::getTickCount();
	}

	//Send local frames to all other frames
	for(int i = 0; i < comm_sz; ++i) {
		if(i != my_rank) {
			//Non-blocking
			MPI_Util::sendFrames(localFrames, i);
		}
	}

	//Receive frames
	for(int i = 0; i < comm_sz; ++i) {
		if(i == my_rank) {
			frames.insert(frames.end(), localFrames.begin(), localFrames.end());
			localFrames.clear();
		}
		else {
			size_t count = ((i+1)*totalImages/comm_sz) - (i*totalImages/comm_sz);
			auto received = MPI_Util::recvFrames(i, orbDetector, i*totalImages/comm_sz,
				focal, pp);
			frames.insert(frames.end(), received.begin(), received.end());
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	
	if(my_rank == 0) {
		featureTransferDone = cv::getTickCount();
	}

  //Phase 2 - Loop through every pair of frames to track feature correspondences accross frames
	size_t frameStart = my_rank*(totalImages-1)/comm_sz,
		frameEnd = (my_rank+1)*(totalImages-1)/comm_sz;
  for(size_t i = frameStart; i < frameEnd; ++i) {
    auto& frame1 = frames[i];

    for(size_t j = i+1; j < frames.size(); ++j) {
      frame1.compare(frames[j]);
    }
  }

	MPI_Barrier(MPI_COMM_WORLD);
	if(my_rank == 0) {
		covisDone = cv::getTickCount();
	}

	//Send covisibility information back to root
	if(my_rank != 0) {
		MPI_Util::sendCovisibility(frames, frameStart, frameEnd);
	}
	else {
		//Receive covisibility information from each node
		MPI_Util::recvCovisibility(frames, comm_sz);
	}

	if(my_rank == 0) {
		covisTransferDone = cv::getTickCount();

		//Camera Intrinsic Matrix
		cv::Mat k = cv::Mat::eye(3, 3, CV_64F);
		k.at<double>(0, 0) = focal;
		k.at<double>(1, 1) = focal;
		k.at<double>(0, 2) = pp.x;
		k.at<double>(1, 2) = pp.y;

		//Phase 3 - Recover motion between frames and triangle keypoints 2D->3D
		cv::Mat T = cv::Mat::eye(4, 4, CV_64F),
			P = cv::Mat::eye(3, 4, CV_64F);

		for(size_t i = 0; i < frames.size()-1; ++i) {
			auto& frame1 = frames[i];
			auto& frame2 = frames[i+1];

			if(frame1.hasKeypoints(frame2)) {
				const auto& pose = frame1.getPose(frame2);
				const auto& kpID1 = frame1.getKeypoints(frame2);
				const auto& kpID2 = frame2.getKeypoints(frame1);

				std::vector<cv::Point2f> points1, points2;
				for(size_t j = 0; j < kpID1.size(); ++j) {
					points1.push_back(frame1.keypoint(kpID1[j]));
					points2.push_back(frame2.keypoint(kpID2[j]));
				}
				
				//Perform local transform
				cv::Mat T{cv::Mat::eye(4, 4, CV_64F)}, localR = pose.r, localT = pose.t;
				localR.copyTo(T(cv::Range(0, 3), cv::Range(0, 3)));
				localT.copyTo(T(cv::Range(0, 3), cv::Range(3, 4)));

				frame2.T(frame1.T()*T);

				//Create projection matrix
				cv::Mat r = frame2.T()(cv::Range(0, 3), cv::Range(0, 3));
				cv::Mat t = frame2.T()(cv::Range(0, 3), cv::Range(3, 4));
				cv::Mat P(3, 4, CV_64F);
				
				P(cv::Range(0, 3), cv::Range(0, 3)) = r.t();
				P(cv::Range(0, 3), cv::Range(3, 4)) = -r.t()*t;
				P = k*P;
				frame2.P(P);

				//Triangulate points
				cv::Mat points4d;
				cv::triangulatePoints(frame1.P(), frame2.P(), points1, points2, points4d);

				//Scale the triangulated points to match scale with existing 3D landmarks
				if(i > 0) {
					double scale = 0.;
					size_t count = 0;
					
					cv::Point3f frame1Pos{
						frame1.T().at<double>(0, 3),
						frame1.T().at<double>(1, 3),
						frame1.T().at<double>(2, 3)
					};

					//Find existing 3D landmarks that are visible in current frame2
					std::vector<cv::Point3f> newPoints, existingPoints;
					for(size_t j = 0; j < kpID1.size(); ++j) {
						if(frame1.hasLandmark(kpID1[j])) {
							cv::Point3f pt3d;
							auto ptID = frame1.getLandmark(kpID1[j]);
							auto avgLandmark = pointCloud.getPoint(ptID);

							pt3d.x = points4d.at<float>(0, j) / points4d.at<float>(3, j);
							pt3d.y = points4d.at<float>(1, j) / points4d.at<float>(3, j);
							pt3d.z = points4d.at<float>(2, j) / points4d.at<float>(3, j);

							newPoints.push_back(pt3d);
							existingPoints.push_back(avgLandmark);
						}
					}
					
					if(newPoints.size() < 2) {
						continue;
					}
					//Calculate average ratio of distance for all landmark/point matches
					//TODO: Consider using RANSAC here if outliers are a problem
					for(int j = 0; j < newPoints.size()-1; ++j) {
						for(int k = j+1; k < newPoints.size(); ++k) {
							double s = cv::norm(existingPoints[j] - existingPoints[k]) /
								cv::norm(newPoints[j] - newPoints[k]);

							scale += s;
							++count;
						}
					}
					//TODO: deal with possible division by zero
					scale /= count;

					//Scale unit translation vector by scale factor and recalculate T, P
					localT *= scale;
					cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
					localR.copyTo(T(cv::Range(0, 3), cv::Range(0, 3)));
					localT.copyTo(T(cv::Range(0, 3), cv::Range(3, 4)));

					//Update global frame2 position based on rescaled T
					//TODO: Consider also rescaling relative pose between current frame and previous frame
					//TODO NOTE: It would be useful to calculate this relative pose and scaling factor
					//between all covisible frames
					frame2.T(frame1.T()*T);

					//Make new projection matrix
					//TODO: Make sure this is correct
					r = frame2.T()(cv::Range(0, 3), cv::Range(0, 3));
					t = frame2.T()(cv::Range(0, 3), cv::Range(3, 4));
					cv::Mat P(3, 4, CV_64F);
					P(cv::Range(0, 3), cv::Range(0, 3)) = r.t();
					P(cv::Range(0, 3), cv::Range(3, 4)) = -r.t()*t;
					P = k*P;

					//Update frame2 projection matrix with correct relative scaling
					frame2.P(P);

					//Re-triangulate points based on new projection matrix with correct relative scale
					cv::triangulatePoints(frame1.P(), frame2.P(), points1, points2, points4d);
				}//End of block if(i > 0)

				//Loop through matched points and update point cloud, frame landmark maps
				for(size_t j = 0; j < kpID1.size(); ++j) {
					cv::Point3f pt3d;

					pt3d.x = points4d.at<float>(0, j) / points4d.at<float>(3, j);
					pt3d.y = points4d.at<float>(1, j) / points4d.at<float>(3, j);
					pt3d.z = points4d.at<float>(2, j) / points4d.at<float>(3, j);

					if(frame1.hasLandmark(kpID1[j])) {
						//Add existing landmark to frame2 landmark map
						auto id = frame1.getLandmark(kpID1[j]);
						frame2.addLandmark(kpID2[j], id);

						//Add new sighting of landmark to point cloud datastructure
						pointCloud.addSighting(id, pt3d);
					}
					else {
						//This is a new landmark

						//Add to point cloud datastructure
						auto id = pointCloud.addPoint(pt3d);

						//Add to landmark maps for frame1, frame2
						frame1.addLandmark(kpID1[j], id);
						frame2.addLandmark(kpID2[j], id);
					}
				}
			}
		}
		motionDone = cv::getTickCount();
		
		for(const auto& frame : frames) {
			std::string name{std::string("frame") + std::to_string(frame.id())};

			//Dump the results to disk
			std::fstream fFeatures{name + "_features.csv", std::fstream::out},
				fCovis{name + "_covisibility.csv", std::fstream::out},
				fLandmarks{name + "_landmarks.csv", std::fstream::out},
				fPose{name + "_pose.csv", std::fstream::out};

			frame.writeFeatures(fFeatures);
			frame.writeCovisibility(fCovis);
			frame.writeLandmarks(fLandmarks);
			frame.writePose(fPose);
		}

		std::fstream fLandmarks{"landmarks.csv", std::fstream::out};
		pointCloud.write(fLandmarks);

		std::cout << comm_sz << ", " << cvThreads << ", "
			<< cvTickToMs(startTime, loadDone) << ", "
			<< cvTickToMs(loadDone, loadTransferDone) << ", "
			<< cvTickToMs(loadTransferDone, featureDone) << ", "
			<< cvTickToMs(featureDone, featureTransferDone) << ", "
			<< cvTickToMs(featureTransferDone, covisDone) << ", "
			<< cvTickToMs(covisDone, covisTransferDone) << ", "
			<< cvTickToMs(covisTransferDone, motionDone) << std::endl;
	}

	MPI_Barrier(MPI_COMM_WORLD);

  return 0;
}
