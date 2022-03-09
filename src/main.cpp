#include<iostream>
#include "manager.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <map>
#include <cmath>
#include <time.h>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <darknet_ros_msgs/BoundingBox.h>
#include <darknet_ros_msgs/BoundingBoxes.h>
using namespace cv;


class ImageConverter
{
  ros::NodeHandle nodeHandle_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  ros::Publisher boundingBoxesPublisher_;
  darknet_ros_msgs::BoundingBoxes boundingBoxesResults_;


  float conf_thre = 0.4;
  char* yolo_engine = NULL;
  char* sort_engine = NULL;

  std::string boundingBoxesTopicName;
  int boundingBoxesQueueSize;
  bool boundingBoxesLatch;

  Trtyolosort * yosort;

public:
  ImageConverter(ros::NodeHandle nh_)
    : nodeHandle_(nh_), it_(nodeHandle_)
  {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/stereo/left/image_color", 1,
      &ImageConverter::imageCb, this);


    nodeHandle_.param("publishers/bounding_boxes/topic", boundingBoxesTopicName, std::string("bounding_boxes"));
    nodeHandle_.param("publishers/bounding_boxes/queue_size", boundingBoxesQueueSize, 1);
    nodeHandle_.param("publishers/bounding_boxes/latch", boundingBoxesLatch, false);

    boundingBoxesPublisher_ = 
      nodeHandle_.advertise<darknet_ros_msgs::BoundingBoxes>(boundingBoxesTopicName, 
                                                             boundingBoxesQueueSize, 
                                                             boundingBoxesLatch);



	// yolo_engine = "../resources/best_colab.engine";
  yolo_engine = "../resources/best_6_0_s.engine";
	sort_engine = "../resources/deepsort.engine";
	
	yosort = new Trtyolosort(yolo_engine,sort_engine);

  }

  ~ImageConverter(){}

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

	
	cv::Mat frame = cv_ptr->image;
	std::vector<DetectBox> det;

	cv::resize(frame, frame, cv::Size(640, 640));
	yosort->TrtDetect(frame,conf_thre,det);


  for (DetectBox detectbox : det){

    darknet_ros_msgs::BoundingBox boundingBox;
    
    boundingBox.Class = detectbox.classID;
    boundingBox.id = detectbox.trackID;
    boundingBox.probability = detectbox.confidence;
    boundingBox.xmin = detectbox.x1;
    boundingBox.ymin = detectbox.y1;
    boundingBox.xmax = detectbox.x2;
    boundingBox.ymax = detectbox.y2;
    boundingBoxesResults_.bounding_boxes.push_back(boundingBox);
  }

  boundingBoxesResults_.header.stamp = ros::Time::now();
  boundingBoxesResults_.header.frame_id = "detection";
  // boundingBoxesResults_.image_header = headerBuff_[(buffIndex_ + 1) % 3];
  boundingBoxesPublisher_.publish(boundingBoxesResults_);

  boundingBoxesResults_.bounding_boxes.clear();



    // // Run Canny edge detector on image
    // cv::Mat src = cv_ptr->image;
    // cv::Mat dst;
    // cv::Canny( src, dst, 0, 0, 3 );

    // // Update GUI Window
    // cv::imshow("source", src);
    // cv::imshow("canny", dst);
    // cv::waitKey(3);

    // sensor_msgs::ImagePtr msg_out = cv_bridge::CvImage(std_msgs::Header(), "mono8", dst).toImageMsg();
    // Output modified video stream
    // image_pub_.publish(msg_out);
  }
};


int main(int argc, char** argv){

	ros::init(argc, argv, "image_converter");
  ros::NodeHandle nh("/detection");
	ImageConverter ic(nh);
	ros::spin();
	return 0;

	// char yolo_engine[]	 = "../resources/best_colab.engine";
  // char yolo_engine[]	 = "../resources/best_6_0_s.engine";
  
	// char sort_engine[] = "../resources/deepsort.engine";
	// float conf_thre = 0.4;
	// Trtyolosort yosort(yolo_engine,sort_engine);	
	// VideoCapture capture;
	// cv::Mat frame;

	// frame = capture.open("../resources/VID_20220207_161128 1.mp4");
	// // frame = capture.open(0);
	// if (!capture.isOpened()){
	// 	std::cout<<"can not open"<<std::endl;
	// 	return -1 ;
	// }
	// std::cout << "open succeed!\n" << std::endl;

	// capture.read(frame);
	// std::cout<<"read_frame\n";
	// std::vector<DetectBox> det;
	
	// int i = 0;
	// while(capture.read(frame)){
	// 	// if (i%1==0){
	// 	cv::resize(frame, frame, cv::Size(640, 640));
	// 	// std::cout<<"origin img size:"<<frame.cols<<" "<<frame.rows<<std::endl;
	// 	auto start = std::chrono::system_clock::now();

	// 	yosort.TrtDetect(frame,conf_thre,det);
	// 	auto end = std::chrono::system_clock::now();
	// 	int delay_infer = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	// 	// std::cout  << "delay_infer:" << delay_infer << "ms" << std::endl;
	// 	// if( waitKey(1) == 27 ) break;
	// 	// }
	// 	i++;
	// }
	// capture.release();
	// return 0;
	
}
