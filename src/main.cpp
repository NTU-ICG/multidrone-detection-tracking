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
using namespace cv;




int main(){
	// calculate every person's (id,(up_num,down_num,average_x,average_y))
	map<int,vector<int>> personstate;
	map<int,int> classidmap;
	bool is_first = true;
	// char yolo_engine[]	 = "../resources/yolov5s.engine";
	char yolo_engine[]	 = "../resources/best.engine";
	char sort_engine[] = "../resources/deepsort.engine";
	float conf_thre = 0.4;
	Trtyolosort yosort(yolo_engine,sort_engine);	
	VideoCapture capture;
	cv::Mat frame;
	// frame = capture.open("../drone_resources/2021-09-01_15.41.54.mp4");
	// frame = capture.open("../drone_resources/20211028_121335.mp4");
	// frame = capture.open("../drone_resources/test.mp4");
	frame = capture.open(0);
	if (!capture.isOpened()){
		std::cout<<"can not open"<<std::endl;
		return -1 ;
	}
	std::cout << "open succeed!\n" << std::endl;

	capture.read(frame);
	std::cout<<"read_frame\n";
	std::vector<DetectBox> det;
	auto start_draw_time = std::chrono::system_clock::now();
	
	clock_t start_draw,end_draw;
	start_draw = clock();
	int i = 0;
	while(capture.read(frame)){
		if (i%1==0){
		cv::resize(frame, frame, cv::Size(640, 640));
		std::cout<<"origin img size:"<<frame.cols<<" "<<frame.rows<<std::endl;
		auto start = std::chrono::system_clock::now();
		// capture >> frame;
		// imshow("frame", frame);
		yosort.TrtDetect(frame,conf_thre,det);

		auto end = std::chrono::system_clock::now();
		int delay_infer = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout  << "delay_infer:" << delay_infer << "ms" << std::endl;
		// if( waitKey(1) == 27 ) break;
		}
		i++;
	}
	capture.release();
	return 0;
	
}
