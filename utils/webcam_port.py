#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 20:28:03 2023

@author: ejjchl77
"""

'''
// importance : ☆☆☆☆☆
// title : 과제5
//         카메라를 control하기 위한 class를 선언하고 다음 기능을 구현하시오
//         1. 자동 카메라 연결(카메라의 개수를 입력 받음, (default = 1)
//	       2. 연결된 카메라의 이미지 반환
// date  : 2020.02.05
// writer : jeong
// result : 카메라가 연결된 갯수만큼 창을 띄워서 동시에 카메라 영상을 틀어줌. 
'''

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


class ControlCam {

protected:
	int CamNum = 0, flag = 0;
	VideoCapture* cap = new VideoCapture[6];
	Mat* image = new Mat[6];
	string Name[6] = { "camera 0","camera 1","camera 2", "camera 3", "camera 4", "camera 5" };
public:

	void CheckCamPort() {// 카메라 열렸는지 확인하여 VideoCapture객체 만들어주는 메서드
		int startflag = 0;
		for (int i = 0; i < 6; i++) { // 5번포트까지 확인
			cap[i] = VideoCapture(i);
			if (cap[i].isOpened()) {
				cout << "Opened Cam Port is : " << i << '\n';
				CamNum++; // 연결된 카메라의 갯수 카운팅
				startflag = 1;
			}
		}
		if (startflag == 0) { // 연결 카메라가 없을경우
			cout << "No camera connecetd" << '\n';
			return;
		}
	}

	void RunCam() {
		for (int i = 0; i < 6; i++) {
			if (cap[i].isOpened()) {
				cap[i].read(image[i]);
				imshow(Name[i], image[i]);
			}
		}
		waitKey(1);
	}

	// 생성자
	ControlCam() {
		CheckCamPort();
		cout << "=== 생성자 호출 : 연결된 카메라 대수 : " << CamNum << "===\n";
	}

	// 소멸자 -> 동적할당 해제용
	~ControlCam() {
		delete cap;
	}
};


int main() {

	ControlCam ctrl;

	while (1) {
		ctrl.RunCam();
	}
	return 0;
}