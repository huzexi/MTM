#pragma once
#include "Rect.h"
#include <string>
#include <opencv/cv.h>
using namespace cv;
using namespace std;

class ResultMonitor
{
public:
	ResultMonitor()
	{

	};
	~ResultMonitor(){};
	void SetFrame(Mat& frame){ Frame = frame; };
	void SetBbox(IntRect& bbox) { BBox = bbox; };
	void Display()
	{
		Display(Frame, BBox);
	};
	Mat Display(const Mat &frame, const IntRect &bbox);
	Mat Display(const Mat &frame, const RectVec &rPool);

	string GetWindowName() const
	{
		return WindowName;
	}

	void SetWindowName(const string& t_WindowName)
	{
		WindowName = t_WindowName;
	}

private:
	void rectangle(Mat& rMat, const IntRect& rRect, const Scalar& rColour) const;

	Mat Frame;
	IntRect BBox;
	int thickness=2;
	const vector<Scalar> Colors=vector<Scalar>{
		CV_RGB(255, 0, 0),
		CV_RGB(0, 255, 0),
		CV_RGB(0, 0, 255),
		CV_RGB(255, 255, 0),
		CV_RGB(255, 0, 255),
		CV_RGB(0, 255, 255),
		CV_RGB(126, 0, 0),
		CV_RGB(0, 126, 0),
		CV_RGB(0, 0, 126),
		CV_RGB(126, 126, 0),
		CV_RGB(126, 0, 126),
		CV_RGB(0, 126, 126),
	};

	string WindowName="Result";

};