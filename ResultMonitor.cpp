#include "ResultMonitor.h"

#include <opencv2/core/core_c.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

Mat ResultMonitor::Display(const Mat &frame, const IntRect &bbox)
{
	RectVec r;
	r.push_back(bbox);
	return Display(frame, r);
}

Mat ResultMonitor::Display(const Mat &frame, const RectVec &rPool)
{
	Mat result;
	if (frame.type() == CV_8UC1)
	{
		cvtColor(frame, result, CV_GRAY2RGB);
	}
	else
	{
		frame.copyTo(result);
	}
	for (size_t i = 0; i < rPool.size(); ++i) {
		if (i < Colors.size()) {
			rectangle(result, rPool[i], Colors[i]);
		}
		else
		{
			rectangle(result, rPool[i], CV_RGB(0, 0, 0));
		}
//		color = CV_RGB(rand() % 256, rand() % 256, rand() % 256);
	}
	imshow(WindowName, result);
	waitKey(1);
	return result;
}

void ResultMonitor::rectangle(Mat& rMat, const IntRect& rRect, const Scalar& rColour) const
{
	IntRect r(rRect);
	cv::rectangle(rMat, Point(r.XMin(), r.YMin()), Point(r.XMax(), r.YMax()), rColour, thickness);
}
