#include "CtWrapper.h"
#include "CompressiveTracker.h"


CtWrapper::CtWrapper() {}


CtWrapper::~CtWrapper()
{
	if(Core!= nullptr) {
		delete Core;
	}
}

void CtWrapper::Init(const cv::Mat &frame, const IntRect &bbox) {
	cvtColor(frame, Frame, CV_RGB2GRAY);
	Core=new CompressiveTracker();
	cv::Rect r = bbox.ToCvRect();
	Core->init(Frame, r);
	Result = r;
}

IntRect CtWrapper::Track(const cv::Mat& frame)
{
	cvtColor(frame, Frame, CV_RGB2GRAY);
	Core->processFrame(Frame, Result, false);
	Core->update(Frame, Result);
	return Result;
}

IntRect CtWrapper::TrackWithoutUpdate(const cv::Mat& frame)
{
    cvtColor(frame, Frame, CV_RGB2GRAY);
    Core->processFrame(Frame, Result, false);
    return Result;
}

void CtWrapper::Update()
{
	Core->update(Frame, Result);
}

ITrackerWrapper* CtWrapper::Clone()
{
	CtWrapper* mirror = new CtWrapper;
	mirror->Frame = Frame.clone();
	mirror->Result = Result;
	mirror->Core = new CompressiveTracker(*this->Core);
	return mirror;
}

