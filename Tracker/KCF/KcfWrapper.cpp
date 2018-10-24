#include "KcfWrapper.h"
#include "kcftracker.hpp"


KcfWrapper::KcfWrapper()
{
	
}

KcfWrapper::~KcfWrapper()
{
	if(Core!= nullptr) {
		delete Core;
	}
}

void KcfWrapper::Init(const cv::Mat &Frame, const IntRect &BBox) {

	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;

	Core=new kcf::KCFTracker(HOG, FIXEDWINDOW, MULTISCALE, LAB, HIS);
	Core->init(cv::Rect(BBox.XMin(), BBox.YMin(), BBox.Width(), BBox.Height()), Frame);
}

IntRect KcfWrapper::Track(const cv::Mat& Frame)
{
	IntRect bb = TrackWithoutUpdate(Frame);
	Update();
	cv::Rect bb_ = ResultPostProc(bb.ToCvRect(), Frame);
	return bb_;
}

IntRect KcfWrapper::TrackWithoutUpdate(const cv::Mat& frame)
{
	Frame = frame;
	cv::Rect bb = Core->trackWithoutUpdate(frame);
    cv::Rect bb_ = ResultPostProc(bb, Frame);
	return bb_;
}

void KcfWrapper::Update()
{
	Core->train(Frame);
}

ITrackerWrapper* KcfWrapper::Clone()
{
	KcfWrapper* mirror = new KcfWrapper;
	*mirror=*this;
	mirror->Core = new kcf::KCFTracker(*this->Core);
	return mirror;
}

void KcfWrapper::ConfigFeature(bool hog, bool lab, bool his) {
	HOG = hog;
	LAB = lab;
	HIS = his;
}

cv::Rect KcfWrapper::ResultPostProc(cv::Rect res, const cv::Mat &frm) {
    res.x = std::max(0, res.x);
    res.y = std::max(0, res.y);

    res.x = std::min(frm.cols - 8, res.x);
    res.y = std::min(frm.rows - 8, res.y);

    res.width = std::max(res.width, 8);
    res.height = std::max(res.height, 8);

    res.width = std::min(res.width, frm.cols);
    res.height = std::min(res.height, frm.rows);

    if(res.x + res.width > frm.cols) {
        res.x = frm.cols - res.width;
    }
    if(res.y + res.height > frm.rows) {
        res.y = frm.rows - res.height;
    }
    return res;
}
