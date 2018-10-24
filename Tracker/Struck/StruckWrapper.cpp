#include "StruckWrapper.h"
#include "Config.h"
#include "Tracker.h"

#include <opencv/cv.h>
#include <opencv/highgui.h>
using namespace std;

StruckWrapper::StruckWrapper()
{
	
}


StruckWrapper::~StruckWrapper()
{
	if(Core!= nullptr) {
		delete Core;
	}
}

void StruckWrapper::Init(const cv::Mat &frame, const IntRect &bbox)
{
	Config *conf = new Config("struck_config.txt");
	srand(conf->seed);
	conf->frameWidth = frame.cols;
	conf->frameHeight = frame.rows;
	Core = new Tracker(*conf);
	Core->Initialise(frame, bbox);
}

IntRect StruckWrapper::Track(const cv::Mat& frame)
{
	TrackWithoutUpdate(frame);
	Update();
	return Core->GetBB();
}

IntRect StruckWrapper::TrackWithoutUpdate(const cv::Mat& frame)
{
	UpdateInfo ui = Core->TrackWithoutUpdate(frame);
	NeedUpdate = ui.NeedUpdate;
	if (NeedUpdate)
	{
		UpdateImage = ui.UpdateImage;
	}
	return Core->GetBB();
}

void StruckWrapper::Update()
{
	if (NeedUpdate)
	{
//		cout << "Updating: " << Name << endl;
		Core->UpdateLearner(*UpdateImage);
		delete UpdateImage;
	}
	
}

ITrackerWrapper* StruckWrapper::Clone()
{
	StruckWrapper *mirror = new StruckWrapper(*this);
	mirror->Core=new Tracker(*this->Core);
	return mirror;
}
