#pragma once
#include "Rect.h"
#include "Tracker/ITrackerWrapper.h"

using namespace std;
using namespace cv;

class TrajectoriesRefiner
{
public:
	TrajectoriesRefiner(int epsilon, float zeta, float delta, int deltaDuration, ITrackerWrapper* (*getNewTracker)(), bool display = false)
		: Epsilon(epsilon),
		Zeta(zeta),
		Delta(delta),
		DeltaDuration(deltaDuration),
		Display(display), 
		GetNewTracker(getNewTracker)
	{}
    TrajectoriesRefiner(int epsilon, float zeta, float delta, int deltaDuration, vector<ITrackerWrapper*> (*getNewTrackers)(), bool display = false)
            : Epsilon(epsilon),
              Zeta(zeta),
              Delta(delta),
              DeltaDuration(deltaDuration),
              Display(display),
              GetNewTrackers(getNewTrackers)
    {}
    vector<RectVec> TrackBack(
		const vector<ITrackerWrapper*>& trkList,
		const vector<RectVec>& trjList,
		const vector<Mat>& frmList
	) const;
    vector<float> CalTrajectoryScores(
            const vector<RectVec> &trjList,
            const vector<RectVec> &trjBkList,
            const vector<Mat> &frmList
    ) const;
	int SelectTrajectory(
		const vector<RectVec>& trjList,
		const vector<RectVec>& trjBkList,
		const vector<Mat>& frmList
	) const;
	vector<int> SelectTrajectories(
            const vector<RectVec> &trjList,
            const vector<RectVec> &trjBkList,
            const vector<Mat> &frmList,
            const size_t tpN,
            int &bestTrjId
    ) const;
	float CalImgDiff(
		const IntRect& inRect, 
		const vector<Mat>& vecFrgImg, 
		const Mat& Images
	) const;
	Mat GetImgDiffMask(int nHeight, int nWidth) const;
	void TimeDomainPattern(
		vector<float>& TrajectoryDist, 
		const vector<vector<float>>& vecOvl,
		const int& frameInterval
	) const;
	bool CheckFrameOcclusion(
		std::vector<std::vector<float>> vecBackCost, 
		size_t trackerAmount, 
		size_t interval
	) const;

	int Epsilon;
	float Zeta;
	float Delta;
	int DeltaDuration;
	bool Display;

    ITrackerWrapper* (*GetNewTracker)();
    vector<ITrackerWrapper*> (*GetNewTrackers)();
};

