#include "TaskConfig.h"
#include "Tracker/ITrackerWrapper.h"
#include "ResultMonitor.h"
#include "TrajectoriesRefiner.h"

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <fstream>


using namespace std;
using namespace cv;


int main(int argc, char* argv[])
{
	// Task Setting
	MtsConfig conf;
	if (argc >= 11)
	{
		conf.SetSeq(argv);		
	}
	if (argc >= 14)
	{
		conf.SetSnapshotInterval(stoi(argv[12]));
        conf.SetSnapshotN(stoi(argv[13]));
	}
	if (argc >= 18)
	{
		conf.SetEpsilon(stoi(argv[14]));
		conf.SetZeta(stof(argv[15]));
		conf.SetDelta(stof(argv[16]));
		conf.SetDeltaDuration(stoi(argv[17]));
	}
	if (argc >= 19)
	{
		conf.SetPreservedOne(stoi(argv[18]) == 1);
	}

	// Initialize
	// Container in a snapshot interval
	vector<ITrackerWrapper*> trkList;	// Store trackers
	vector<RectVec> trjList;			// Store trajectories of multiple trackers
	vector<Mat> frmList;				// Store frames

	// Container in the whole tracking
	RectVec finalTrj(conf.SeqEndFrame - conf.SeqStartFrame + 1);
	ResultMonitor monitor;

	RectVec resList;                    // Results of trackers in current frame

	// Single tracker mode don't need, multiple trackers mode need in fist frame
	bool toCloneTrk = conf.GetSnapshotN()>0;

	bool toRefine = false;									// Time to launch refiner

	// Start tracking
	for (int frmId = conf.SeqStartFrame, _frmId=1; frmId <= conf.SeqEndFrame; ++frmId,++_frmId)
	{
		cout << "Tracking Frame: " << frmId << endl;
		Mat frm = imread(conf.GetFramePath(frmId),1);		// Current frame image
		frmList.push_back(frm);

		if (frmId == conf.SeqStartFrame)
		{
			//First frame
			IntRect& bb = conf.SeqBbox;

            trkList=conf.GetNewTrackers();
//			ITrackerWrapper* newTrk = conf.GetNewTracker();

			for(ITrackerWrapper* trk:trkList) {
				trk->Init(frm, bb);

                RectVec newTrj;
                resList.push_back(bb);
                newTrj.push_back(bb);
                trjList.push_back(newTrj);
			}
//          newTrk->Init(frm, bb);
//			trkList.push_back(newTrk);
			if (conf.GetSnapshotN() == 0 && conf.GetTpN() == 1) {
				finalTrj[_frmId - 1] = resList.front();
			}
		}
		else
		{
			// Tracking all tracker without update
			#pragma omp parallel for
			for (int trkId = 0; trkId < trkList.size(); trkId++)
			{
				IntRect& bb=resList.at(trkId);
				bb = trkList[trkId]->TrackWithoutUpdate(frm);
				trjList[trkId].push_back(bb);
			}

			// Update latest tracker
			for (int trkOff = conf.GetTpN(); trkOff>0; trkOff--) {
                int trkId=trkList.size()-trkOff;
                trkList[trkId]->Update();
            }

			// Single tracker to skip
			if (conf.GetSnapshotN() == 0 && conf.GetTpN() == 1) {
				// TODO: Make it more scientific.
				finalTrj[_frmId-1] = resList.front();
				if (conf.EnableMonitor)
				{
					monitor.Display(frm, resList);
				}
				continue;
			}
			size_t trjLen;
			trjLen = conf.GetSnapshotInterval()* cv::max(size_t(1),conf.GetSnapshotN());

			toRefine = _frmId % trjLen == 0 || frmId == conf.SeqEndFrame;
			toCloneTrk = _frmId % conf.GetSnapshotInterval() == 0;

			// --------------------------------------------------------------------
			// Refine trajectories when amount of trackers reaches target
			if (toRefine)
			{
				cout << "Track back." << endl;
				TrajectoriesRefiner refiner(
					conf.GetEpsilon(),
					conf.GetZeta(),
					conf.GetDelta(),
					conf.GetDeltaDuration(),
                    &(conf._GetNewTrackers)
                );

				vector<RectVec> trjBkList = refiner.TrackBack(trkList, trjList, frmList);
                int bestTrjId;
                vector<int> bestTrjIds = refiner.SelectTrajectories(trjList, trjBkList, frmList, conf.GetTpN(),
                                                                    bestTrjId);
				RectVec& bestTrj = trjList[bestTrjId];
				ITrackerWrapper* bestTrk = trkList[bestTrjId];

				// Copy selected trajectory to final trajectory
				size_t startFrmId = _frmId - bestTrj.size();
				for (size_t id = 0; id < bestTrj.size(); id++)
				{
					finalTrj[startFrmId + id] = bestTrj[id];
				}

				// Clean tracker list and re-initialize trackers for a new round
                vector<ITrackerWrapper*> newTrkList;
                for (size_t id = 0; id < bestTrjIds.size(); id++) {
                    newTrkList.push_back(trkList[bestTrjIds[id]]);
                }
				for (size_t trkId = 0; trkId < trkList.size(); trkId++)
				{
                    if(std::find(bestTrjIds.begin(), bestTrjIds.end(), trkId) == bestTrjIds.end()) {
                        delete trkList[trkId];
                    }
				}
                trkList.clear();
				trkList=newTrkList;
                vector<ITrackerWrapper*> seedTrkList=conf.GetNewTrackers();
                RectVec newResList;
                newResList.insert(newResList.end(),trkList.size(),IntRect());
                for (size_t trkId = 0; trkId < trkList.size(); trkId++)
                {
                    float overlapRatio=resList[bestTrjId].Overlap(resList[bestTrjIds[trkId]]);
                    if (overlapRatio<0.8)
                    {
                        delete trkList[trkId];
                        trkList[trkId]=seedTrkList[trkId];
                        trkList[trkId]->Init(frm,bestTrj.back());
                        newResList[trkId]=bestTrj.back();
                    }
                    else {
                        newResList[trkId]=resList[bestTrjIds[trkId]];
                    }
                }
                resList=newResList;

				RectVec newTrj;
				trjList.clear();
                trjList.insert(trjList.end(), conf.GetTpN(), newTrj);

				// Clean frame list of this round
				frmList.clear();
			}

		}

		if (toCloneTrk)
		{
            int trkN=trkList.size();
            for(size_t trkOff=conf.GetTpN();trkOff>0;trkOff--) {
                int trkId=trkN-trkOff;
                trkList.push_back(trkList[trkId]->Clone());
                trjList.push_back(trjList[trkId]);
                resList.push_back(resList[trkId]);
            }
//			trkList.push_back(trkList.back()->clone());
//			trjList.push_back(trjList.back());
//			resList.push_back(resList.back());

			toCloneTrk = false;
		}

		if (conf.EnableMonitor)
		{
			monitor.Display(frm, resList);
		}

	}

	//Saving Result
	ofstream outFile;
	outFile.open(conf.GetResultOutputPath(), ios::out);
	for (auto t : finalTrj)
	{
		outFile << t.XMin() << "," << t.YMin() << "," << t.Width() << "," << t.Height() << endl;
	}
	if (outFile.is_open())
	{
		outFile.close();
	}
//	system("pause");
	return EXIT_SUCCESS;
}