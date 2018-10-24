#ifndef StructuredSVM_H
#define StructuredSVM_H

/* 
 * Struck: Structured Output Tracking with Kernels
 * 
 * Code to accompany the paper:
 *   Struck: Structured Output Tracking with Kernels
 *   Sam Hare, Amir Saffari, Philip H. S. Torr
 *   International Conference on Computer Vision (ICCV), 2011
 * 
 * Copyright (C) 2011 Sam Hare, Oxford Brookes University, Oxford, UK
 * 
 * This file is part of Struck.
 * 
 * Struck is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Struck is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with Struck.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#include <vector>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include "Rect.h"
namespace sowp
{ 
struct SupportPattern
{
	std::vector<Eigen::VectorXd> x;
	std::vector<Rect> yv;
  std::vector<cv::Mat> img;
	int y;
	int refCount;
};

struct SupportVector
{
	SupportPattern* x;  
	int y;
	double b;
	double g;
};

class StructuredSVM
{
public:
	StructuredSVM(const StructuredSVM& t_Other)
		: ITER(t_Other.ITER),
		  C(t_Other.C),
		  MAX_BUDGET(t_Other.MAX_BUDGET),
		  K(t_Other.K),
		  W(t_Other.W)
	{
		for (SupportPattern* t : t_Other.sps)
		{
			SupportPattern *mirrorT = new SupportPattern(*t);
			sps.push_back(mirrorT);
		}
		for (SupportVector* t : t_Other.svs)
		{
			SupportVector *mirrorT = new SupportVector(*t);
			for (size_t i = 0; i < t_Other.sps.size(); ++i)
			{
				if (t->x == t_Other.sps[i])
				{
					mirrorT->x = sps[i];
				}
			}
			svs.push_back(mirrorT);
		}
	}


private:
	const int ITER = 10;
  const double C = 10;
  const int MAX_BUDGET = 100;
public:
  StructuredSVM();
  ~StructuredSVM();

  double test(const Eigen::VectorXd &feature);
  double validation_test(const Eigen::VectorXd &feature);
  void train(const std::vector<Rect> &samples, const std::vector<Eigen::VectorXd> &features, int y);

private:
  Eigen::MatrixXd K;
  Eigen::VectorXd W;

  std::vector<SupportPattern*> sps;
  std::vector<SupportVector*> svs; 
  
	inline double Loss(const Rect& y1, const Rect& y2) const
	{
    return 1.0-y1.overlap_ratio(y2);
	}

	double ComputeDual() const;
	void SMOStep(int ipos, int ineg);	
  std::pair<int, double> MinGradient(int ind);

	void ProcessNew(int ind);
	void Reprocess();
	void ProcessOld();
	void Optimize();

	int AddSupportVector(SupportPattern* x, int y, double g);
	void RemoveSupportVector(int ind);
	void RemoveSupportVectors(int ind1, int ind2);
	void SwapSupportVectors(int ind1, int ind2);
	
	void BudgetMaintenance();
	void BudgetMaintenanceRemove();

  double evaluate(const Eigen::VectorXd& x) const;
  inline double compute_kernel_score(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const
  {
    return x1.dot(x2);
  };
  inline double compute_kernel_score(const Eigen::VectorXd& x) const
  {
    return x.squaredNorm();
  };
};
}
#endif