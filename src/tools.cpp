#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

// create epsilon to avoid small values
const float EPSYLON = 0.0001;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	VectorXd rmse(4);
	rmse << 0,0,0,0;

	if(estimations.size() == 0)
	{
		cout<<"Estimations vector has size 0"<<endl;
		return rmse;
	}
	else if(estimations.size() != ground_truth.size())
	{
		cout<<"Estimations vector has different size than ground truth"<<endl;
		return rmse;
	}

	//accumulate squared residuals
	int acc_res = 0;
	VectorXd aux_vec;
	for(int i=0; i < estimations.size(); ++i){
		aux_vec = estimations[i] - ground_truth[i];
		aux_vec = aux_vec.array() * aux_vec.array();
		rmse += aux_vec;
	}

	//calculate the mean
	rmse = rmse/estimations.size();

	//calculate the squared root
	rmse = rmse.array().sqrt();

	//return the result
	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

  MatrixXd Hj(3,4);
  //recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  //pre-compute a set of terms to avoid repeated calculation
  float c1 = px*px+py*py;
  float c2 = sqrt(c1);
  float c3 = (c1*c2);

  //check division by zero
  if(fabs(c1) < EPSYLON){
    cout << "CalculateJacobian () - Error - Division by Zero" << endl;
    return Hj;
  }

  //compute the Jacobian matrix
  Hj << (px/c2), (py/c2), 0, 0,
      -(py/c1), (px/c1), 0, 0,
      py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

  return Hj;

}
