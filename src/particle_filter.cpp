/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <math.h>
#include <random>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include "particle_filter.h"


using std::normal_distribution;
using std::default_random_engine;

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // 1.Step: Set the number of particles
  num_particles = 500;

  // 2.Step: add random Gaussian noise to each particle x/y/theta.
  normal_distribution<double> noise_x(x, std[0]);
  normal_distribution<double> noise_y(y, std[1]);
  normal_distribution<double> noise_theta(theta, std[2]);

  // 3.Step: now initialize the particles based on estimates of
  //         x, y, theta and their uncertainties from GPS
  for (unsigned i = 0; i < num_particles; i++) {
    Particle newParticle;
    newParticle.x = noise_x(randomGenerator);
    newParticle.y = noise_y(randomGenerator);
    newParticle.theta = noise_theta(randomGenerator);
    newParticle.id = i;
    newParticle.weight = 1.0f;
    particles.push_back(newParticle);
  }

  // 4. Step: Initialize all particles weights to 1
  weights.resize(num_particles, 1.0f);

  // 5.Step: Set initialized to True
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t,
                                double std_pos[],
                                double velocity,
                                double yaw_rate) {
  // Add measurements to each particle and add random Gaussian noise.

  for (int i = 0; i < particles.size(); i++) {
    Particle tempParticle = particles[i];
    std::vector<double> state_prediction = Bicycle_Model(delta_t,
                                              std_pos,
                                              tempParticle,
                                              velocity,
                                              yaw_rate);

    particles[i].x = state_prediction[0];  // x position
    particles[i].y = state_prediction[1];  // y position
    particles[i].theta = state_prediction[2];  // yaw rate
  }
}

std::vector<double> ParticleFilter::Bicycle_Model(double delta_t,
                              double std_pos[],
                              const Particle& p,
                              double velocity,
                              double yaw_rate) {

  normal_distribution<double> noise_x(0, std_pos[0]);
  normal_distribution<double> noise_y(0, std_pos[1]);
  normal_distribution<double> noise_theta(0, std_pos[2]);

  // Now just apply the bicycle motion equations from Lesson 12, chapter 3
  double p_x = p.x;
  double p_y = p.y;
  double yaw = p.theta;

  double v = velocity;
  double yawd = yaw_rate;

  // predicted state values to be returned at the end
  double px_p, py_p;

  // take care about risk of "division by zero"
  if (std::fabs(yawd) > 0.01) {
    px_p = p_x + v/yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
    py_p = p_y + v/yawd * (cos(yaw) - cos(yaw+yawd*delta_t) );
  } else {
    px_p = p_x + v*delta_t*cos(yaw);
    py_p = p_y + v*delta_t*sin(yaw);
  }

  // Yaw angle
  double pyaw_p = yaw + yawd*delta_t;

  // Add noise
  px_p = px_p + noise_x(randomGenerator);
  py_p = py_p + noise_y(randomGenerator);
  pyaw_p = pyaw_p + noise_theta(randomGenerator);

  // return predicted state
  return {px_p, py_p, pyaw_p};
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
