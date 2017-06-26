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
#include "map.h"
#include "particle_filter.h"
#include "helper_functions.h"

using std::normal_distribution;
using std::default_random_engine;

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // 1.Step: Set the number of particles
  num_particles = 200;

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
  double px_p;
  double py_p;
  double pyaw_p;

  // take care about risk of "division by zero"
  if (std::fabs(yawd) > 0.01) {
    px_p = p_x + v/yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
    py_p = p_y + v/yawd * (cos(yaw) - cos(yaw+yawd*delta_t) );
  } else {
    px_p = p_x + v*delta_t*cos(yaw);
    py_p = p_y + v*delta_t*sin(yaw);
  }

  // Yaw angle
  pyaw_p = yaw + yawd*delta_t;

  // Add Gaussian noise
  px_p = px_p + noise_x(randomGenerator);
  py_p = py_p + noise_y(randomGenerator);
  pyaw_p = pyaw_p + noise_theta(randomGenerator);

  // return predicted state variables
  return {px_p, py_p, pyaw_p};
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs>& observations) {

 for (unsigned int i = 0; i < observations.size(); i++) {
    // Find the predicted measurement that is closest to each observed measurement
    // and assign the observed measurement to this particular landmark.


    // grab current observation
    LandmarkObs o = observations[i];

    // init minimum distance to maximum possible
    double min_dist = numeric_limits<double>::max();

    // init id of landmark from map placeholder to be associated with the observation
    int map_id = -1;
    
    for (unsigned int j = 0; j < predicted.size(); j++) {
      // grab current prediction
      LandmarkObs p = predicted[j];
      
      // get distance between current/predicted landmarks
      double cur_dist = dist(o.x, o.y, p.x, p.y);

      // find the predicted landmark nearest the current observed landmark
      if (cur_dist < min_dist) {
        min_dist = cur_dist;
        map_id = p.id;
      }
    }

    // set the observation's id to the nearest predicted landmark's id
    observations[i].id = map_id;
  }

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

  // 1. Step: iterate through all particles 
 
  for (int i = 0; i < num_particles; i++) {

    // get the particle x, y coordinates
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;

    // create a vector to hold the map landmark locations predicted to be within sensor range of the particle
    vector<LandmarkObs> predictions;

    // for each map landmark...
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {

      // get id and x,y coordinates
      float lm_x = map_landmarks.landmark_list[j].x_f;
      float lm_y = map_landmarks.landmark_list[j].y_f;
      int lm_id = map_landmarks.landmark_list[j].id_i;
      
      // only consider landmarks within sensor range of the particle (rather than using the "dist" method considering a circular 
      // region around the particle, this considers a rectangular region but is computationally faster)
      if (fabs(lm_x - p_x) <= sensor_range && fabs(lm_y - p_y) <= sensor_range) {

        // add prediction to vector
        predictions.push_back(LandmarkObs{ lm_id, lm_x, lm_y });
      }
    }

    // create and populate a copy of the list of observations transformed from vehicle coordinates to map coordinates
    vector<LandmarkObs> transformed_os;
    for (unsigned int j = 0; j < observations.size(); j++) {
      double t_x = cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y + p_x;
      double t_y = sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y + p_y;
      transformed_os.push_back(LandmarkObs{ observations[j].id, t_x, t_y });
    }

    // perform dataAssociation for the predictions and transformed observations on current particle
    dataAssociation(predictions, transformed_os);

    // reinit weight
    particles[i].weight = 1.0;

    for (unsigned int j = 0; j < transformed_os.size(); j++) {
      
      // placeholders for observation and associated prediction coordinates
      double o_x, o_y, pr_x, pr_y;
      o_x = transformed_os[j].x;
      o_y = transformed_os[j].y;

      int associated_prediction = transformed_os[j].id;
      LandmarkObs a;
      // get the x,y coordinates of the prediction associated with the current observation
      for (unsigned int k = 0; k < predictions.size(); k++) {
        if (predictions[k].id == associated_prediction) {
          pr_x = predictions[k].x;
          pr_y = predictions[k].y;
          a = predictions[k];
        }
      }

      // product of this observation weight with total observations weight
      particles[i].weight *= gaussProbability(a, transformed_os[j],std_landmark);
    }
  }      
  
}

double ParticleFilter::gaussProbability(const LandmarkObs& obs,
                                        const LandmarkObs &lm,
                                        const double sigma[]) {
  double cov_x = sigma[0]*sigma[0];
  double cov_y = sigma[1]*sigma[1];
  double normalizer = 2.0*M_PI*sigma[0]*sigma[1];
  double dx = (obs.x - lm.x);
  double dy = (obs.y - lm.y);

  return exp(-(dx*dx/(2*cov_x) + dy*dy/(2*cov_y)))/normalizer;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  vector<Particle> new_particles;

  // get all of the current weights
  vector<double> weights;
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }

  // generate random starting index for resampling wheel
  uniform_int_distribution<int> uniintdist(0, num_particles-1);
  auto index = uniintdist(randomGenerator);

  // get max weight
  double max_weight = *max_element(weights.begin(), weights.end());

  // uniform random distribution [0.0, max_weight)
  uniform_real_distribution<double> unirealdist(0.0, max_weight);

  double beta = 0.0;

  // spin the resample wheel!
  for (int i = 0; i < num_particles; i++) {
    beta += unirealdist(randomGenerator) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }

  particles = new_particles;

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

void ParticleFilter::calculateLocalToGlobal(LandmarkObs& obs,
                                            const Particle& p) {
  obs.x_global = p.x + obs.x * cos(p.theta) - obs.y * sin(p.theta);
  obs.y_global = p.y + obs.x * sin(p.theta) + obs.y * cos(p.theta);
}
