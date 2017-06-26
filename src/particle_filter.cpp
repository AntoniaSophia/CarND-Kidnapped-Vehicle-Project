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
  num_particles = 150;

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
  // Find the predicted measurement which is closest
  // to each observed measurement and assign the
  // observed measurement to this particular landmark.


  // 1.Step: iterate through the observations
  for (int i = 0; i < observations.size(); i++) {
    LandmarkObs nextObservation = observations[i];

    // 2.Step: initializations
    // Start with minimum distance of INFINITY (it can only get closer... ;-))
    double minimumDistance = INFINITY;

    // Start with invalid id of landmark from map placeholder
    // (that means: no match found yet)
    int mapId = -1;

    // 3.Step: now step over the predictions and
    // try to find the nearest neighbour
    for (int j = 0; j < predicted.size(); j++) {
      LandmarkObs nextPrediction = predicted[j];

      // get Euklidian distance between current/predicted landmarks
      double currentDistance = dist(nextObservation.x, nextObservation.y,
                             nextPrediction.x,  nextPrediction.y);

      // find the predicted landmark nearest the current observed landmark
      if (currentDistance < minimumDistance) {   // Bingo, a new nearest neighbour was found!
        minimumDistance = currentDistance;
        mapId = nextPrediction.id;
      }
    }

    // 4.Step: finally set the observation id to the nearest neighbour
    //        predicted landmark's id
    observations[i].id = mapId;
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
  for (int index_p = 0; index_p < num_particles; index_p++) {
    Particle nextParticle = particles[index_p];

    // 2.Step: filter all prediction landmarks within sensor range
    // keep in mind: nearest neighbour is calculated by function dataAssociation

    // create a vector which keeps the landmark locations within sensor range
    vector<LandmarkObs> predictions;

    // go through all landmarks and check the distance
    for (Map::single_landmark_s nextLandmark : map_landmarks.landmark_list) {
      double distance = dist(nextLandmark.x_f, nextLandmark.y_f,
                             nextParticle.x,   nextParticle.y);
      // only use those landmarks as predictions which are within sensor range
      if (distance < sensor_range) {
        predictions.push_back(LandmarkObs{ nextLandmark.id_i, nextLandmark.x_f, nextLandmark.y_f });
      }
    }

    // 3.Step: convert the observation coordinates from local vehicle to
    //         global map coordinates

    // create a vector which keeps the transformed observations
    vector<LandmarkObs> transformedObservations;

    for (unsigned int j = 0; j < observations.size(); j++) {
      transformedObservations.push_back(
                calculateLocalToGlobal(observations[j], nextParticle));
    }

    // 4.Step: perform dataAssociation for the predictions and
    //         transformed observations on current particle
    dataAssociation(predictions, transformedObservations);

    // 5.Step: reinitialize the weight for this particle with 1
    particles[index_p].weight = 1.0;


    // 6.Step: Calculate the weight by using the Multimodal Gaussian formula

    // iterate through all observations in global coordinates
    // --> these you can compare with the real landmark map coordinates
    for (unsigned int j = 0; j < transformedObservations.size(); j++) {
      // now find the corresponding prediction to this observation
      LandmarkObs correspondingPrediction;
      for (unsigned int k = 0; k < predictions.size(); k++) {
        if (predictions[k].id == transformedObservations[j].id) {  // Bingo!!
          correspondingPrediction = predictions[k];
        }
      }

      // finally: product of this observation weight
      // with total observations weight
      particles[index_p].weight *= gaussProbability(correspondingPrediction,
                                                    transformedObservations[j],
                                                    std_landmark);
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
  // Resample particles with replacement with
  // probability proportional to their weight.
  vector<Particle> newParticles;

  // get all of the current weights
  vector<double> weights;
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }

  // generate random starting index for resampling wheel
  uniform_int_distribution<int> uniintdist(0, num_particles-1);
  int index = uniintdist(randomGenerator);

  // get maximum weight
  double maximumWeight = *max_element(weights.begin(), weights.end());

  // uniform random distribution [0.0, max_weight)
  uniform_real_distribution<double> unirealdist(0.0, maximumWeight);

  double beta = 0.0;

  // Reusing the resample wheel from Sebastian
  for (int i = 0; i < num_particles; i++) {
    beta += unirealdist(randomGenerator) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    newParticles.push_back(particles[index]);
  }

  particles = newParticles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
  // particle: the particle to assign each listed association,
  // and association's (x,y) world coordinates mapping to

  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  // Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = associations;
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

LandmarkObs ParticleFilter::calculateLocalToGlobal(const LandmarkObs& obs,
                                            const Particle& p) {
  LandmarkObs returnValue;

  returnValue.x = p.x + obs.x * cos(p.theta) - obs.y * sin(p.theta);
  returnValue.y = p.y + obs.x * sin(p.theta) + obs.y * cos(p.theta);
  returnValue.id = obs.id;

  return returnValue;
}
