#ifndef NEURON_HPP
#define NEURON_HPP

#include <iostream>

class Neuron
{
public:
	Neuron(int num_prev_neurons=0) { init_neuron(num_prev_neurons); }
	~Neuron();

	float output(float input[], int n);

protected:
	float _o;	///< output
	float* _w;	///< array of weights

	static float lambda;
	static float WEIGHT_LOW;
	static float WEIGHT_HIGH;

	void init_neuron(int num_prev_neurons=0);
	static float random_weight(float low=WEIGHT_LOW, float high=WEIGHT_HIGH);
	static float sigmoid(float x);
};

#endif /*!NEURON_HPP*/
