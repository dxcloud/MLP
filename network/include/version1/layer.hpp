#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>
#include <iostream>

#include "neuron.hpp"

class Layer{
public:
	Layer(int num_neurons);
	~Layer();

	void add_neuron(Neuron* neuron);
	void compute_output(float input[]);
	float* get_output() const { return _output; }
	void set_output(float o[]);

protected:

	int _num_neurons;
	float * _output;
	std::vector<Neuron*> _neurons;
};

#endif /*!LAYER_HPP*/
