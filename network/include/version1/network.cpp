#include "network.hpp"

#include <cmath>
#include <cstdlib>


Network::~Network()
{
	if (_input != NULL)
		delete[] _input;
	for(int i = 0; i < _num_layers; ++i)
		delete(_layers[i]);
}

Network::Network(float input[], int num_input, int num_output, int num_layers):
	_num_inputs(num_input),
	_num_outputs(num_output),
	_num_layers(num_layers)
{
	_input = new float [num_input];
	for (int i = 0; i < num_input; ++i)
		_input[i] = input[i];
}

void Network::add_layer(Layer* layer)
{
	_layers.push_back(layer);
}

void Network::forward_propagation()
{
	_layers[0]->set_output(_input);
	for (int i = 1; i < _num_layers; ++i) {
		_layers[i]->compute_output(_layers[i-1]->get_output());
	}
}

void Network::compute_error(float target[])
{
	float e;
	float* y = _layers[_num_layers - 1]->get_output();
	_mse = 0.0f;
	for (int i = 0; i < _num_outputs; ++i) {
		e = target[i] - y[i];
		_mse += e * e;
	}
	_mse /= _num_outputs;
}
