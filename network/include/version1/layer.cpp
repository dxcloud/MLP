#include "layer.hpp"

#include <cstdlib>

Layer::Layer(int num_neurons):
	_num_neurons(num_neurons)
{
	#if DEBUG
		std::cout << "1 layer created with " << num_neurons << " neurons" << std::endl;
	#endif
	_output = new float[num_neurons];
}

Layer::~Layer()
{
	if (_output != NULL)
		delete[] _output;
	for(int i = 0; i < _num_neurons; i++)
		delete(_neurons[i]);
}

void Layer::add_neuron(Neuron *neuron)
{
	_neurons.push_back(neuron);
}

void Layer::set_output(float o[])
{
	for (int i = 0; i < _num_neurons; ++i)
		_output[i] = o[i];
}


void Layer::compute_output(float input[])
{
	for (int i = 0; i < _num_neurons; ++i) {
		_output[i] = _neurons[i]->output(input, _num_neurons);
		#if DEBUG
			std::cout << "output[" << i << "] = " << _output[i] << std::endl;
		#endif
	}
}
