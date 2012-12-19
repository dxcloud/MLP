#include "neuron.hpp"

#include <cstdlib>
#include <cmath>

float Neuron::lambda = 1.0f;
float Neuron::WEIGHT_LOW = -0.5f;
float Neuron::WEIGHT_HIGH = 0.5f;

Neuron::~Neuron()
{
	if (_w != NULL)
		delete[] _w;
}

void Neuron::init_neuron(int num_prev_neurons)
{
	_w = NULL;
	if (num_prev_neurons != 0) {
		_w = new float [num_prev_neurons];
		for (int i = 0; i < num_prev_neurons; ++i) {
			_w[i] = random_weight();
			#if DEBUG
				std::cout << num_prev_neurons << "->w[" << i << "] = " << _w[i] << std::endl;
			#endif
		}
	}
}

float Neuron::random_weight(float low, float high)
{
	return low + (float)rand() / ((float)RAND_MAX / (high - low));
}

float Neuron::output(float input[], int n)
{
	#if DEBUG
		std::cout << "start output with " << n << " neurons" << std::endl;
	#endif
	float sum = 0.0f;
	for (int i = 0; i < n; ++i) {
		#if DEBUG
			std::cout << input[i] << " * " << _w[i] << std::endl;
		#endif
		sum += input[i] * _w[i];
	}
	_o = sigmoid(sum);
	#if DEBUG
		std::cout << "output " << _o << std::endl;
	#endif
	return _o;
}

float Neuron::sigmoid(float x)
{
	return 1.0f / (1 + expf(-lambda * x));
}
