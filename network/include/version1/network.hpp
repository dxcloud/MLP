#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "layer.hpp"
#include "constant.hpp"

#include <vector>

class Network
{
public:
	Network(float input[], int num_input=NUM_NEURONS_IN, int num_output=NUM_NEURONS_OUT, int num_layers=NUM_LAYERS);
	~Network();

	void add_layer(Layer* layer);
	void init_input(float input[], int n);

	void forward_propagation();
	void back_propagation();
	void compute_error(float target[]);

protected:
	float*_input;		///< Array contains input values
	int _num_inputs;	///< Number of input values
	int _num_outputs;	///< Number of output values
	int _num_layers;	///< Number of layers
	float _mse;			///< Mean Squared Error

	std::vector<Layer*> _layers;
};

#endif /*!NETWORK_HPP*/
