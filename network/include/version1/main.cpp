#include "network.hpp"
#include "layer.hpp"
#include "neuron.hpp"
#include "constant.hpp"

#include <iostream>

#define N 4

int main(int argc, char *argv[])
{
	int i;
	float input[N] = {1.2f, 2.3f, 0.6f, 9.3f};
	Network network(input, N);

	Layer * l1 = new Layer(NUM_NEURONS_IN);
	Layer * l2 = new Layer(NUM_NEURONS_HIDDEN);
	Layer * l3 = new Layer(NUM_NEURONS_OUT);
	
	for (i=0; i<NUM_NEURONS_IN; i++) {
		l1->add_neuron(new Neuron());
	}
	for (i=0; i<NUM_NEURONS_HIDDEN; i++) {
		l2->add_neuron(new Neuron(NUM_NEURONS_IN));
	}
	for (i=0; i<NUM_NEURONS_OUT; i++) {
		l3->add_neuron(new Neuron(NUM_NEURONS_HIDDEN));
	}

	network.add_layer(l1);
	network.add_layer(l2);
	network.add_layer(l3);
	network.forward_propagation();

	return 1;
}
