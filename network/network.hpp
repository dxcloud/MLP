#ifndef NETWORK_HPP_
#define NETWORK_HPP_

class Node
{
public:	
	float  x;     /* sortie */
	float  e;     /* erreur */
	float* w;     /* poids  */
	float* dw;    /* dernier poids pour les momentum  */
	float* wsave; /* poids sauvegardé */
};

class Layer {
public:	
	int nbNodes;
	Node* pNodes;
};

class Network {
public:	
	int    nbLayers;
	Layer* pLayers;
	
	float dEta;
	float dAlpha;
	float dGain;
	float dAvgTestError;

	float dMSE;
	float dMAE;

	// randomly init weights
	void randomWeights();

	// set input values in the first layer
	void SetInputSignal (float* input);

	// get output values from last layer
	void GetOutputSignal(float* output);

	// save the weight on each node
	void SaveWeights();
	
	// restore the weight on each node
	void RestoreWeights();

	void PropagateSignal();
	void ComputeOutputError(float* target);
	void BackPropagateError();
	void AdjustWeights();

	void Simulate(float* input, float* output, float* target, bool training);

	Network(int nl, int npl[]);
	~Network();

	int Train(const char* fnames);
	int Test (const char* fname);
	int Evaluate();

	void Run(const char* fname, const int& maxiter);

};

#endif
