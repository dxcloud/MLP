#include <cmath>
#include <ctime>
#include <cassert>
#include <fstream>
#include <iostream>
#include <algorithm>
 
#include <boost/array.hpp>
#include <boost/random.hpp>
#include <boost/cstdlib.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
using namespace std;
using namespace boost;
using namespace boost::numeric;
 
 
const double       learning_rate = 0.1;  // define learning rate
const double       lambda        = 0.0;  // define lambda for weight decay
const unsigned int mnbatch_sz    = 1250; // define size of batch
const unsigned int epoc          = -1;   // define number of epoch
const unsigned int midsz         = 200;  // define number of hiden neurons
 
 
template <typename T = double>
struct sum {
	sum(const T & init = T()) : value(init) { }
	void operator()(const T & val) {
		value += val;
	}
	T value;
};
 
struct tanh_nl {
	template <typename T>
	T operator()(const T & vec) {
		T res(vec.size());
		for (size_t i = 0, e = vec.size(); i != e; ++i) {
			res(i) = tanh(vec(i));
		}
		return res;
	}
};
 
struct softmax_nl {
	template <typename T>
	T operator()(const T & vec) {
		T tmp(vec.size());
		for (size_t i = 0, e = vec.size(); i != e; ++i) {
			tmp(i) = exp(vec(i));
		}
		typename T::value_type exp_sum =
		            for_each(tmp.begin(), tmp.end(),
		                     sum<typename T::value_type>(0)).value;
		T res(vec.size());
		for (size_t i = 0, e = vec.size(); i != e; ++i) {
			res(i) = tmp(i) / exp_sum;
		}
		return res;
	}
};
 
template <int Dim, int NbClass>
class MLP_tanh_softmax
{
public:
 
	typedef ublas::vector<double>   mlp_vec;
	typedef ublas::matrix<double>   mlp_mat;
	typedef tuple<mlp_mat, mlp_mat,              // W1, W2
	              mlp_vec, mlp_vec> mlp_params;  // b1, b2
	typedef array<mlp_vec, 4>       fprop_vecs;
 
#define PARAMS_SIZE mlp_mat(midsz, Dim), mlp_mat(NbClass, midsz), \
                    mlp_vec(midsz),      mlp_vec(NbClass)
#define INIT_MATS(x) do { \
                     get<0>((x)) = ublas::zero_matrix<double>(midsz, Dim); \
                     get<1>((x)) = ublas::zero_matrix<double>(NbClass, midsz); \
                     } while(false)
#define INIT_VECS(x) do { \
                     get<2>((x)) = ublas::zero_vector<double>(midsz); \
                     get<3>((x)) = ublas::zero_vector<double>(NbClass); \
                     } while(false)
 
	MLP_tanh_softmax()
	{
		reset();
	}
 
	void reset()
	{
		params = mlp_params(PARAMS_SIZE);
 
		mt19937 rng(static_cast<boost::uint32_t>(time(0)));
 
		// Initialisation aléatoire des paramêtres de la couche cachée dans ]-1/Dim, 1/Dim[
		// Où Dim est la taille des vecteurs d'entrée
		double ini = 1.0 / sqrt(double(Dim));
		uniform_real<> dist1(-ini, ini);
		variate_generator<mt19937&, uniform_real<> > rand1(rng, dist1);
		for (mlp_mat::array_type::iterator
		            it  = get<0>(params).data().begin(),
		            end = get<0>(params).data().end();
		            it != end; ++it) {
			*it = rand1();
		}
 
		// Initialisation aléatoire des paramêtres de la couche de sortie dans ]-1/midsz, 1/midsz[
		//Où midsz est la taille de la couche cachée
		ini = 1.0 / sqrt(double(midsz));
		uniform_real<> dist2(-ini, ini);
		variate_generator<mt19937&, uniform_real<> > rand2(rng, dist2);
		for (mlp_mat::array_type::iterator
		            it  = get<1>(params).data().begin(),
		            end = get<1>(params).data().end();
		            it != end; ++it) {
			*it = rand2();
		}
 
		// Initialisation des biais à 0
		INIT_VECS(params);
	}
 
	int test(const mlp_vec & vec)
	{
		fprop_vecs tmp;
		forward_prop(vec, tmp);
		return distance(tmp[3].begin(),
		                max_element(tmp[3].begin(), tmp[3].end()));
	}
 
	int test(const vector<mlp_vec> & test_set,
	         const vector<int>     & classes,
	         size_t first, size_t size)
	{
		int res = 0;
		for (size_t i = first, e = first + size; i != e; ++i) {
			int pred = test(test_set[i]);
			if (pred == classes[i])
				++res;
		}
		return res;
	}
 
	// Entraine le réseau de neurones par mini batch jusqu'à ce que l'on trouve un minimum local
	// (supposé atteind lorsque l'entrainement stagne pour stop_count epoques)
 
	// L'ensemble de données est coupé en deux partie (chacune composés d'éléments contigues dans l'ensemble)
	// La partie d'entrainement, sur laquel sera effectuée la rétropropagation, et la partie validation 
	// avec laquel on teste les performances (et donc l'évolution itérative).
	static const unsigned stop_count = 5;	
	void train(const vector<mlp_vec> & data_set,
	           const vector<int>     & classes,
	           size_t train_sz, size_t valid_sz,
	           ostream & ostr)
	{
		bool min_found = false; //< At least local one
		double actual_min = 100.0;
		unsigned int pos = 0, count = 0, egal_count = 0;
		unsigned int i = 0;
		while (i < epoc) {
			if (pos == 0) { // One more epoc
				static double last_error = 100.0;
 
				ostr << "Iteration " << count++ << endl;
 
				int tst = 0;
				double error_percent = 0.0;
				if (valid_sz != 0) {
					tst = test(data_set, classes, 0, train_sz);
					error_percent =
					      100.0 * (train_sz - tst) / double(train_sz);
					ostr << "Erreur Entrainement : "
					     << error_percent << endl;
 
					tst = test(data_set, classes,
					           train_sz, valid_sz);
					error_percent =
						100.0 * (valid_sz - tst) / double(valid_sz);
					ostr << "Erreur Validation : "
					     << error_percent << endl;
 
					ostr << endl;
				}
 
				if (last_error == error_percent) {
					if (min_found && ++egal_count == stop_count)
						break;
				} else {
					if (error_percent <= actual_min) {
						min_found = false;
						actual_min = 100.0;
					}
					if (error_percent > last_error) {
						min_found = true;
						actual_min = last_error;
					}
					egal_count = 0;
					last_error = error_percent;
				}
			}
 
			int mnbatch_size = min(mnbatch_sz, train_sz - pos);
 
			mlp_params grad = calc_grad(data_set, classes,
			                            pos, mnbatch_size);
 
			get<0>(params) -= learning_rate * get<0>(grad);
			get<1>(params) -= learning_rate * get<1>(grad);
			get<2>(params) -= learning_rate * get<2>(grad);
			get<3>(params) -= learning_rate * get<3>(grad);
 
			if ((pos += mnbatch_size) > train_sz) {
				++i;
				pos = 0;
			}
		}
	}
 
	void dump(ostream & ostr)
	{
		ostr <<   "W1:\t" << get<0>(params)
		     << "\nW2:\t" << get<1>(params)
		     << "\nb1:\t" << get<2>(params)
		     << "\nb2:\t" << get<3>(params)
		     << endl;
	}
 
private:
 
	mlp_params calc_grad(const vector<mlp_vec> & train_set,
	                     const vector<int>     & classes,
	                     size_t first,
	                     unsigned int mnbatch_size)
	{
		mlp_params grad(PARAMS_SIZE);
		INIT_MATS(grad);   INIT_VECS(grad);
		for (unsigned int i = 0; i < mnbatch_size; ++i) {
			int index = first + i;
 
			fprop_vecs tmp;
			forward_prop(train_set[index], tmp);
 
			mlp_params tmp_grad;
			back_prop(train_set[index],
			          classes[index],
			          tmp, tmp_grad);
 
			get<0>(grad) += get<0>(tmp_grad);
			get<1>(grad) += get<1>(tmp_grad);
			get<2>(grad) += get<2>(tmp_grad);
			get<3>(grad) += get<3>(tmp_grad);
		}
 
		get<0>(grad) /= double(mnbatch_size);
		get<1>(grad) /= double(mnbatch_size);
		get<2>(grad) /= double(mnbatch_size);
		get<3>(grad) /= double(mnbatch_size);
 
		return grad;
	}
 
	void forward_prop(const mlp_vec & vec, fprop_vecs & res)
	{
		fprop_vecs ret;
 
		ret[0] = prod(get<0>(params), vec) + get<2>(params);
		ret[1] = mid_func(ret[0]);
 
		ret[2] = prod(get<1>(params), ret[1]) + get<3>(params);
		ret[3] = out_func(ret[2]);
 
#if !(defined NDEBUG)
	cout << "Input vector:\t" <<  vec   << endl;
	cout << "Middle activ:\t" << ret[0] << endl;
	cout << "Middle w/ nl:\t" << ret[1] << endl;
	cout << "Output activ:\t" << ret[2] << endl;
	cout << "Output w/ nl:\t" << ret[3] << endl;
#endif
 
		swap(ret, res);
	}
 
	void back_prop(const mlp_vec & vec, int clas,
	               const fprop_vecs & fp_vecs,
	               mlp_params & res)
	{
		mlp_params ret(PARAMS_SIZE);
 
		for (size_t i = 0, e = get<3>(ret).size(); i != e; ++i) {
			// Caca beurk, mais évite un branchement... :-)
			get<3>(ret)[i] = fp_vecs[3][i] - double(i == clas);
		}
		for (size_t i = 0, e = get<1>(ret).size1(); i != e; ++i) {
			for (size_t j = 0, f = get<1>(ret).size2(); j != f; ++j) {
				get<1>(ret)(i,j) = get<3>(ret)[i] * fp_vecs[1][j] +
				                   2 * lambda * get<1>(params)(i,j);
			}
		}
 
		mlp_vec dcdhs(fp_vecs[1].size());
		for (size_t i = 0, e = dcdhs.size(); i != e; ++i) {
			double sum = 0.0;
			for (size_t j = 0, f = get<1>(params).size1(); j != f; ++j) {
				sum += get<3>(ret)[j] * get<1>(params)(j,i);
			}
			dcdhs[i] = sum;
		}
 
		for (size_t i = 0, e = get<2>(ret).size(); i != e; ++i) {
			get<2>(ret)[i] = dcdhs[i] *
			                 (1 - fp_vecs[1][i] * fp_vecs[1][i]);
		}
		for (size_t i = 0, e = get<0>(ret).size1(); i != e; ++i) {
			for (size_t j = 0, f = get<0>(ret).size2(); j != f; ++j) {
				get<0>(ret)(i,j) = get<2>(ret)[i] * vec[j] +
				                   2 * lambda * get<0>(params)(i,j);
			}
		}
 
		swap(ret, res);
	}
 
#undef PARAMS_SIZE
#undef INIT_MATS
#undef INIT_VECS
 
	mlp_params params; // mlp_mat W1; mlp_mat W2; mlp_vec b1; mlp_vec b2;
 
	tanh_nl    mid_func;
	softmax_nl out_func;
};
 
 
int main()
{
	// Vecteurs de dimmension 784, 10 classes
	typedef MLP_tanh_softmax<784, 10> MLPts;
	MLPts mlp_ts = MLPts();
 
	// Loading MNIST
	cout << "Loading... ";   cout.flush();
 
	vector<MLPts::mlp_vec> data;
	vector<int>            classes;
	ifstream train("mnist.txt");
	size_t i = 0, count = 0;
	double tmp;
	while (train >> tmp && count < 30000) {
		if (i == 784) {
			classes.push_back(int(tmp));
			i = 0;
			++count;
			continue;
		} else if (i == 0) {
			data.push_back(MLPts::mlp_vec(784));
		}
 
		data.back()[i++] = tmp;
	}
	cout << "Loaded. (" << data.size() << " items)" << endl;
 
	ofstream output("res.txt");
	mlp_ts.train(data, classes, 25000, 5000, output);
	output << endl;
	mlp_ts.dump(output);
 
	return exit_success;
}
