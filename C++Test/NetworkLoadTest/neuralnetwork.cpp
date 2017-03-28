#include "neuralnetwork.h"
#include <iostream>
#include "activationfunction.h"

using std::vector;
using arma::mat;
using arma::dot;
using std::cout;
using std::endl;

NeuralNetwork::NeuralNetwork(int inputs,
                             int nLayers,
                             int nNodes,
                             int outputs,
                             vector<mat> weights,
                             vector<mat> biases) :
        m_inputs (inputs),
        m_nLayers(nLayers),
        m_nNodes (nNodes),
        m_outputs(outputs),
        m_weights(weights),
        m_biases (biases),
        m_relu   ("relu"),
        m_sigmoid("sigmoid") {
}

mat NeuralNetwork::evaluate(mat y_) {
    //cout << "-------------" << endl;
    for (int k = 0; k < m_nLayers+1; k++) {
        y_ = y_ * m_weights.at(k) + m_biases.at(k);
        m_sigmoid.evaluate(y_);
        //cout << "y_(" << k << ") = " << y_;
    }
    y_ = y_ * m_weights.at(m_nLayers+1) + m_biases.at(m_nLayers+1);
    //cout << "-------------" << endl;
    return y_;
}
