#pragma once
#include <vector>
#include <armadillo>
#include "activationfunction.h"

class NeuralNetwork {
private:
    int m_inputs;
    int m_nLayers;
    int m_nNodes;
    int m_outputs;
    std::vector<arma::mat>  m_weights;
    std::vector<arma::mat>  m_biases;
    ActivationFunction      m_relu;
    ActivationFunction      m_sigmoid;

public:
    NeuralNetwork(int, int, int, int,
                  std::vector<arma::mat>,
                  std::vector<arma::mat>);
    arma::mat evaluate(arma::mat);
};
