#include "activationfunction.h"
#include <math.h>

using arma::mat;

ActivationFunction::ActivationFunction(std::string type) {
    if (type == "relu") {
        m_type = 0;
    } else if (type == "sigmoid") {
        m_type = 1;
    }
}

arma::mat ActivationFunction::operator()(arma::mat x) {
    return x;
}

double ActivationFunction::relu(double x) {
    return (x > 0 ? x : 0);
}

double ActivationFunction::reluDerivative(double x) {
    return (x > 0 ? 1 : 0);
}

double ActivationFunction::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double ActivationFunction::sigmoidDerivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

double ActivationFunction::evaluate(double x) {
    if (m_type == 0) {
        return relu(x);
    } else if (m_type == 1) {
        return sigmoid(x);
    } else {
        return NAN;
    }
}

double ActivationFunction::derivative(double x) {
    if (m_type == 0) {
        return reluDerivative(x);
    } else if (m_type == 1) {
        return sigmoidDerivative(x);
    } else {
        return NAN;
    }
}

void ActivationFunction::evaluate(mat& x) {
    for (int i = 0; i < x.size(); i++) {
        x.at(i) = evaluate(x.at(i));
    }
}
