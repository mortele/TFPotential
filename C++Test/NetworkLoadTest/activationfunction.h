#pragma once
#include <armadillo>
#include <string>
#include <math.h>

class ActivationFunction {
private:
    int     m_type = NAN;

    double      relu                (double x);
    double      reluDerivative      (double x);
    double      sigmoid             (double x);
    double      sigmoidDerivative   (double x);
    double      evaluate            (double x);
    double      derivative          (double x);

public:
    ActivationFunction   (std::string type);
    arma::mat operator() (arma::mat x);
    void evaluate   (arma::mat& x);
};

