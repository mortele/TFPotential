#include <iostream>
#include <fstream>
#include <armadillo>
#include <vector>
#include <string>
#include "neuralnetwork.h"

using std::cout;
using std::endl;
using arma::mat;
using arma::zeros;
using std::vector;
using std::string;

int main(int, char**) {

    std::ifstream inFile;
    std::string fileName = "/Users/morten/Documents/Master/TFPotential/C++Test/network-99";
    inFile.open(fileName, std::ios::in);
    if (inFile.is_open()) {
        cout << "Successfully openend file: " << fileName << endl;
    } else {
        cout << "Unable to open file: " << fileName << ", exiting." << endl;
        exit(1);
    }
    int inputs, nLayers, nNodes, outputs;
    inFile >> inputs >> nLayers >> nNodes >> outputs;

//    cout << "inputs/layers/nodes/outputs " << inputs
//           << ", " << nLayers << ", " << nNodes << ", "
//           << outputs << endl;

    vector<mat> weights(nLayers+2);
    vector<mat> biases(nLayers+2);


    for (int k = 0; k < nLayers+2; k++) {
        int iLimit = nNodes;
        int jLimit = nNodes;
        if (k == 0) iLimit = inputs;
        else if (k == nLayers+1) jLimit = outputs;

        weights.at(k) = zeros<mat>(iLimit, jLimit);
        biases.at(k)  = zeros<mat>(1,      jLimit);

        for (int i = 0; i < iLimit; i++) {
            for (int j = 0; j < jLimit; j++) {
                inFile >> weights.at(k)(i,j);
            }
        }
        for (int j = 0; j < jLimit; j++) {
            inFile >> biases.at(k)(j);
        }
    }

    for (int k = 0; k < nLayers+2; k++) {
        //cout << "w[" << k << "] = " << weights.at(k) << endl;
        //cout << "b[" << k << "] = " << biases.at(k)  << endl;
    }

    // [0.1],[-0.14243]
    mat x = zeros<mat>(1,1);
    x(0,0) = 0.1;
    //x(1,0) = 0.2;

    NeuralNetwork nn(inputs, nLayers, nNodes, outputs, weights, biases);
    mat y_ = nn.evaluate(x);
    cout << "nn(0.1) = " << y_ << endl;


    return 0;
}
