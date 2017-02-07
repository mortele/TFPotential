#include <iostream>
#include <fstream>
#include <armadillo>
#include <vector>

using std::cout;
using std::endl;
using arma::mat;
using std::vector;

int main(int, char**) {

    std::ifstream inFile;
    inFile.open("../network-12", std::ios::in);

    int inputs, nLayers, nNodes, outputs;
    inFile >> inputs >> nLayers >> nNodes >> outputs;

    cout << inputs << ", " << nLayers << ", " << nNodes << ", " << outputs << endl;

    vector<mat> weights(nLayers);
    vector<mat> biases(nLayers);



    return 0;
}
