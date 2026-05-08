#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <iostream>

struct BackProp_struct {
    std::vector<float> dX;
    std::vector<std::vector<float>> dW;
};

#endif
