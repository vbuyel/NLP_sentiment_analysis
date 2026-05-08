#include "../structure.hpp"
#include "neural_network.hpp"

std::vector<float> delta_cross_entropy(const std::vector<float> &curr_vector, int target_ind) {
    std::vector<float> result = curr_vector;
    result[target_ind] -= 1;
    return result;
}

void relu(std::vector<float> &curr_vector) {
    for (auto &elem : curr_vector) {
        elem = std::max(0.0f, elem);
    }
}

void relu_backward(std::vector<float> &dX, const std::vector<float> &X) {
    for (int i = 0; i < X.size(); i++) {
        if (X[i] <= 0) {
            dX[i] = 0;
        }
    }
}

BackProp_struct linear_backward(
                                 const std::vector<float> &dY, // [out]
                                 const std::vector<float> &X, // [in]
                                 const std::vector<std::vector<float>> &W // [out][in]
                                ) {
    size_t out = W.size();
    size_t in  = X.size();

    std::vector<float> dX(in, 0.0f);
    std::vector<std::vector<float>> dW(out, std::vector<float>(in, 0.0f));

    for (int o = 0; o < out; o++) {
        for (int i = 0; i < in; i++) {
            dW[o][i] = dY[o] * X[i];
            dX[i] += dY[o] * W[o][i];
        }
    }
    return {dX, dW};
}

std::vector<float> fc_layer(
                             const std::vector<float> &input,
                             const std::vector<std::vector<float>> &weights // [out][in]
                            ) {
    size_t out_features = weights.size();
    size_t in_features = input.size();

    std::vector<float> output(out_features, 0.0f);

    for (int o = 0; o < out_features; o++) {
        for (int i = 0; i < in_features; i++) {
            output[o] += weights[o][i] * input[i];
        }
    }
    return output;
}
