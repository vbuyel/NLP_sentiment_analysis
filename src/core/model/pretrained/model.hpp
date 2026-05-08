#pragma once

#include <torch/torch.h>
#include <torch/serialize.h>
#include <Eigen/Dense>
#include "../../structure.hpp"

class LinearModel : public torch::nn::Module {
public:
    LinearModel();

    Eigen::Vector3f predict(const Eigen::MatrixXf& embeddings);

    torch::Tensor forward(torch::Tensor x);
    void fit(int num_epochs);
    void load(torch::serialize::InputArchive& archive);

private:
    torch::Tensor layer1_weights; // [AMOUNT_ROWS, MAX_TOKENS_AMOUNT]
    torch::Tensor layer1_bias;    // [MAX_TOKENS_AMOUNT]

    torch::Tensor layer2_weights; // [MAX_TOKENS_AMOUNT, AMOUNT_ROWS]
    torch::Tensor layer2_bias;    // [AMOUNT_ROWS]

    torch::Tensor layer3_weights; // [AMOUNT_ROWS, 3]
    torch::Tensor layer3_bias;    // [3]

    torch::nn::LayerNorm ln1{nullptr};
    torch::nn::LayerNorm ln2{nullptr};
};
