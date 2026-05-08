#include "model.hpp"
#include <cassert>

#define FIRST_HIDEN_LAYER 12000

LinearModel::LinearModel() {
    layer1_weights = register_parameter(
        "layer1_weights",
        torch::randn({FIRST_HIDEN_LAYER, AMOUNT_ROWS}) / std::sqrt(6.0f)
    );
    layer1_bias = register_parameter(
        "layer1_bias",
        torch::randn({FIRST_HIDEN_LAYER}) / std::sqrt(6.0f)
    );

    layer2_weights = register_parameter(
        "layer2_weights",
        torch::randn({AMOUNT_ROWS, FIRST_HIDEN_LAYER}) / std::sqrt(6.0f)
    );
    layer2_bias = register_parameter(
        "layer2_bias",
        torch::randn({AMOUNT_ROWS}) / std::sqrt(6.0f)
    );

    layer3_weights = register_parameter(
        "layer3_weights",
        torch::randn({3, AMOUNT_ROWS}) / std::sqrt(6.0f)
    );
    layer3_bias = register_parameter(
        "layer3_bias",
        torch::randn({3}) / std::sqrt(6.0f)
    );

    ln1 = register_module("ln1", torch::nn::LayerNorm(std::vector<int64_t>{FIRST_HIDEN_LAYER}));
    ln2 = register_module("ln2", torch::nn::LayerNorm(std::vector<int64_t>{AMOUNT_ROWS}));
}

torch::Tensor LinearModel::forward(torch::Tensor x) {
    x = x.unsqueeze(0);

    auto layer1 = torch::matmul(x, layer1_weights.t()) + layer1_bias;
    layer1 = ln1(layer1);
    layer1 = torch::relu(layer1);

    auto layer2 = torch::matmul(layer1, layer2_weights.t()) + layer2_bias;
    layer2 = ln2(layer2);
    layer2 = torch::relu(layer2);

    auto layer3 = torch::matmul(layer2, layer3_weights.t()) + layer3_bias;

    return layer3.squeeze(0);
}

Eigen::Vector3f LinearModel::predict(const Eigen::MatrixXf& embeddings) {
    torch::NoGradGuard no_grad;
    this->eval();

    Eigen::VectorXf pooled = embeddings.colwise().mean();

    assert(pooled.size() == AMOUNT_ROWS);

    torch::Tensor x = torch::from_blob(pooled.data(),
                                        {AMOUNT_ROWS},
                                        torch::TensorOptions().dtype(torch::kFloat)
                                        ).clone();

    torch::Tensor logits = forward(x);
    torch::Tensor probs = torch::softmax(logits, 0);

    Eigen::Vector3f result;
    std::memcpy(
        result.data(),
        probs.data_ptr<float>(),
        3 * sizeof(float)
    );

    return result;
}
