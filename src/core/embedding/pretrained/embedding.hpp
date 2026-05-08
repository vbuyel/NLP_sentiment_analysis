#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <Eigen/Dense>

#include "../../structure.hpp"

extern Eigen::MatrixXf tokens_matrix;
extern std::vector<std::vector<float>> layer1_weights;
extern std::vector<std::vector<float>> layer2_weights;

class Embedding {
public:
    Eigen::MatrixXf embed(const std::vector<std::string>& tokens);
    void fit();
private:
    std::unordered_map<std::string, Eigen::VectorXf> table;
};
