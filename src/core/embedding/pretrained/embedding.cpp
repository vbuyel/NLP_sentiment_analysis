#include "embedding.hpp"
#include <random>

#include "../../structure.hpp"
#include "../train/embedding_structure.hpp"

Eigen::MatrixXf Embedding::embed(const std::vector<std::string>& user_tokens) {
    Eigen::MatrixXf matrix(user_tokens.size(), AMOUNT_ROWS);

    for (int i = 0; i < user_tokens.size(); i++) {
        matrix.row(i) = tokens_matrix.row(tokens[user_tokens[i]]);
    }

    return matrix;
}

void Embedding::fit() {
    embedding();
}
