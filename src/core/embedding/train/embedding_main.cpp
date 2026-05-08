#include <iostream>
#include <cmath>

#include "../../main_types.hpp"
#include "../../structure.hpp"

#include "embedding_structure.hpp"

void initialize_matrix(std::vector<std::vector<float>> &matrix, int y_size, int x_size) {
    for (int i = 0; i < y_size; i++) {
        float min_limit = (float) -1/std::sqrt(6);
        float max_limit = (float) 1/std::sqrt(6);
        
        for (int j = 0; j < x_size; j++) {
            matrix[i][j] = (rand() % int((max_limit - min_limit) * 1000)) / 1000.0 + min_limit;
        }
    }
}

void embedding(void) {
    
    tokens_matrix = Eigen::MatrixXf::Random(tokens.size(), AMOUNT_ROWS) / std::sqrt(6);
    layer1_weights.assign(MAX_TOKENS_AMOUNT, std::vector<float>(AMOUNT_ROWS, 0.0f));
    layer2_weights.assign(int(tokens.size()), std::vector<float>(MAX_TOKENS_AMOUNT, 0.0f));

    initialize_matrix(layer1_weights, MAX_TOKENS_AMOUNT, AMOUNT_ROWS);
    initialize_matrix(layer2_weights, int(tokens.size()), MAX_TOKENS_AMOUNT);
    
    word_matrix_train();
}
