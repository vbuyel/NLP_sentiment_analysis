#include <cmath>
#include <vector>
#include "../structure.hpp"

std::vector<float> matrix_columns_sum(const std::vector<std::vector<float>> &matrix) {
    std::vector<float> result(AMOUNT_ROWS, 0);
    
    for (int i = 0; i < AMOUNT_ROWS; i++) {
        for (int j = 0; j < MAX_TOKENS_AMOUNT; j++) {
            result[j] += matrix[i][j];
        }
    }
    
    return result;
}

float vector_exp_sum(const std::vector<float> &array) {
    float result = 0;
    for (int i = 0; i < int(array.size()); i++) {
        result += std::exp(array[i]);
    }
    return result;
}

std::vector<float> softmax(const std::vector<float> &input) {
    float max_val = *std::max_element(input.begin(), input.end());
    std::vector<float> exp_x(input.size());

    float sum = 0;
    for (int i = 0; i < input.size(); i++) {
        exp_x[i] = std::exp(input[i] - max_val);
        sum += exp_x[i];
    }

    for (int i = 0; i < input.size(); i++) {
        exp_x[i] /= sum;
    }
    return exp_x;
}

float count_mean(const std::vector<float> &input_vector) {
    float sum = 0;
    for (const auto &value : input_vector) {
        sum += value;
    }
    return sum / input_vector.size();
}

float count_variance(const std::vector<float> &input_vector, float mean) {
    float sum = 0;
    for (int i = 0; i < input_vector.size(); i++) {
        float difference = input_vector[i] - mean;
        sum += difference * difference;
    }
    return sum / input_vector.size();
}

void batch_norm_1d(std::vector<float> &vector_to_norm) {
    float mean = count_mean(vector_to_norm);
    float variance = count_variance(vector_to_norm, mean);
    
    for (int i = 0; i < int(vector_to_norm.size()); i++) {
        vector_to_norm[i] = (vector_to_norm[i] - mean) / sqrt(variance + 0.001);
    }
}
