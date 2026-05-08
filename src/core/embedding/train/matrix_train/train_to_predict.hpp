#ifndef TRAIN_TO_PREDICT_HPP
#define TRAIN_TO_PREDICT_HPP

#include <iostream>
#include "../../../structure.hpp"
#include "../../../neural_network_functions/neural_network.hpp"

std::vector<float> softmax(const std::vector<float> &input);
std::vector<float> matrix_columns_sum(const std::vector<std::vector<float>> &matrix);

std::vector<float> fc_layer(
                             const std::vector<float> &input,
                             const std::vector<std::vector<float>> &weights
                            );
void batch_norm_1d(std::vector<float> &vector_to_norm);
BackProp_struct linear_backward(
                                 const std::vector<float> &dY,
                                 const std::vector<float> &X,
                                 const std::vector<std::vector<float>> &W
                                );
std::vector<float> delta_cross_entropy(const std::vector<float> &curr_vector, int target_ind);
void relu_backward(std::vector<float> &dX, const std::vector<float> &X);
void relu(std::vector<float> &curr_vector);

#endif
