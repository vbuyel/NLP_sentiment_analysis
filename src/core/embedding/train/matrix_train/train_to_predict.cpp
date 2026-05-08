#include "../../../structure.hpp"
#include "train_to_predict.hpp"

float lr_value = LEARNING_RATE;
float lr_step = LEARNING_RATE / MAX_EPOCH_AMOUNT;

void update_sgd(
                 std::vector<std::vector<float>> &W,
                 const std::vector<std::vector<float>> &dW
                ) {
    for (size_t o = 0; o < W.size(); o++) {
        for (size_t i = 0; i < W[o].size(); i++) {
            W[o][i] -= lr_value * dW[o][i];
        }
    }
}

void matrix_train(std::string token_current, std::string token_to_predict, std::string token_last) {
    
    std::vector<float> input_layer, layer1, layer2, loss_vector;
    std::vector<float> probabilities(int(tokens.size()), 0);
    std::string token_predicted;
    float max_value = 1;
    int ind_max_value, i;
    
    input_layer.assign(AMOUNT_ROWS, 0.0f);
    
    lr_value = LEARNING_RATE;
    
    for (int epoch = 0; probabilities[tokens[token_to_predict]] < max_value*0.95 && epoch < MAX_EPOCH_AMOUNT; epoch++) {
        
        {
            std::lock_guard<std::mutex> lock(train_mutex);
            
            for (int i = 0; i < AMOUNT_ROWS; i++) {
                input_layer[i] = tokens_matrix(tokens[token_current], i) + tokens_matrix(tokens[token_last], i);
            }
        }
        
        layer1 = fc_layer(input_layer, layer1_weights);
        relu(layer1);
        
        layer2 = fc_layer(layer1, layer2_weights);
        
        probabilities = softmax(layer2);
        
        ind_max_value = 0;
        i = 0;
        max_value = 0;
        for (const auto prob : probabilities) {
            if (max_value < prob) {
                ind_max_value = i;
                max_value = prob;
            }
            i++;
        }
        
        for (const auto &kv : tokens) {
            if (kv.second == ind_max_value) {
                token_predicted = kv.first;
            }
        }
        
        if (probabilities[tokens[token_to_predict]] < max_value*0.95) {
            
            // Soft Max + Cross Entropy Loss
            loss_vector = delta_cross_entropy(probabilities, tokens[token_to_predict]);
            
            auto layer2_backward = linear_backward(loss_vector, layer1, layer2_weights);
            relu_backward(layer2_backward.dX, layer1);
            
            auto layer1_backward = linear_backward(layer2_backward.dX, input_layer, layer1_weights);
            
            {
                std::lock_guard<std::mutex> lock(train_mutex);
                
                // Update the weights2
                update_sgd(layer2_weights, layer2_backward.dW);
                // Update the weights1
                update_sgd(layer1_weights, layer1_backward.dW);
                
                // Update tokens' vectors
                for (int i = 0; i < AMOUNT_ROWS; i++) {
                    tokens_matrix(tokens[token_current], i) -= lr_value * layer1_backward.dX[i];
                    tokens_matrix(tokens[token_last], i) -= lr_value * layer1_backward.dX[i];
                }
            }
        }
    }
    
   std::cout << token_current << " |  " << token_to_predict << "  | " << token_last << "\t=>\t" << token_predicted;
   std::cout << "\t\t\t\ttokent_to_predict " << probabilities[tokens[token_to_predict]] << ", token_predicted " << probabilities[tokens[token_predicted]] << std::endl;
}
