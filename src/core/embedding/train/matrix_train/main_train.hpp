#ifndef MAIN_TRAIN_HPP
#define MAIN_TRAIN_HPP

#include "../../../structure.hpp"

uint8_t determine_token_length(const std::string& string, int index, const std::map<KEY, VALUE>& tokens);
void matrix_train(std::string token_current, std::string token_to_predict, std::string token_last);

#endif
