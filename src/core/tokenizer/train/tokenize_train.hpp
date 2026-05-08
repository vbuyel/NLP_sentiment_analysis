#ifndef TOKENIZE_TRAIN_HPP
#define TOKENIZE_TRAIN_HPP

#include <iostream>
#include <map>
#include "../../main_types.hpp"

void tokenizer_BPE(void);

std::string find_new_token(const std::string& text_line, const std::map<KEY, VALUE>& tokens);
uint8_t determine_token_length(const std::string& str, int index, const std::map<KEY, VALUE>& tokens);

#endif
