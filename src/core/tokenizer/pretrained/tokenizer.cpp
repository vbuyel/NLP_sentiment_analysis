#include "tokenizer.hpp"
#include "../train/tokenize_train.hpp"
#include "../../structure.hpp"
#include <sstream>

std::vector<std::string> Tokenizer::tokenize(const std::string &text) {
    std::vector<std::string> user_tokens;
    std::istringstream iss(text);
    std::string word;
    
    while (iss >> word) {
        for (int i = 0; i < word.length(); ) {
            int length = determine_token_length(word, i, tokens);
            length += (length == 0);
            word = word.substr(i, length);
            user_tokens.push_back(word);
            i += length;
        }
    }
    return user_tokens;
}

void Tokenizer::fit() {
    tokenizer_BPE();
}
