#include "main_train.hpp"

void word_matrix_train(void) {
    std::fstream inputFile;
    inputFile.open(SYSTEM_FILE_PATH_TOKENS, std::ios::in);
    
    
    // 1.1 Prepare to get tokens
    std::string word, word_last, token_current_and_last;
    std::string token_current, token_to_predict, token_last;
    uint8_t jump = 0, jump_last = 0;
    uint8_t curr_position = 0;
    uint8_t flag = 0;
    
    inputFile >> word;
    word_last = word;
    jump = determine_token_length(word, 0, tokens);
    token_to_predict = word.substr(curr_position, jump);
    curr_position = jump;
    
    // 1.2 Get tokens (known | unknown | known)
    int iteration = 0;
    
    while (iteration < std::ceil(MAX_TOKENS_AMOUNT * 0.1) && word_last[0]) {
        std::vector<std::thread> workers;
        workers.reserve(NUM_THREADS);
        
        for (size_t i = 0; i < NUM_THREADS && iteration < std::ceil(MAX_TOKENS_AMOUNT * 0.1) && word_last[0]; ++i) {
            
            // Get the 1-st token
            token_current = token_to_predict;
            
            // Get the 2-nd token
            if (curr_position < word.length()) {
                jump = determine_token_length(word, curr_position, tokens);
                jump += (jump == 0);
                token_to_predict = word.substr(curr_position, jump);
                curr_position += jump;
            } else {
                if (flag) {
                    word = word_last;
                    flag = 0;
                } else {
                    inputFile >> word;
                }
                jump = determine_token_length(word, 0, tokens);
                jump += (jump == 0);
                token_to_predict = word.substr(0, jump);
                curr_position = jump;
            }
            
            // Get the 3-rd token
            if (curr_position < word.length()) {
                jump_last = determine_token_length(word, curr_position, tokens);
                jump_last += (jump_last == 0);
                token_last = word.substr(curr_position, jump_last);
            } else {
                inputFile >> word_last;
                jump_last = determine_token_length(word_last, 0, tokens);
                jump_last += (jump_last == 0);
                token_last = word_last.substr(0, jump_last);
                
                flag = 1;
            }
            
            // Predict the next token
            if (word_last[0]) {
                auto token1 = token_current;
                auto token2 = token_to_predict;
                auto token3 = token_last;
                workers.emplace_back([token1, token2, token3]() {
                    matrix_train(token1, token2, token3);
                });
                iteration++;
            }
        }
        
        for (auto &th : workers) {
            th.join();
        }
    }
}
