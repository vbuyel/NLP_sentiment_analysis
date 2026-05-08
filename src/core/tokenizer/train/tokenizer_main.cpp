#include "tokenize_train.hpp"
#include "../../structure.hpp"

std::map<KEY, VALUE> print_user_input(std::string input) {
    std::map<KEY, VALUE> seq_tokenized = {};
    
    std::vector<VALUE> input_ids;
    
    std::cout << std::endl << "[";
    for (int i = 0; i < input.length();) {
        int length = determine_token_length(input, i, tokens);
        std::string token = input.substr(i, length);
        
        std::cout << token;

        input_ids.push_back(tokens[token]);
        seq_tokenized.insert({token, tokens[token]});
        
        i += token.length();
        
        if (token.length() == 0) {
            printf("\nOutput error. Simbol %c is not found.\n", input[i]);
            return seq_tokenized;
        }
        
        if (i < input.length()) {
            std::cout << "|";
        }
    }
    std::cout << "]" << std::endl;
    
    
    // 3. Print each word's vector
    std::cout << std::endl << "[";
    for (int i = 0; i < input_ids.size(); i++) {
        std::cout << static_cast<int>(input_ids[i]);
        
        if (i < input_ids.size()-1) {
            std::cout << "|";
        }
    }
    std::cout << "]" << std::endl;

    return seq_tokenized;
}

void train_tokenization(std::string user_str) {
    for (int i = 0; user_str[i] != '\0'; i++) {
        user_str[i] = (user_str[i] > 0x40 && user_str[i] < 0x5B) ? user_str[i] = user_str[i]+0x20 : user_str[i];
    }
    
    std::string new_key = "new_key";
    new_key = find_new_token(user_str, tokens);
    
    if (new_key != "") {
        tokens.insert({new_key, tokens.size()});
    }
}

void tokenizer_BPE(void) {

    std::fstream inputFile;
    inputFile.open(SYSTEM_FILE_PATH_TOKENS, std::ios::in);
    
    if (inputFile.is_open()) {
        
        for (int i = 0x20; i <= 0x7E; i++) {
            std::string key(1, static_cast<char>(i));
            tokens.insert({key, tokens.size()});
        }
        
        std::string file_str, result_str, temp_str;
        
        for (int i = 0; std::getline(inputFile, file_str); i++) {
            
            int j = 0;
            int num_founded_in_line = 0;
            int tokens_size = -1;
            
            // ============== Each Line =================
            
            while (j < file_str.length() && j < MAX_TOKENS_AMOUNT && tokens_size != int(tokens.size())) {
                std::vector<std::thread> workers;
                workers.reserve(NUM_THREADS);
                
                for (size_t k = 0; k < NUM_THREADS && j < file_str.length() && j < MAX_TOKENS_AMOUNT && tokens_size != int(tokens.size()); k++) {
                    tokens_size = int(tokens.size());
                    result_str = "";
                    int len = int(file_str.length());
                    int value = MAX_TOKENS_AMOUNT * 10;
                    
                    for (; j < len && j < value; j++) {
                        if (!(file_str[j] >= 0x21 && file_str[j] <= 0x26)
                            && !(file_str[j] >= 0x28 && file_str[j] <= 0x40)
                            && !(file_str[j] >= 0x3A && file_str[j] <= 0x40)
                            && !(file_str[j] >= 0x5B && file_str[j] <= 0x60)
                            && !(file_str[j] >= 0x7B && file_str[j] <= 0x7E)) {
                            
                            result_str += file_str[j];
                        } else {
                            result_str += ' ';
                        }
                    }
                    
                    auto curr_string = result_str;
                    workers.emplace_back([curr_string]() {
                        train_tokenization(curr_string);
                    });
                    num_founded_in_line++;
                }
                
                for (auto &th : workers) {
                    th.join();
                }
            }
            
            // ============ End of each line ==========
            
            if (!std::getline(inputFile, temp_str) && int(tokens.size()) < MAX_TOKENS_AMOUNT && tokens_size != int(tokens.size())) {
                inputFile.clear();
                inputFile.seekg(0, std::ios::beg);
            }
        }
    }
    inputFile.close();
    temp_tokens.clear();
}

