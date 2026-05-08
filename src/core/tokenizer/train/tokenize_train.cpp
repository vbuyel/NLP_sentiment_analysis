#include "tokenize_train.hpp"
#include "../../structure.hpp"

uint8_t determine_token_length(const std::string &string, int index, const std::map<KEY, VALUE> &tokens) {
    int add_to_index = 0;
    while ((add_to_index+1 + index) <= string.length() && tokens.count(string.substr(index, add_to_index+1))) {
        add_to_index++;
    }
    return add_to_index;
}

std::string choose_new_token(/*const std::map<KEY, VALUE>& map*/) {
    int max_value = (temp_tokens.size() > 0)
                    ? static_cast<int>(temp_tokens.begin()->second)
                    : -1;
    std::string new_key = (temp_tokens.size() > 0)
                            ? temp_tokens.begin()->first
                            : "";
    
    for (const auto& kv : temp_tokens) {
        if (static_cast<int>(kv.second) > max_value) {
            max_value = kv.second;
            new_key = kv.first;
        }
    }
    
    temp_tokens.erase(new_key);
    std::cout << new_key << " ";
    return new_key;
}

std::string find_new_token(const std::string &text_line, const std::map<KEY, VALUE> &tokens) {
    std::map<KEY, VALUE> temp_map;
    std::string fixed_temp_str = "";
    std::string prev_temp_str = "";
    std::string temp_str = "", founded_str = "";
    int jump = 1, max_amount = 0, chack_value = 0;
    
    if (int(temp_tokens.size()) > 0) {
        for (auto kv : temp_tokens) {
            if (max_amount < kv.second) {
                max_amount = kv.second;
                founded_str = kv.first;
            }
        }
    }
    
    for (int i = 0; i < text_line.length() && jump > 0; ) {
        
        jump = determine_token_length(text_line, i, tokens);
        jump += (jump == 0);
        
        temp_str = text_line.substr(i, jump);
        fixed_temp_str = prev_temp_str + temp_str;
        
        if (fixed_temp_str[0] != ' ' && fixed_temp_str[fixed_temp_str.length()-1] != ' ' && !tokens.count(fixed_temp_str)) {
            chack_value = temp_tokens[fixed_temp_str] = temp_tokens[fixed_temp_str]+1;
            if (max_amount < chack_value) {
                max_amount = chack_value;
                founded_str = fixed_temp_str;
            }
        }
        
        prev_temp_str = temp_str;
        i += jump;
    }
    temp_tokens.erase(founded_str);
    std::cout << founded_str << " "; // only for debug (1/3 times faster without it)
    return founded_str;
}
