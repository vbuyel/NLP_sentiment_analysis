#include "../pretrained/model.hpp"
#include "../../structure.hpp"
#include "../../embedding/pretrained/embedding.hpp"
#include "../../tokenizer/pretrained/tokenizer.hpp"
#include <cmath>
#include <filesystem>

void LinearModel::load(torch::serialize::InputArchive& archive) {
    archive.read("layer1_weights", layer1_weights);
    archive.read("layer1_bias", layer1_bias);

    archive.read("layer2_weights", layer2_weights);
    archive.read("layer2_bias", layer2_bias);

    archive.read("layer3_weights", layer3_weights);
    archive.read("layer3_bias", layer3_bias);

    torch::serialize::InputArchive ln1_archive;
    archive.read("ln1", ln1_archive);
    ln1->load(ln1_archive);

    torch::serialize::InputArchive ln2_archive;
    archive.read("ln2", ln2_archive);
    ln2->load(ln2_archive);
}

inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
}

std::vector<std::string> split(const std::string &s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

void LinearModel::fit(int num_epochs) {
    this->train();

    Tokenizer tokenizer;
    Embedding embedding;
    torch::optim::Adam optimizer(this->parameters(), torch::optim::AdamOptions(LEARNING_RATE));

    std::ifstream file(SYSTEM_FILE_PATH_MODEL_TRAIN);
    std::string line;
    std::vector<std::string> elements;

    
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        while (std::getline(file, line)) {
            elements = split(line, ';');
            ltrim(elements[1]);

            auto user_tokens = tokenizer.tokenize(elements[0]);
            if (user_tokens.size() == 0) continue;

            Eigen::MatrixXf tokens_vectors = embedding.embed(user_tokens);

            Eigen::VectorXf pooled_vectors = tokens_vectors.colwise().mean();
            torch::Tensor laten_vector = torch::from_blob(pooled_vectors.data(),
                                            {AMOUNT_ROWS},
                                            torch::kFloat).clone();

            torch::Tensor logits = forward(laten_vector).unsqueeze(0);

            torch::Tensor target = torch::tensor(stoi(elements[1]), torch::kLong).unsqueeze(0);

            auto loss = torch::nn::functional::cross_entropy(logits, target);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }

        file.clear();
        file.seekg(0, std::ios::beg);
        std::cout << "Epoch: " << epoch+1 << std::endl;
    }
}
