#include <QApplication>
#include "mainwindow.hpp"

#include <fstream>
#include "../core/model/pretrained/model.hpp"
#include "mainwindow.hpp"
#include "../core/structure.hpp"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    MainWindow w;

    {
        // Load pretrained tokens
        std::ifstream file;
        file.open("../pretrained_weights/tokens.txt");
        if (!file) throw std::runtime_error("Cannot open tokenizer file.");

        std::string line;
        std::vector<std::string> elements;
        while (std::getline(file, line)) {
            elements = split(line, ' ');
            if (elements.size() > 2) {
                tokens[" "] = std::stoi(elements.back());
            } else {
                tokens[elements[0]] = std::stoi(elements[1]);
            }
        }
    }

    {
        // Load Tokens' Vectors
        std::ifstream file("../pretrained_weights/embeding_matrix.bin", std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open embedding file.");

        size_t rows = 0, cols = 0;
        file.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
        file.read(reinterpret_cast<char*>(&cols), sizeof(size_t));

        size_t total_elements = tokens.size() * AMOUNT_ROWS;
        std::vector<float> flat_buffer(total_elements);

        file.read(reinterpret_cast<char*>(flat_buffer.data()), total_elements * sizeof(float));
        file.close();

        tokens_matrix.resize(tokens.size(), AMOUNT_ROWS);

        for (size_t i = 0; i < tokens.size(); ++i) {
            for (size_t j = 0; j < AMOUNT_ROWS; ++j) {
                tokens_matrix(i, j) = flat_buffer[i * AMOUNT_ROWS + j];
            }
        }
    }

    torch::serialize::InputArchive archive;
    archive.load_from("../pretrained_weights/linear_model.pt");
    w.model.load(archive);
    w.model.eval();

    w.show();
    return app.exec();
}
