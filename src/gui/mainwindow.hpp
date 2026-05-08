#pragma once

#include <QMainWindow>
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>

#include "probabilitychart/probabilitychart.hpp"
#include "../core/tokenizer/pretrained/tokenizer.hpp"
#include "../core/embedding/pretrained/embedding.hpp"
#include "../core/model/pretrained/model.hpp"

std::vector<std::string> split(const std::string &s, char delimiter);

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);

    LinearModel model;

private Q_SLOTS:
    void runPrediction();
    void chooseDatasetTokens();
    void chooseDatasetModel();
    void openTrainingSettings();
    void startTraining();

private:
    QLineEdit* inputEdit;
    QPushButton* runButton;
    QPushButton* datasetButtonTokens;
    QPushButton* datasetButtonModel;
    QPushButton* settingsButton;
    QPushButton* startTrainingButton;
    QLabel* datasetLabelTokens;
    QLabel* datasetLabelModel;

    ProbabilityChart* chart;

    // ML components
    Tokenizer tokenizer;
    Embedding embedding;

    // settings
    int maxTokens = 128;
    bool trainTokenizerAndEmbed = false;
    bool trainModel = false;
};
