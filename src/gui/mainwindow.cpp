#include "mainwindow.hpp"
#include "trainingsettingsdialog.hpp"
// #include "showtrain/showModel.hpp"
#include "../core/structure.hpp"

#include <QVBoxLayout>
#include <QFileDialog>
#include <QtConcurrent>

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent), maxTokens(128) {

    QWidget* central = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(central);

    inputEdit = new QLineEdit(this);
    inputEdit->setPlaceholderText("Enter sentence...");

    runButton = new QPushButton("Run Prediction", this);
    datasetButtonTokens = new QPushButton("Choose dataset for training Tokenizer and Embedding", this);
    datasetButtonModel = new QPushButton("Choose dataset for training Model", this);
    settingsButton = new QPushButton("Training Settings", this);
    startTrainingButton = new QPushButton("Start Training", this);

    datasetLabelTokens = new QLabel("Dataset for tokens: none", this);
    datasetLabelModel = new QLabel("Dataset for model: none", this);

    chart = new ProbabilityChart(this);

    layout->addWidget(inputEdit);
    layout->addWidget(runButton);
    layout->addWidget(datasetButtonTokens);
    layout->addWidget(datasetButtonModel);
    layout->addWidget(datasetLabelTokens);
    layout->addWidget(datasetLabelModel);
    layout->addWidget(settingsButton);
    layout->addWidget(startTrainingButton);
    layout->addWidget(chart);

    setCentralWidget(central);
    setWindowTitle("Sentiment Analyzer");

    connect(runButton, &QPushButton::clicked,
            this, &MainWindow::runPrediction);

    connect(datasetButtonTokens, &QPushButton::clicked,
            this, &MainWindow::chooseDatasetTokens);

    connect(datasetButtonModel, &QPushButton::clicked,
            this, &MainWindow::chooseDatasetModel);

    connect(settingsButton, &QPushButton::clicked,
            this, &MainWindow::openTrainingSettings);

    connect(startTrainingButton, &QPushButton::clicked,
            this, &MainWindow::startTraining);
}

void MainWindow::runPrediction() {
    const std::string text = inputEdit->text().toStdString();
    if (text.empty()) {
        return;
    }

    auto user_tokens = tokenizer.tokenize(text);
    if (user_tokens.size() > static_cast<size_t>(maxTokens)) {
        user_tokens.resize(maxTokens);
    }

    Eigen::MatrixXf embeddings = embedding.embed(user_tokens);
    Eigen::Vector3f probs = model.predict(embeddings);

    chart->updateProbabilities(probs[0], probs[1], probs[2]);
}

void MainWindow::chooseDatasetTokens() {
    QString file = QFileDialog::getOpenFileName(
        this,
        "Select Dataset for Tokenizer / Embedding",
        "",
        "Text Files (*.txt)"
    );

    if (file.isEmpty()) {
        return;
    }

    SYSTEM_FILE_PATH_TOKENS = file.toStdString();
    datasetLabelTokens->setText("Tokenizer dataset: " + file);
}

void MainWindow::chooseDatasetModel() {
    QString file = QFileDialog::getOpenFileName(
        this,
        "Select Dataset for Model",
        "",
        "Text Files (*.txt)"
    );

    if (file.isEmpty()) {
        return;
    }

    SYSTEM_FILE_PATH_MODEL_TRAIN = file.toStdString();
    datasetLabelModel->setText("Model dataset: " + file);
}

void MainWindow::openTrainingSettings() {
    TrainingSettingsDialog dialog(this);

    if (dialog.exec() == QDialog::Accepted) {
        maxTokens = dialog.maxTokens();
        trainTokenizerAndEmbed = dialog.trainTokenizerAndEmbed();
        trainModel = dialog.trainModel();
    }
}

void MainWindow::startTraining() {
    if (SYSTEM_FILE_PATH_TOKENS != "" && trainTokenizerAndEmbed) {
        tokens.clear();
        for (int i = 0x20; i <= 0x7E; i++) {
            std::string key(1, static_cast<char>(i));
            tokens.insert({key, tokens.size()});
        }
        temp_tokens.clear();
        tokenizer.fit();

        tokens_matrix = Eigen::MatrixXf::Random(tokens.size(), AMOUNT_ROWS) / std::sqrt(6);
        embedding.fit();
    }

    if (SYSTEM_FILE_PATH_MODEL_TRAIN != "" && trainModel) {
        model.fit(1);
    }
}
