#include "trainingsettingsdialog.hpp"
#include <QVBoxLayout>
#include <QDialogButtonBox>

TrainingSettingsDialog::TrainingSettingsDialog(QWidget* parent) : QDialog(parent) {
    labelTokensMax = new QLabel("Maximum tokens (or words) in your input line:", this);
    
    tokenSpin = new QSpinBox();
    tokenSpin->setRange(1, 1024);
    tokenSpin->setValue(128);

    tokenizerAndEmbedCheck = new QCheckBox("Train Tokenizer and Embedding (~11 min)");
    modelCheck = new QCheckBox("Train Model (~8 min per each launch)");

    QDialogButtonBox* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->addWidget(labelTokensMax);
    layout->addWidget(tokenSpin);
    layout->addWidget(tokenizerAndEmbedCheck);
    layout->addWidget(modelCheck);
    layout->addWidget(buttons);

    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
}

int TrainingSettingsDialog::maxTokens() const { return tokenSpin->value(); }
bool TrainingSettingsDialog::trainTokenizerAndEmbed() const { return tokenizerAndEmbedCheck->isChecked(); }
bool TrainingSettingsDialog::trainModel() const { return modelCheck->isChecked(); }
