#pragma once
#include <QDialog>
#include <QCheckBox>
#include <QSpinBox>
#include <QLabel>

class TrainingSettingsDialog : public QDialog {
    Q_OBJECT

public:
    explicit TrainingSettingsDialog(QWidget* parent = nullptr);

    int maxTokens() const;
    bool trainTokenizerAndEmbed() const;
    bool trainModel() const;

private:
    QLabel* labelTokensMax;
    QSpinBox* tokenSpin;
    QCheckBox* tokenizerAndEmbedCheck;
    QCheckBox* modelCheck;
};
