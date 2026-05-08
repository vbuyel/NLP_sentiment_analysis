#pragma once

#include <QtCharts/QChartView>
#include <QtCharts/QBarSeries>
#include <QtCharts/QBarSet>
#include <QtCharts/QValueAxis>
#include <QtCharts/QBarCategoryAxis>

class ProbabilityChart : public QChartView {
    Q_OBJECT

public:
    explicit ProbabilityChart(QWidget* parent = nullptr);

    void updateProbabilities(float positive, float neutral, float negative);

private:
    QBarSet* barSet;
    QBarSeries* series;
    QChart* chart;
};
