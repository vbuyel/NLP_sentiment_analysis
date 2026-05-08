#include "probabilitychart.hpp"

ProbabilityChart::ProbabilityChart(QWidget* parent)
    : QChartView(parent),
      barSet(new QBarSet("Probability")),
      series(new QBarSeries()),
      chart(new QChart()) {

    *barSet << 0 << 0 << 0;

    series->append(barSet);

    chart->addSeries(series);
    chart->setTitle("Sentiment Probability");
    chart->setAnimationOptions(QChart::SeriesAnimations);

    QStringList categories;
    categories << "Negative" << "Neutral" << "Positive";

    QBarCategoryAxis* axisX = new QBarCategoryAxis();
    axisX->append(categories);
    chart->addAxis(axisX, Qt::AlignBottom);
    series->attachAxis(axisX);

    QValueAxis* axisY = new QValueAxis();
    axisY->setRange(0.0, 1.0);
    axisY->setTitleText("Probability");
    chart->addAxis(axisY, Qt::AlignLeft);
    series->attachAxis(axisY);

    setChart(chart);
    setRenderHint(QPainter::Antialiasing);
}

void ProbabilityChart::updateProbabilities(float positive, float neutral, float negative) {
    barSet->replace(0, positive);
    barSet->replace(1, neutral);
    barSet->replace(2, negative);
}
