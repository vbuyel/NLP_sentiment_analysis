cmake_minimum_required(VERSION 3.20)
project(SentimentAnalysis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# ---- LibTorch ----
set(CMAKE_PREFIX_PATH "/Users/vladbuyel/Documents/Libraries/C++/libtorch")
find_package(Torch REQUIRED)

# ---- Qt & Eigen ----
find_package(Qt6 REQUIRED COMPONENTS Widgets Charts Concurrent)
find_package(Eigen3 REQUIRED)

add_executable(SentimentAnalysis
    src/gui/main.cpp
    src/gui/mainwindow.cpp
    src/gui/trainingsettingsdialog.cpp
    src/gui/probabilitychart/probabilitychart.cpp

    src/core/structure.cpp

    src/core/neural_network_functions/neural_network.cpp
    src/core/neural_network_functions/math_functions.cpp

    src/core/model/pretrained/model.cpp
    src/core/model/train/model_train_main.cpp
    src/core/tokenizer/train/tokenize_train.cpp
    src/core/tokenizer/train/tokenizer_main.cpp
    src/core/tokenizer/pretrained/tokenizer.cpp
    src/core/embedding/train/embedding_main.cpp
    src/core/embedding/train/matrix_train/main_train.cpp
    src/core/embedding/train/matrix_train/train_to_predict.cpp
    src/core/embedding/pretrained/embedding.cpp
)

target_include_directories(SentimentAnalysis PRIVATE src)

target_link_libraries(SentimentAnalysis
    Qt6::Widgets
    Qt6::Concurrent
    Qt6::Charts
    Eigen3::Eigen
    ${TORCH_LIBRARIES}
)

# ---- IMPORTANT ----
target_compile_definitions(SentimentAnalysis PRIVATE QT_NO_KEYWORDS)

# ---- macOS runtime ----
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
