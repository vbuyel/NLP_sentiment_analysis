#ifndef STRUCTURE_HPP
#define STRUCTURE_HPP

#include <thread>
#include <mutex>
#include <map>
#include <vector>
#include <fstream>
#include <Eigen/Dense>
#include "main_types.hpp"

// Token dictionary size in the beginning = 94
// Max amount of tokens: 12000 -> Tokenization time: 0h 30min
//                             -> Embedding time: 3h 22min
#define MAX_TOKENS_AMOUNT 2500
#define AMOUNT_ROWS 125
#define MAX_EPOCH_AMOUNT 50
#define LEARNING_RATE 0.0003

extern std::string SYSTEM_FILE_PATH_TOKENS;
extern std::string SYSTEM_FILE_PATH_MODEL_TRAIN;

extern size_t NUM_THREADS;
extern std::mutex train_mutex;

extern std::map<KEY, VALUE> tokens;
extern std::map<KEY, VALUE> temp_tokens;

extern Eigen::MatrixXf tokens_matrix;
extern std::vector<std::vector<float>> layer1_weights;
extern std::vector<std::vector<float>> layer2_weights;

extern std::vector<std::vector<float>> model_weights;

#endif
