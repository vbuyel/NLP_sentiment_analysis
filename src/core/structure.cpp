#include <string>
#include <map>

#include "structure.hpp"
#include "main_types.hpp"

std::mutex train_mutex;

size_t NUM_THREADS = 1;
std::map<KEY, VALUE> tokens;
std::map<KEY, VALUE> temp_tokens;

Eigen::MatrixXf tokens_matrix;
std::vector<std::vector<float>> layer1_weights;
std::vector<std::vector<float>> layer2_weights;

std::string SYSTEM_FILE_PATH_MODEL_TRAIN = "";
std::string SYSTEM_FILE_PATH_TOKENS = "";
