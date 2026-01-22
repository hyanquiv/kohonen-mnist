#pragma once

#include <vector>
#include <string>
#include <filesystem>
#include "mnist/mnist_reader.hpp"

using MNISTDataset = mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t>;

inline constexpr int SOM_SIZE = 8;
inline constexpr int INPUT_SIZE = 784;
inline constexpr int EPOCHS = 100;
inline constexpr int SAMPLES = 1000;
inline constexpr int TEST_SAMPLES = 200;
inline const std::string RESULT_DIR = "resultados";

class Kohonen3D {
public:
    Kohonen3D();

    bool loadWeights(const std::string& filename);
    void saveWeights();
    void initialize();
    void train(MNISTDataset& dataset);
    void evaluate(MNISTDataset& dataset);

    const std::vector<std::vector<std::vector<std::vector<float>>>>& getWeights() const;

    bool weightsLoaded() const { return m_weightsLoaded; }
    bool trainingCompleted() const { return m_trainingCompleted; }

private:
    std::vector<std::vector<std::vector<std::vector<float>>>> m_weights;
    int m_totalNeurons;
    int m_surfaceNeurons;
    bool m_trainingCompleted;
    bool m_weightsLoaded;
};
