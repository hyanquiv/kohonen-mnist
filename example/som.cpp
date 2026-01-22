#include "som.hpp"

#include <random>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cfloat>

namespace fs = std::filesystem;

Kohonen3D::Kohonen3D()
    : m_totalNeurons(0), m_surfaceNeurons(0), m_trainingCompleted(false), m_weightsLoaded(false)
{
    m_weights.resize(SOM_SIZE,
        std::vector<std::vector<std::vector<float>>>(
            SOM_SIZE,
            std::vector<std::vector<float>>(
                SOM_SIZE,
                std::vector<float>(INPUT_SIZE)
            )
        )
    );
}

void Kohonen3D::initialize() {
    m_totalNeurons = SOM_SIZE * SOM_SIZE * SOM_SIZE;
    m_surfaceNeurons = 6 * SOM_SIZE * SOM_SIZE - 12 * SOM_SIZE + 8;

    if (!fs::exists(RESULT_DIR)) {
        fs::create_directory(RESULT_DIR);
    }

    std::string weightsFile = RESULT_DIR + "/som_weights.bin";
    if (fs::exists(weightsFile)) {
        std::cout << "Cargando pesos preentrenados..." << std::endl;
        if (loadWeights(weightsFile)) {
            m_weightsLoaded = true;
            std::cout << "Pesos cargados exitosamente!" << std::endl;
            return;
        } else {
            std::cerr << "Error al cargar pesos. Se procederá a entrenar." << std::endl;
        }
    }

    m_weightsLoaded = false;
}

bool Kohonen3D::loadWeights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return false;

    int savedSize;
    file.read(reinterpret_cast<char*>(&savedSize), sizeof(savedSize));

    if (savedSize != SOM_SIZE) {
        std::cerr << "Tamaño de SOM incompatible: " << savedSize << " vs " << SOM_SIZE << std::endl;
        return false;
    }

    for (int x = 0; x < SOM_SIZE; ++x) {
        for (int y = 0; y < SOM_SIZE; ++y) {
            for (int z = 0; z < SOM_SIZE; ++z) {
                for (int i = 0; i < INPUT_SIZE; ++i) {
                    file.read(reinterpret_cast<char*>(&m_weights[x][y][z][i]), sizeof(float));
                }
            }
        }
    }

    return file.good();
}

void Kohonen3D::saveWeights() {
    std::ofstream file(RESULT_DIR + "/som_weights.bin", std::ios::binary);
    if (!file) {
        std::cerr << "Error al abrir archivo para guardar pesos" << std::endl;
        return;
    }

    file.write(reinterpret_cast<const char*>(&SOM_SIZE), sizeof(SOM_SIZE));

    for (int x = 0; x < SOM_SIZE; ++x) {
        for (int y = 0; y < SOM_SIZE; ++y) {
            for (int z = 0; z < SOM_SIZE; ++z) {
                for (int i = 0; i < INPUT_SIZE; ++i) {
                    file.write(reinterpret_cast<const char*>(&m_weights[x][y][z][i]), sizeof(float));
                }
            }
        }
    }
}

void Kohonen3D::train(MNISTDataset& dataset) {
    if (m_weightsLoaded) {
        std::cout << "Usando pesos preentrenados. Saltando entrenamiento." << std::endl;
        m_trainingCompleted = true;
        return;
    }

    std::cout << "\n=== INICIANDO ENTRENAMIENTO DE RED KOHONEN 3D ===" << std::endl;
    std::cout << "Tamaño del SOM: " << SOM_SIZE << "x" << SOM_SIZE << "x" << SOM_SIZE << std::endl;
    std::cout << "Neuronas totales: " << m_totalNeurons << std::endl;
    std::cout << "Dimensiones de entrada: " << INPUT_SIZE << " (28x28)" << std::endl;
    std::cout << "Épocas: " << EPOCHS << " | Muestras por época: " << SAMPLES << std::endl;
    std::cout << "Inicializando pesos... ";

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int x = 0; x < SOM_SIZE; ++x) {
        for (int y = 0; y < SOM_SIZE; ++y) {
            for (int z = 0; z < SOM_SIZE; ++z) {
                for (int i = 0; i < INPUT_SIZE; ++i) {
                    m_weights[x][y][z][i] = dis(gen);
                }
            }
        }
    }
    std::cout << "COMPLETADO\n" << std::endl;

    const float initialLR = 0.3f;
    const float initialRadius = SOM_SIZE / 2.0f;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        float learningRate = initialLR * exp(-(float)epoch / EPOCHS);
        float radius = initialRadius * exp(-(float)epoch / EPOCHS);

        std::vector<int> indices(SAMPLES);
        for (int i = 0; i < SAMPLES; ++i) {
            indices[i] = rand() % dataset.training_images.size();
        }

        float epochDist = 0.0f;

        for (int i = 0; i < SAMPLES; ++i) {
            int sampleIdx = indices[i];
            std::vector<float> sample(INPUT_SIZE);
            for (int j = 0; j < INPUT_SIZE; ++j) {
                sample[j] = dataset.training_images[sampleIdx][j] / 255.0f;
            }

            int bmuX = 0, bmuY = 0, bmuZ = 0;
            float minDist = FLT_MAX;

            for (int x = 0; x < SOM_SIZE; ++x) {
                for (int y = 0; y < SOM_SIZE; ++y) {
                    for (int z = 0; z < SOM_SIZE; ++z) {
                        float dist = 0.0f;
                        for (int k = 0; k < INPUT_SIZE; ++k) {
                            float diff = sample[k] - m_weights[x][y][z][k];
                            dist += diff * diff;
                        }

                        if (dist < minDist) {
                            minDist = dist;
                            bmuX = x; bmuY = y; bmuZ = z;
                        }
                    }
                }
            }

            epochDist += minDist;

            for (int x = 0; x < SOM_SIZE; ++x) {
                for (int y = 0; y < SOM_SIZE; ++y) {
                    for (int z = 0; z < SOM_SIZE; ++z) {
                        float d = sqrt(
                            pow(x - bmuX, 2) +
                            pow(y - bmuY, 2) +
                            pow(z - bmuZ, 2)
                        );

                        if (d <= radius) {
                            float influence = exp(-(d * d) / (2 * radius * radius));
                            for (int k = 0; k < INPUT_SIZE; ++k) {
                                m_weights[x][y][z][k] += learningRate * influence * (sample[k] - m_weights[x][y][z][k]);
                            }
                        }
                    }
                }
            }
        }

        epochDist /= SAMPLES;
        float progress = (epoch + 1) * 100.0f / EPOCHS;

        std::cout << "Época " << std::setw(3) << epoch + 1 << "/" << EPOCHS;
        std::cout << " | Progreso: " << std::fixed << std::setprecision(1) << progress << "%";
        std::cout << " | Tasa: " << std::scientific << learningRate;
        std::cout << " | Radio: " << std::fixed << std::setprecision(2) << radius;
        std::cout << " | Dist: " << std::fixed << std::setprecision(4) << epochDist << std::endl;
    }

    saveWeights();
    m_trainingCompleted = true;

    std::cout << "\nENTRENAMIENTO COMPLETADO EXITOSAMENTE!" << std::endl;
    std::cout << "Pesos guardados en: " << RESULT_DIR << "/som_weights.bin" << std::endl;
}

void Kohonen3D::evaluate(MNISTDataset& dataset) {
    if (!m_trainingCompleted && !m_weightsLoaded) {
        std::cerr << "El modelo no está entrenado. No se puede evaluar." << std::endl;
        return;
    }

    std::cout << "\n=== EVALUANDO RENDIMIENTO ===" << std::endl;
    std::cout << "Muestras de prueba: " << TEST_SAMPLES << std::endl;

    int correct = 0;
    int neuronActivations[SOM_SIZE][SOM_SIZE][SOM_SIZE] = {{{0}}};

    for (int i = 0; i < TEST_SAMPLES; ++i) {
        int sampleIdx = rand() % dataset.test_images.size();
        std::vector<float> sample(INPUT_SIZE);
        for (int j = 0; j < INPUT_SIZE; ++j) {
            sample[j] = dataset.test_images[sampleIdx][j] / 255.0f;
        }

        int bmuX = 0, bmuY = 0, bmuZ = 0;
        float minDist = FLT_MAX;

        for (int x = 0; x < SOM_SIZE; ++x) {
            for (int y = 0; y < SOM_SIZE; ++y) {
                for (int z = 0; z < SOM_SIZE; ++z) {
                    float dist = 0.0f;
                    for (int k = 0; k < INPUT_SIZE; ++k) {
                        float diff = sample[k] - m_weights[x][y][z][k];
                        dist += diff * diff;
                    }

                    if (dist < minDist) {
                        minDist = dist;
                        bmuX = x; bmuY = y; bmuZ = z;
                    }
                }
            }
        }

        neuronActivations[bmuX][bmuY][bmuZ]++;
        if (minDist < 100.0f) {
            correct++;
        }
    }

    double accuracy = static_cast<double>(correct) / TEST_SAMPLES * 100.0;
    std::cout << "Precisión: " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
    std::cout << "Activaciones por neurona:" << std::endl;

    for (int x = 0; x < SOM_SIZE; ++x) {
        for (int y = 0; y < SOM_SIZE; ++y) {
            for (int z = 0; z < SOM_SIZE; ++z) {
                if (neuronActivations[x][y][z] > 0) {
                    std::cout << "Neurona (" << x << "," << y << "," << z << "): "
                              << neuronActivations[x][y][z] << " activaciones" << std::endl;
                }
            }
        }
    }
}

const std::vector<std::vector<std::vector<std::vector<float>>>>& Kohonen3D::getWeights() const {
    return m_weights;
}
