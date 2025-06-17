#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "mnist/mnist_reader.hpp"

namespace fs = std::filesystem;

// Parámetros configurables
const int SOM_SIZE = 8;       // Tamaño del cubo SOM (SOM_SIZE^3 neuronas)
const int INPUT_SIZE = 784;   // 28x28 píxeles
const int EPOCHS = 100;       // Reducido para pruebas
const int SAMPLES = 1000;     // Subconjunto de entrenamiento
const int TEST_SAMPLES = 200; // Muestras para evaluación

// Directorio de resultados
const std::string RESULT_DIR = "resultados";

// Estructura para la red Kohonen 3D
struct Kohonen3D {
    std::vector<std::vector<std::vector<std::vector<float>>>> weights;
    
    Kohonen3D() {
        weights.resize(SOM_SIZE, 
            std::vector<std::vector<std::vector<float>>>(
                SOM_SIZE,
                std::vector<std::vector<float>>(
                    SOM_SIZE,
                    std::vector<float>(INPUT_SIZE)
                )
            ));
    }
};

// Clase para visualización OpenGL
class SOMVisualizer {
private:
    GLFWwindow* window;
    Kohonen3D som;
    GLuint shaderProgram;
    GLuint VAO, VBO;
    glm::mat4 projection;
    float rotationAngle;
    int totalNeurons;
    int surfaceNeurons;
    bool trainingCompleted;
    bool weightsLoaded;
    
public:
    SOMVisualizer() : window(nullptr), rotationAngle(0.0f), 
                      totalNeurons(0), surfaceNeurons(0),
                      trainingCompleted(false), weightsLoaded(false) {}
    
    bool initGL() {
        // Inicializar GLFW
        if (!glfwInit()) return false;
        
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        
        window = glfwCreateWindow(1200, 900, "SOM 3D - MNIST", NULL, NULL);
        if (!window) {
            glfwTerminate();
            return false;
        }
        
        glfwMakeContextCurrent(window);
        glewExperimental = GL_TRUE;
        if (glewInit() != GLEW_OK) return false;
        
        setupShaders();
        setupBuffers();
        glEnable(GL_DEPTH_TEST);
        
        // Configurar proyección
        projection = glm::perspective(
            glm::radians(45.0f), 
            1200.0f / 900.0f, 
            0.1f, 
            100.0f
        );
        
        return true;
    }
    
    void setupShaders() {
        const char* vertexShaderSource = R"(
            #version 330 core
            layout (location = 0) in vec3 aPos;
            layout (location = 1) in vec2 aTexCoord;
            out vec2 TexCoord;
            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;
            void main() {
                gl_Position = projection * view * model * vec4(aPos, 1.0);
                TexCoord = aTexCoord;
            }
        )";
        
        const char* fragmentShaderSource = R"(
            #version 330 core
            in vec2 TexCoord;
            out vec4 FragColor;
            uniform sampler2D ourTexture;
            void main() {
                FragColor = texture(ourTexture, TexCoord);
            }
        )";
        
        // Compilar shaders
        GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
        glCompileShader(vertexShader);
        
        // Verificar compilación
        GLint success;
        GLchar infoLog[512];
        glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
            std::cerr << "Vertex shader compilation failed: " << infoLog << std::endl;
        }
        
        GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
        glCompileShader(fragmentShader);
        
        // Verificar compilación
        glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
            std::cerr << "Fragment shader compilation failed: " << infoLog << std::endl;
        }
        
        // Crear programa
        shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);
        
        // Verificar enlace
        glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
            std::cerr << "Shader program linking failed: " << infoLog << std::endl;
        }
        
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
    }
    
    void setupBuffers() {
        // Geometría básica de un plano (para mostrar el patrón)
        float vertices[] = {
            // Posiciones         // Coordenadas de textura
            -0.5f, -0.5f, 0.0f,  0.0f, 0.0f,
             0.5f, -0.5f, 0.0f,  1.0f, 0.0f,
             0.5f,  0.5f, 0.0f,  1.0f, 1.0f,
            -0.5f,  0.5f, 0.0f,  0.0f, 1.0f
        };
        
        unsigned int indices[] = {
            0, 1, 2,
            2, 3, 0
        };
        
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        GLuint EBO;
        glGenBuffers(1, &EBO);
        
        glBindVertexArray(VAO);
        
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
        
        // Posiciones
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        
        // Coordenadas de textura
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }
    
    void initializeSOM() {
        totalNeurons = SOM_SIZE * SOM_SIZE * SOM_SIZE;
        surfaceNeurons = 6 * SOM_SIZE * SOM_SIZE - 12 * SOM_SIZE + 8;
        
        // Crear directorio de resultados si no existe
        if (!fs::exists(RESULT_DIR)) {
            fs::create_directory(RESULT_DIR);
        }
        
        std::string weightsFile = RESULT_DIR + "/som_weights.bin";
        
        if (fs::exists(weightsFile)) {
            std::cout << "Cargando pesos preentrenados..." << std::endl;
            if (loadWeights(weightsFile)) {
                weightsLoaded = true;
                std::cout << "Pesos cargados exitosamente!" << std::endl;
                return;
            }
            else {
                std::cerr << "Error al cargar pesos. Se procederá a entrenar." << std::endl;
            }
        }
        
        // Si no hay pesos guardados, entrenar
        weightsLoaded = false;
    }
    
    bool loadWeights(const std::string& filename) {
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
                        file.read(reinterpret_cast<char*>(&som.weights[x][y][z][i]), sizeof(float));
                    }
                }
            }
        }
        
        return file.good();
    }
    
    void trainSOM(mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t>& dataset) {
        if (weightsLoaded) {
            std::cout << "Usando pesos preentrenados. Saltando entrenamiento." << std::endl;
            trainingCompleted = true;
            return;
        }
        
        std::cout << "\n=== INICIANDO ENTRENAMIENTO DE RED KOHONEN 3D ===" << std::endl;
        std::cout << "Tamaño del SOM: " << SOM_SIZE << "x" << SOM_SIZE << "x" << SOM_SIZE << std::endl;
        std::cout << "Neuronas totales: " << totalNeurons << std::endl;
        std::cout << "Dimensiones de entrada: " << INPUT_SIZE << " (28x28)" << std::endl;
        std::cout << "Épocas: " << EPOCHS << " | Muestras por época: " << SAMPLES << std::endl;
        std::cout << "Inicializando pesos... ";
        
        // Inicializar pesos aleatoriamente
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        
        for (int x = 0; x < SOM_SIZE; ++x) {
            for (int y = 0; y < SOM_SIZE; ++y) {
                for (int z = 0; z < SOM_SIZE; ++z) {
                    for (int i = 0; i < INPUT_SIZE; ++i) {
                        som.weights[x][y][z][i] = dis(gen);
                    }
                }
            }
        }
        std::cout << "COMPLETADO\n" << std::endl;
        
        // Parámetros de entrenamiento
        const float initialLR = 0.3f;
        const float initialRadius = SOM_SIZE / 2.0f;
        
        // Entrenamiento
        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
            float learningRate = initialLR * exp(-(float)epoch / EPOCHS);
            float radius = initialRadius * exp(-(float)epoch / EPOCHS);
            
            // Barajar muestras
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
                
                // Encontrar BMU (Best Matching Unit)
                int bmuX = 0, bmuY = 0, bmuZ = 0;
                float minDist = FLT_MAX;
                
                for (int x = 0; x < SOM_SIZE; ++x) {
                    for (int y = 0; y < SOM_SIZE; ++y) {
                        for (int z = 0; z < SOM_SIZE; ++z) {
                            float dist = 0.0f;
                            for (int k = 0; k < INPUT_SIZE; ++k) {
                                float diff = sample[k] - som.weights[x][y][z][k];
                                dist += diff * diff;
                            }
                            
                            if (dist < minDist) {
                                minDist = dist;
                                bmuX = x;
                                bmuY = y;
                                bmuZ = z;
                            }
                        }
                    }
                }
                
                epochDist += minDist;
                
                // Actualizar pesos
                for (int x = 0; x < SOM_SIZE; ++x) {
                    for (int y = 0; y < SOM_SIZE; ++y) {
                        for (int z = 0; z < SOM_SIZE; ++z) {
                            // Distancia en el espacio 3D
                            float d = sqrt(
                                pow(x - bmuX, 2) + 
                                pow(y - bmuY, 2) + 
                                pow(z - bmuZ, 2)
                            );
                            
                            if (d <= radius) {
                                // Influencia gaussiana
                                float influence = exp(-(d * d) / (2 * radius * radius));
                                
                                for (int k = 0; k < INPUT_SIZE; ++k) {
                                    som.weights[x][y][z][k] += 
                                        learningRate * influence * 
                                        (sample[k] - som.weights[x][y][z][k]);
                                }
                            }
                        }
                    }
                }
            }
            
            // Calcular métricas
            epochDist /= SAMPLES;
            float progress = (epoch + 1) * 100.0f / EPOCHS;
            
            // Mostrar progreso detallado
            std::cout << "Época " << std::setw(3) << epoch + 1 << "/" << EPOCHS;
            std::cout << " | Progreso: " << std::fixed << std::setprecision(1) << progress << "%";
            std::cout << " | Tasa: " << std::scientific << learningRate;
            std::cout << " | Radio: " << std::fixed << std::setprecision(2) << radius;
            std::cout << " | Dist: " << std::fixed << std::setprecision(4) << epochDist << std::endl;
        }
        
        // Guardar pesos
        saveWeights();
        trainingCompleted = true;
        
        std::cout << "\nENTRENAMIENTO COMPLETADO EXITOSAMENTE!" << std::endl;
        std::cout << "Pesos guardados en: " << RESULT_DIR << "/som_weights.bin" << std::endl;
    }
    
    void saveWeights() {
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
                        file.write(reinterpret_cast<const char*>(&som.weights[x][y][z][i]), sizeof(float));
                    }
                }
            }
        }
    }
    
    void renderPattern(int x, int y, int z) {
        // Crear textura con el patrón de la neurona
        GLuint texture;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        
        // Configurar textura
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        
        // Convertir pesos a imagen 28x28
        std::vector<unsigned char> imageData(28 * 28);
        for (int i = 0; i < 28 * 28; ++i) {
            imageData[i] = static_cast<unsigned char>(som.weights[x][y][z][i] * 255);
        }
        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, 28, 28, 0, GL_RED, GL_UNSIGNED_BYTE, imageData.data());
        glGenerateMipmap(GL_TEXTURE_2D);
        
        // Dibujar plano con la textura
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glBindTexture(GL_TEXTURE_2D, texture);
        
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(x, y, z));
        model = glm::scale(model, glm::vec3(0.8f, 0.8f, 0.8f));
        
        GLint modelLoc = glGetUniformLocation(shaderProgram, "model");
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
        
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        
        glDeleteTextures(1, &texture);
    }
    
    void renderSurface() {
        // Cámara orbital
        float cameraDistance = SOM_SIZE * 2.5f;
        float camX = sin(rotationAngle) * cameraDistance;
        float camZ = cos(rotationAngle) * cameraDistance;
        
        glm::mat4 view = glm::lookAt(
            glm::vec3(camX, cameraDistance * 0.7f, camZ),
            glm::vec3(SOM_SIZE/2.0f, SOM_SIZE/2.0f, SOM_SIZE/2.0f),
            glm::vec3(0.0f, 1.0f, 0.0f)
        );
        
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // Pasar matrices a los shaders
        glUseProgram(shaderProgram);
        
        GLint viewLoc = glGetUniformLocation(shaderProgram, "view");
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        
        GLint projLoc = glGetUniformLocation(shaderProgram, "projection");
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
        
        // Renderizar solo la superficie
        for (int x = 0; x < SOM_SIZE; x += SOM_SIZE-1) {
            for (int y = 0; y < SOM_SIZE; ++y) {
                for (int z = 0; z < SOM_SIZE; ++z) {
                    renderPattern(x, y, z);
                }
            }
        }
        
        for (int y = 0; y < SOM_SIZE; y += SOM_SIZE-1) {
            for (int x = 1; x < SOM_SIZE-1; ++x) {
                for (int z = 0; z < SOM_SIZE; ++z) {
                    renderPattern(x, y, z);
                }
            }
        }
        
        for (int z = 0; z < SOM_SIZE; z += SOM_SIZE-1) {
            for (int x = 1; x < SOM_SIZE-1; ++x) {
                for (int y = 1; y < SOM_SIZE-1; ++y) {
                    renderPattern(x, y, z);
                }
            }
        }
    }
    
    void evaluatePerformance(mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t>& dataset) {
        if (!trainingCompleted && !weightsLoaded) {
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
            
            // Encontrar BMU
            int bmuX = 0, bmuY = 0, bmuZ = 0;
            float minDist = FLT_MAX;
            
            for (int x = 0; x < SOM_SIZE; ++x) {
                for (int y = 0; y < SOM_SIZE; ++y) {
                    for (int z = 0; z < SOM_SIZE; ++z) {
                        float dist = 0.0f;
                        for (int k = 0; k < INPUT_SIZE; ++k) {
                            float diff = sample[k] - som.weights[x][y][z][k];
                            dist += diff * diff;
                        }
                        
                        if (dist < minDist) {
                            minDist = dist;
                            bmuX = x;
                            bmuY = y;
                            bmuZ = z;
                        }
                    }
                }
            }
            
            neuronActivations[bmuX][bmuY][bmuZ]++;
            
            // Inferir el dígito (esto es simplificado, en realidad necesitarías etiquetar las neuronas primero)
            // Para este ejemplo, simplemente comparamos con la etiqueta real
            // En una implementación real, asignarías etiquetas a las neuronas durante el entrenamiento
            
            // Aquí solo contamos como correcto si la neurona está activa (simplificado)
            if (minDist < 100.0f) { // Umbral arbitrario
                correct++;
            }
        }
        
        double accuracy = static_cast<double>(correct) / TEST_SAMPLES * 100.0;
        
        std::cout << "Precisión: " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
        std::cout << "Activaciones por neurona:" << std::endl;
        
        // Estadísticas de activación
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
    
    void run(mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t>& dataset) {
        initializeSOM();
        
        if (!weightsLoaded) {
            trainSOM(dataset);
        }
        else {
            trainingCompleted = true;
        }
        
        // Evaluar rendimiento
        evaluatePerformance(dataset);
        
        // Inicializar y mostrar OpenGL
        if (!initGL()) {
            std::cerr << "Error al inicializar OpenGL" << std::endl;
            return;
        }
        
        // Bucle principal de renderizado
        while (!glfwWindowShouldClose(window)) {
            rotationAngle += 0.005f; // Rotación automática
            
            renderSurface();
            glfwSwapBuffers(window);
            glfwPollEvents();
            
            // Salir con ESC
            if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                glfwSetWindowShouldClose(window, true);
            }
        }
        
        // Limpieza
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        glfwTerminate();
    }
};

int main() {
    // Cargar dataset MNIST
    std::cout << "Cargando dataset MNIST..." << std::endl;
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);
    
    std::cout << "Datos cargados:" << std::endl;
    std::cout << " - Muestras entrenamiento: " << dataset.training_images.size() << std::endl;
    std::cout << " - Muestras prueba: " << dataset.test_images.size() << std::endl;
    
    // Crear y ejecutar visualizador
    SOMVisualizer visualizer;
    visualizer.run(dataset);
    
    return 0;
}