#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "mnist/mnist_reader.hpp"

// Parámetros configurables
const int SOM_SIZE = 8;       // Tamaño del cubo SOM (SOM_SIZE^3 neuronas)
const int INPUT_SIZE = 784;    // 28x28 píxeles
const int EPOCHS = 100;       // Reducido para pruebas
const int SAMPLES = 1000;     // Subconjunto de entrenamiento

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
    
public:
    SOMVisualizer() : window(nullptr), rotationAngle(0.0f), 
                      totalNeurons(0), surfaceNeurons(0) {}
    
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
            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;
            void main() {
                gl_Position = projection * view * model * vec4(aPos, 1.0);
            }
        )";
        
        const char* fragmentShaderSource = R"(
            #version 330 core
            out vec4 FragColor;
            void main() {
                FragColor = vec4(1.0, 1.0, 1.0, 1.0); // Blanco sólido
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
        // Geometría básica de un cubo (centrado en origen)
        float vertices[] = {
            // Position
            -0.4f, -0.4f, -0.4f,
             0.4f, -0.4f, -0.4f,
             0.4f,  0.4f, -0.4f,
            -0.4f,  0.4f, -0.4f,
            
            -0.4f, -0.4f,  0.4f,
             0.4f, -0.4f,  0.4f,
             0.4f,  0.4f,  0.4f,
            -0.4f,  0.4f,  0.4f
        };
        
        unsigned int indices[] = {
            // Cara trasera
            0, 1, 2, 2, 3, 0,
            // Cara delantera
            4, 5, 6, 6, 7, 4,
            // Cara izquierda
            0, 3, 7, 7, 4, 0,
            // Cara derecha
            1, 2, 6, 6, 5, 1,
            // Cara superior
            3, 2, 6, 6, 7, 3,
            // Cara inferior
            0, 1, 5, 5, 4, 0
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
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }
    
    void trainSOM(mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t>& dataset) {
        totalNeurons = SOM_SIZE * SOM_SIZE * SOM_SIZE;
        surfaceNeurons = 6 * SOM_SIZE * SOM_SIZE - 12 * SOM_SIZE + 8; // Fórmula para neuronas de superficie
        
        std::cout << "\n=== Iniciando entrenamiento de red Kohonen 3D ===" << std::endl;
        std::cout << "Tamaño del SOM: " << SOM_SIZE << "x" << SOM_SIZE << "x" << SOM_SIZE << std::endl;
        std::cout << "Neuronas totales: " << totalNeurons << std::endl;
        std::cout << "Neuronas de superficie: " << surfaceNeurons << std::endl;
        std::cout << "Dimensiones de entrada: " << INPUT_SIZE << " (28x28)" << std::endl;
        std::cout << "Epocas: " << EPOCHS << " | Muestras por época: " << SAMPLES << std::endl;
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
        std::cout << "Completado\n" << std::endl;
        
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
            std::cout << "Epoca " << epoch + 1 << "/" << EPOCHS;
            std::cout << " | Progreso: " << std::fixed  << progress << "%";
            std::cout << " | Tasa aprend: " << std::scientific << learningRate;
            std::cout << " | Radio: " << std::fixed << radius;
            std::cout << " | Distancia: " << std::fixed  << epochDist << std::endl;
        }
        
        std::cout << "\nEntrenamiento completado exitosamente!" << std::endl;
        std::cout << "Guardando resultados... ";
        saveResults();
        std::cout << "Completado" << std::endl;
    }
    
    void saveResults() {
        // Guardar estructura del SOM
        std::ofstream somFile("som_structure.bin", std::ios::binary);
        somFile.write(reinterpret_cast<const char*>(&SOM_SIZE), sizeof(SOM_SIZE));
        
        // Guardar pesos solo de las neuronas de superficie
        for (int x = 0; x < SOM_SIZE; x += SOM_SIZE-1) {
            for (int y = 0; y < SOM_SIZE; ++y) {
                for (int z = 0; z < SOM_SIZE; ++z) {
                    saveNeuron(somFile, x, y, z);
                }
            }
        }
        
        for (int y = 0; y < SOM_SIZE; y += SOM_SIZE-1) {
            for (int x = 1; x < SOM_SIZE-1; ++x) {
                for (int z = 0; z < SOM_SIZE; ++z) {
                    saveNeuron(somFile, x, y, z);
                }
            }
        }
        
        for (int z = 0; z < SOM_SIZE; z += SOM_SIZE-1) {
            for (int x = 1; x < SOM_SIZE-1; ++x) {
                for (int y = 1; y < SOM_SIZE-1; ++y) {
                    saveNeuron(somFile, x, y, z);
                }
            }
        }
        
        somFile.close();
    }
    
    void saveNeuron(std::ofstream& file, int x, int y, int z) {
        file.write(reinterpret_cast<const char*>(&x), sizeof(x));
        file.write(reinterpret_cast<const char*>(&y), sizeof(y));
        file.write(reinterpret_cast<const char*>(&z), sizeof(z));
        
        for (int k = 0; k < INPUT_SIZE; ++k) {
            float weight = som.weights[x][y][z][k];
            file.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
        }
    }
    
    void render() {
        // Cámara orbital
        float cameraDistance = SOM_SIZE * 2.5f;
        float camX = sin(rotationAngle) * cameraDistance;
        float camZ = cos(rotationAngle) * cameraDistance;
        
        glm::mat4 view = glm::lookAt(
            glm::vec3(camX, cameraDistance * 0.7f, camZ),
            glm::vec3(SOM_SIZE/2.0f, SOM_SIZE/2.0f, SOM_SIZE/2.0f),
            glm::vec3(0.0f, 1.0f, 0.0f)
        );
        
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Fondo negro
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        
        // Pasar matrices a los shaders
        GLint viewLoc = glGetUniformLocation(shaderProgram, "view");
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        
        GLint projLoc = glGetUniformLocation(shaderProgram, "projection");
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
        
        // Dibujar solo las neuronas de superficie
        for (int x = 0; x < SOM_SIZE; x += SOM_SIZE-1) {
            for (int y = 0; y < SOM_SIZE; ++y) {
                for (int z = 0; z < SOM_SIZE; ++z) {
                    drawNeuron(x, y, z);
                }
            }
        }
        
        for (int y = 0; y < SOM_SIZE; y += SOM_SIZE-1) {
            for (int x = 1; x < SOM_SIZE-1; ++x) {
                for (int z = 0; z < SOM_SIZE; ++z) {
                    drawNeuron(x, y, z);
                }
            }
        }
        
        for (int z = 0; z < SOM_SIZE; z += SOM_SIZE-1) {
            for (int x = 1; x < SOM_SIZE-1; ++x) {
                for (int y = 1; y < SOM_SIZE-1; ++y) {
                    drawNeuron(x, y, z);
                }
            }
        }
        
        glBindVertexArray(0);
        glfwSwapBuffers(window);
    }
    
    void drawNeuron(int x, int y, int z) {
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(x, y, z));
        
        GLint modelLoc = glGetUniformLocation(shaderProgram, "model");
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
        
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
    }
    
    void run(mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t>& dataset) {
        // Primero entrenar (sin abrir ventana)
        trainSOM(dataset);
        
        // Luego inicializar y mostrar OpenGL
        if (!initGL()) {
            std::cerr << "Error al inicializar OpenGL" << std::endl;
            return;
        }
        
        // Bucle principal de renderizado
        while (!glfwWindowShouldClose(window)) {
            rotationAngle += 0.005f; // Rotación automática
            
            render();
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
    
    std::cout << "Datos cargados: " << dataset.training_images.size() << " muestras de entrenamiento" << std::endl;
    
    // Preprocesamiento (opcional)
    // binarize_dataset(dataset);
    // normalize_dataset(dataset);
    
    // Crear y ejecutar visualizador
    SOMVisualizer visualizer;
    visualizer.run(dataset);
    
    return 0;
}