#include "visualizer.hpp"

#include <iostream>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

SOMVisualizer::SOMVisualizer(Kohonen3D* somPtr)
    : m_window(nullptr), m_som(somPtr), m_shaderProgram(0), m_VAO(0), m_VBO(0), m_rotationAngle(0.0f)
{
}

SOMVisualizer::~SOMVisualizer() {
}

bool SOMVisualizer::initGL() {
    if (!glfwInit()) return false;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    m_window = glfwCreateWindow(1200, 900, "SOM 3D - MNIST", NULL, NULL);
    if (!m_window) {
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(m_window);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) return false;

    setupShaders();
    setupBuffers();
    glEnable(GL_DEPTH_TEST);

    m_projection = glm::perspective(glm::radians(45.0f), 1200.0f / 900.0f, 0.1f, 100.0f);

    return true;
}

void SOMVisualizer::setupShaders() {
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

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

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
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cerr << "Fragment shader compilation failed: " << infoLog << std::endl;
    }

    m_shaderProgram = glCreateProgram();
    glAttachShader(m_shaderProgram, vertexShader);
    glAttachShader(m_shaderProgram, fragmentShader);
    glLinkProgram(m_shaderProgram);
    glGetProgramiv(m_shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(m_shaderProgram, 512, NULL, infoLog);
        std::cerr << "Shader program linking failed: " << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void SOMVisualizer::setupBuffers() {
    float vertices[] = {
        -0.5f, -0.5f, 0.0f,  0.0f, 0.0f,
         0.5f, -0.5f, 0.0f,  1.0f, 0.0f,
         0.5f,  0.5f, 0.0f,  1.0f, 1.0f,
        -0.5f,  0.5f, 0.0f,  0.0f, 1.0f
    };

    unsigned int indices[] = { 0,1,2, 2,3,0 };

    glGenVertexArrays(1, &m_VAO);
    glGenBuffers(1, &m_VBO);
    GLuint EBO;
    glGenBuffers(1, &EBO);

    glBindVertexArray(m_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void SOMVisualizer::renderPattern(int x, int y, int z) {
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    std::vector<unsigned char> imageData(28 * 28);
    const auto& weights = m_som->getWeights();
    for (int i = 0; i < 28 * 28; ++i) {
        imageData[i] = static_cast<unsigned char>(weights[x][y][z][i] * 255);
    }

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, 28, 28, 0, GL_RED, GL_UNSIGNED_BYTE, imageData.data());
    glGenerateMipmap(GL_TEXTURE_2D);

    glUseProgram(m_shaderProgram);
    glBindVertexArray(m_VAO);
    glBindTexture(GL_TEXTURE_2D, texture);

    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(x, y, z));
    model = glm::scale(model, glm::vec3(0.8f, 0.8f, 0.8f));

    GLint modelLoc = glGetUniformLocation(m_shaderProgram, "model");
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    glDeleteTextures(1, &texture);
}

void SOMVisualizer::renderSurface() {
    float cameraDistance = SOM_SIZE * 2.5f;
    float camX = sin(m_rotationAngle) * cameraDistance;
    float camZ = cos(m_rotationAngle) * cameraDistance;

    glm::mat4 view = glm::lookAt(
        glm::vec3(camX, cameraDistance * 0.7f, camZ),
        glm::vec3(SOM_SIZE/2.0f, SOM_SIZE/2.0f, SOM_SIZE/2.0f),
        glm::vec3(0.0f, 1.0f, 0.0f)
    );

    glClearColor(0.0f,0.0f,0.0f,1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(m_shaderProgram);
    GLint viewLoc = glGetUniformLocation(m_shaderProgram, "view");
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
    GLint projLoc = glGetUniformLocation(m_shaderProgram, "projection");
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(m_projection));

    for (int x = 0; x < SOM_SIZE; x += SOM_SIZE-1) {
        for (int y = 0; y < SOM_SIZE; ++y) {
            for (int z = 0; z < SOM_SIZE; ++z) {
                renderPattern(x,y,z);
            }
        }
    }

    for (int y = 0; y < SOM_SIZE; y += SOM_SIZE-1) {
        for (int x = 1; x < SOM_SIZE-1; ++x) {
            for (int z = 0; z < SOM_SIZE; ++z) {
                renderPattern(x,y,z);
            }
        }
    }

    for (int z = 0; z < SOM_SIZE; z += SOM_SIZE-1) {
        for (int x = 1; x < SOM_SIZE-1; ++x) {
            for (int y = 1; y < SOM_SIZE-1; ++y) {
                renderPattern(x,y,z);
            }
        }
    }
}

void SOMVisualizer::run(MNISTDataset& dataset) {
    m_som->initialize();

    if (!m_som->weightsLoaded()) {
        m_som->train(dataset);
    }

    m_som->evaluate(dataset);

    if (!initGL()) {
        std::cerr << "Error al inicializar OpenGL" << std::endl;
        return;
    }

    while (!glfwWindowShouldClose(m_window)) {
        m_rotationAngle += 0.005f;

        renderSurface();
        glfwSwapBuffers(m_window);
        glfwPollEvents();

        if (glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(m_window, true);
        }
    }

    glDeleteVertexArrays(1, &m_VAO);
    glDeleteBuffers(1, &m_VBO);
    glfwTerminate();
}
