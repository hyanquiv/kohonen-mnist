#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "som.hpp"

class SOMVisualizer {
public:
    explicit SOMVisualizer(Kohonen3D* somPtr);
    ~SOMVisualizer();

    bool initGL();
    void run(MNISTDataset& dataset);

private:
    void setupShaders();
    void setupBuffers();
    void renderPattern(int x, int y, int z);
    void renderSurface();

    GLFWwindow* m_window;
    Kohonen3D* m_som;
    GLuint m_shaderProgram;
    GLuint m_VAO, m_VBO;
    glm::mat4 m_projection;
    float m_rotationAngle;
};
