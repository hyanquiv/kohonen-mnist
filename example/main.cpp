#include <iostream>
#include "som.hpp"
#include "visualizer.hpp"

int main() {
    std::cout << "Cargando dataset MNIST..." << std::endl;
    MNISTDataset dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    std::cout << "Datos cargados:" << std::endl;
    std::cout << " - Muestras entrenamiento: " << dataset.training_images.size() << std::endl;
    std::cout << " - Muestras prueba: " << dataset.test_images.size() << std::endl;

    Kohonen3D som;
    SOMVisualizer visualizer(&som);
    visualizer.run(dataset);

    return 0;
}