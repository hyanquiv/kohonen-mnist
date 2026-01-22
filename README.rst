Kohonen + OpenGL + MNIST
========================

Aplicación en C++ que carga el dataset MNIST, entrena (o carga) una red Kohonen 3D (Self-Organizing Map)
y visualiza la superficie externa del cubo SOM en una ventana OpenGL usando GLFW, GLEW y GLM.

Qué hace el programa
---------------------

- Carga el dataset MNIST mediante `mnist::read_dataset`.
- Inicializa una red Kohonen 3D de dimensión configurable (`SOM_SIZE`), con vectores de entrada de 28x28 (784) píxeles.
- Intenta cargar pesos preentrenados desde `resultados/som_weights.bin`; si no existen, entrena la red con parámetros definidos en el código (`EPOCHS`, `SAMPLES`, etc.) y guarda los pesos.
- Evalúa de forma simplificada el modelo sobre un subconjunto de test (`TEST_SAMPLES`) y muestra una métrica de precisión muy básica (umbral simple).
- Inicia una ventana OpenGL y renderiza solo la superficie del cubo SOM: cada neurona de la superficie se muestra como un plano texturizado con su patrón 28x28 (los pesos de la neurona).
- La visualización orbita automáticamente; presione ESC para cerrar la ventana.

Archivos importantes
-------------------

- Código de ejemplo y entrada principal: example/main.cpp
- Cargador MNIST: include/mnist/
- Archivos del dataset (opcionales en el repositorio): `train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, `t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte`.

Dependencias y compilación
--------------------------

El proyecto está pensado para compilarse con CMake y gestionarse con `vcpkg` en Windows (Visual Studio), aunque CMake funciona en otras plataformas si se satisfacen las dependencias.

Dependencias principales (instalar vía vcpkg o el gestor que prefiera):

- `glfw3`
- `glew`
- `glm`

Ejemplo rápido con `vcpkg` (Windows PowerShell):

```powershell
git clone https://github.com/microsoft/vcpkg.git
.\vcpkg\bootstrap-vcpkg.bat
.\vcpkg\vcpkg install glfw3 glew glm

mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=..\vcpkg\scripts\buildsystems\vcpkg.cmake
cmake --build . --config Release
```

Ejecución
---------

- Asegúrese de que los ficheros del dataset MNIST estén accesibles para el programa. El proyecto puede usar la macro `MNIST_DATA_LOCATION` para localizar los archivos; por simplicidad coloque los ficheros `*-idx*` en la raíz del repositorio donde están incluidos en este proyecto.
- Ejecute el binario generado (por ejemplo `example.exe` o el ejecutable que cree CMake). Al arrancar mostrará información de carga, entrenará si no hay pesos guardados, evaluará y abrirá la ventana de visualización.

Notas
-----

- La evaluación incluida en el ejemplo es muy simplificada y sirve solo como demostración; en una versión completa debe asignarse etiquetas a neuronas y realizarse métricas apropiadas.
- El tamaño y los parámetros del SOM se definen como constantes en `example/main.cpp`.

Fork
----

Este proyecto fue bifurcado desde: https://github.com/wichtounet/mnist
