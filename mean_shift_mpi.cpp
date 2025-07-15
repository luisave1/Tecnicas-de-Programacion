//=======================MEAN SHIFT CON MPI====================
//==============ALGORITMOS PARALELOS DISTRIBUIDOS===========
//============L. Fernando Cc.==============================



#include <mpi.h> // Para paralelizar el trabajo con múltiples procesadores
#include <opencv2/opencv.hpp> // Para procesar y manipular imágenes
#include <iostream> 
#include <cmath> 
#include <vector> 

using namespace std; 
using namespace cv; 

// --- Constantes del Algoritmo Mean Shift ---
const float hs = 8.0f; // Radio espacial para buscar vecinos
const float hr = 16.0f; // Radio de color para buscar vecinos
const int maxIter = 5; // Máximo de veces que un píxel se "mueve"
const float tol_color = 0.3f; // Tolerancia de cambio de color para detener el movimiento
const float tol_spatial = 0.3f; // Tolerancia de cambio de posición para detener el movimiento

// --- Estructura para representar un píxel en 5 dimensiones (posición + color Lab) ---
struct Point5D {
    float x, y, l, a, b; // x,y: coordenadas; l,a,b: componentes de color Lab

    Point5D() : x(0), y(0), l(0), a(0), b(0) {} // Constructor vacío
    Point5D(float x_, float y_, float l_, float a_, float b_)
        : x(x_), y(y_), l(l_), a(a_), b(b_) {} // Constructor con valores

    // Calcula la distancia entre colores de dos píxeles
    float colorDist(const Point5D& p) const {
        return sqrt((l - p.l) * (l - p.l) + (a - p.a) * (a - p.a) + (b - p.b) * (b - p.b));
    }

    // Calcula la distancia entre posiciones de dos píxeles
    float spatialDist(const Point5D& p) const {
        return sqrt((x - p.x) * (x - p.x) + (y - p.y) * (y - p.y));
    }

    // Suma dos puntos 5D (suma componente a componente)
    Point5D operator+(const Point5D& p) const {
        return Point5D(x + p.x, y + p.y, l + p.l, a + p.a, b + p.b);
    }

    // Divide un punto 5D por un número (divide componente a componente)
    Point5D operator/(float val) const {
        return Point5D(x / val, y / val, l / val, a / val, b / val);
    }
};

// Convierte un píxel de una imagen Lab de OpenCV a un Point5D
Point5D getPoint5D(int i, int j, const Mat& labImg) {
    Vec3b color = labImg.at<Vec3b>(i, j); // Obtiene el color BGR del píxel
    return Point5D((float)j, (float)i, // Transforma a formato Lab (L, a, b)
        color[0] * 100.0f / 255.0f,
        (float)color[1] - 128.0f,
        (float)color[2] - 128.0f);
}

// --- Función principal del algoritmo Mean Shift, paralelizado con MPI ---
void meanShiftMPI(Mat& labImg, int rank, int size) {
    int rows = labImg.rows, cols = labImg.cols; // Filas y columnas de la imagen
    int chunk = rows / size; // Número de filas que procesa cada proceso MPI
    int startRow = rank * chunk; // Fila donde empieza este proceso
    int endRow = (rank == size - 1) ? rows : startRow + chunk; // Fila donde termina este proceso

    Mat resultChunk = labImg.clone(); // Copia de la imagen para guardar resultados de este proceso

    // Bucle para cada píxel asignado a este proceso
    for (int y = startRow; y < endRow; ++y) {
        for (int x = 0; x < cols; ++x) {
            Point5D current = getPoint5D(y, x, labImg); // Punto actual (píxel a procesar)
            Point5D prev; // Punto anterior para verificar convergencia
            int iter = 0; // Contador de iteraciones

            // Bucle principal del algoritmo Mean Shift para un solo píxel
            do {
                prev = current; // Guarda el punto actual para comparar después
                Point5D sum(0, 0, 0, 0, 0); // Suma de vecinos válidos
                int count = 0; // Cantidad de vecinos válidos

                // Bucle para buscar vecinos en una ventana alrededor del píxel actual
                for (int j = -hs; j <= hs; ++j) {
                    for (int i = -hs; i <= hs; ++i) {
                        int nx = x + i; // Coordenada X del vecino
                        int ny = y + j; // Coordenada Y del vecino

                        // Si el vecino está dentro de los límites de la imagen
                        if (nx >= 0 && nx < cols && ny >= 0 && ny < rows) {
                            Point5D neighbor = getPoint5D(ny, nx, labImg); // Obtiene el vecino
                            // Si el vecino está lo suficientemente cerca (espacial y en color)
                            if (current.spatialDist(neighbor) <= hs &&
                                current.colorDist(neighbor) <= hr) {
                                sum = sum + neighbor; // Lo añade a la suma
                                count++; // Incrementa el contador
                            }
                        }
                    }
                }

                // Si se encontraron vecinos, calcula el nuevo centro de masa
                if (count > 0) {
                    current = sum / count;
                }

                iter++; // Siguiente iteración
            // Continúa mientras el píxel siga moviéndose significativamente o no haya llegado a maxIter
            } while (current.colorDist(prev) > tol_color &&
                     current.spatialDist(prev) > tol_spatial &&
                     iter < maxIter);

            // Convierte el píxel final de vuelta a formato Lab (0-255) y lo guarda en el resultado
            int l = static_cast<int>(current.l * 255.0f / 100.0f);
            int a = static_cast<int>(current.a + 128.0f);
            int b = static_cast<int>(current.b + 128.0f);
            resultChunk.at<Vec3b>(y, x) = Vec3b(saturate_cast<uchar>(l),
                                                saturate_cast<uchar>(a),
                                                saturate_cast<uchar>(b));
        }
    }

    // --- Comunicación MPI ---
    // Recopila los resultados de todos los procesos en la imagen 'labImg' del proceso raíz (rank 0)
    MPI_Gather(resultChunk.ptr(startRow), chunk * cols * 3, MPI_UNSIGNED_CHAR,
               labImg.ptr(0), chunk * cols * 3, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);
}

// --- Función principal del programa ---
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); // Inicia MPI

    int rank, size; // ID de este proceso y número total de procesos
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Obtiene el ID de este proceso
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Obtiene el número total de procesos

    Mat Img; // Objeto para la imagen

    if (rank == 0) { // --- Solo el proceso maestro (rank 0) ---
        Img = imread("C:/Users/LUIS FERNANDO/Pictures/arte/THL.jpg"); // Carga la imagen
        if (Img.empty()) { // Si la imagen no se carga, termina
            cerr << "No se pudo abrir la imagen." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        resize(Img, Img, Size(256, 256), 0, 0, INTER_LINEAR); // Redimensiona
        cvtColor(Img, Img, COLOR_BGR2Lab); // Convierte a Lab
    }

    MPI_Barrier(MPI_COMM_WORLD); // Todos esperan aquí hasta que el maestro cargue la imagen
    double start = MPI_Wtime(); // Inicia el temporizador

    int rows = 0, cols = 0; // Dimensiones de la imagen
    if (rank == 0) { // El maestro obtiene las dimensiones
        rows = Img.rows;
        cols = Img.cols;
    }
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD); // El maestro envía las filas
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD); // El maestro envía las columnas

    if (rank != 0) { // Si no es el maestro, crea un objeto Mat vacío del tamaño correcto
        Img = Mat(rows, cols, CV_8UC3);
    }

    MPI_Bcast(Img.data, rows * cols * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD); // El maestro envía la imagen a todos

    meanShiftMPI(Img, rank, size); // Ejecuta el algoritmo de segmentación

    double end = MPI_Wtime(); // Finaliza el temporizador

    if (rank == 0) { // --- Solo el proceso maestro (rank 0) ---
        cvtColor(Img, Img, COLOR_Lab2BGR); // Vuelve a convertir a BGR para mostrar
        cout << "Tiempo de ejecución (MPI): " << (end - start) * 1000 << " ms" << endl; // Muestra el tiempo
        namedWindow("Resultado MPI"); // Crea una ventana
        imshow("Resultado MPI", Img); // Muestra la imagen segmentada
        waitKey(0); // Espera a que el usuario presione una tecla
    }

    MPI_Finalize(); // Finaliza MPI
    return 0; // Termina el programa
}
