# Tarea-ML-Clase-25
Tarea final del curso de Fundamentos de Aprendizaje Automático 2025

## Descripción de la tarea
La tarea final de aprendizaje automático de este semestre incluye cuatro temas: reconocimiento de dígitos escritos a mano, detección de imágenes médicas, detección de saliencia de imágenes y generación de imágenes en color. Cada tema se completará en una clase de laboratorio y el informe se presentará dentro de una semana después de la clase. A continuación se detallan las descripciones específicas de cada tema:

1. Reconocimiento de dígitos escritos a mano: problema clásico de aprendizaje automático, consiste en reconocer 10 clases de dígitos escritos a mano en imágenes en escala de grises, es un problema de clasificación.

2. Detección de imágenes médicas: utilizar imágenes de fondo de ojo en color para determinar si hay enfermedad, es un problema de clasificación.

3. Predicción de saliencia de imágenes: en imágenes en color, predecir las áreas que captan fácilmente la atención del ojo humano (generar mapa de saliencia), es un problema de regresión.

4. Generación de imágenes en color: diseñar y entrenar una red generativa para producir imágenes en color. Se puede optar por usar GAN, VAE u otros modelos generativos, con el objetivo de generar imágenes en color similares a la distribución de datos de entrenamiento a partir de ruido aleatorio.

## Contenido de la tarea

1. Entrenar modelos de aprendizaje automático en el conjunto de entrenamiento.
2. Probar el rendimiento del modelo en el conjunto de prueba.
3. Presentar informe experimental y código del modelo.

## Tema 1: Reconocimiento de dígitos escritos a mano

### Archivos de datos
* Conjunto de datos de entrenamiento (10 clases, 60.000 dígitos en total): almacenados en formato bmp en **1-Digit-TrainSet.zip**.
* Conjunto de datos de prueba (10 clases, 10.000 dígitos en total): almacenados en formato bmp en **1-Digit-TestSet.zip**.

En cada conjunto de datos, el primer dígito del nombre de archivo representa su clasificación real (etiqueta), es decir, ground truth.

![](/1-Digit-Example.png)

### Métricas de rendimiento
Precisión de clasificación en el conjunto de prueba.

## Tema 2: Detección de imágenes médicas

### Archivos de datos
* Conjunto de datos de entrenamiento (2 clases, 1639 imágenes en total): almacenadas en formato jpg en **2-MedImage-TrainSet.zip**.
* Conjunto de datos de prueba (2 clases, 250 imágenes en total): almacenadas en formato jpg en **2-MedImage-TestSet.zip**.

![](/2-MedImage-Example.png)

En cada conjunto de datos, los archivos que comienzan con "disease" son imágenes con enfermedad, y los que comienzan con "normal" son imágenes sin enfermedad.

### Métricas de rendimiento
La métrica básica es la precisión de clasificación en el conjunto de prueba. Considerando que el número de muestras con y sin enfermedad no es equilibrado, y que los dos tipos de errores (clasificar sin enfermedad como enfermedad, y clasificar enfermedad como sin enfermedad) conllevan diferentes riesgos, para reflejar completamente el rendimiento del clasificador, también se pueden proporcionar precisión, AUC, curva ROC (las funciones de métricas están proporcionadas en la carpeta ROC, consulte instruction.txt para el uso detallado del código) u otras métricas.

## Tema 3: Predicción de saliencia de imágenes

### Archivos de datos
* Conjunto de datos de entrenamiento (1600 imágenes para detectar y 1600 mapas de saliencia correspondientes): almacenados en formato jpg en **3-Saliency-TrainSet.zip**.
* Conjunto de datos de prueba (400 imágenes para detectar y 400 mapas de saliencia correspondientes): almacenados en formato jpg en **3-Saliency-TestSet.zip**.

En cada conjunto de datos, las imágenes para detectar son imágenes en color observadas directamente por el ojo humano, guardadas en la carpeta **Stimuli**; los mapas de saliencia correspondientes (es decir, ground truth) son imágenes en escala de grises del mismo tamaño, donde las áreas más brillantes representan mayor saliencia, guardadas en la carpeta **FIXATIONMAPS**. Considerando que el contenido de las imágenes puede afectar los resultados, cada conjunto de datos incluye 20 tipos diferentes de imágenes, almacenadas en 20 carpetas (como **Action**, **Affective**, **Art**...), por lo tanto, al analizar los resultados, se puede proporcionar tanto el rendimiento general como el análisis por tipo.

![](/3-Saliency-Example.png)

### Métricas de rendimiento

Métricas subjetivas: comparación subjetiva entre el mapa de saliencia predicho y el mapa de saliencia ground truth.

Métricas objetivas: coeficiente de correlación (CC), divergencia KL (las funciones de métricas están en el archivo metric.py, se pueden llamar directamente, con instrucciones de uso incluidas), u otras métricas que midan la similitud de imágenes de saliencia.

## Tema 4: Generación de imágenes en color

### Archivos de datos
CIFAR-10 es un conjunto de datos ampliamente utilizado para tareas de clasificación de imágenes, que contiene 10 categorías diferentes de imágenes en color. Cada categoría contiene 6000 imágenes, para un total de 60,000 imágenes de 32x32 píxeles, divididas en 50,000 imágenes de entrenamiento y 10,000 imágenes de prueba. Este conjunto de datos es muy común en tareas de aprendizaje automático y visión por computadora.

Las categorías de CIFAR-10 incluyen: avión (airplane), automóvil (automobile), pájaro (bird), gato (cat), ciervo (deer), perro (dog), rana (frog), caballo (horse), barco (ship), camión (truck). Al analizar los resultados, se puede proporcionar tanto el rendimiento general como el análisis por tipo.

![](/4-CIFAR10-Example.png)

### Métricas de rendimiento
Métricas subjetivas: evaluación subjetiva de la calidad de las imágenes generadas, comparando con las imágenes reales del conjunto de datos. En el informe se puede mostrar la evolución de los resultados de las imágenes generadas a medida que aumentan las iteraciones de entrenamiento.

Métricas objetivas: usar métricas de evaluación como Inception Score (IS) y Frechet Inception Distance (FID) para analizar la calidad de las imágenes generadas.

```python
# Primero instalar la biblioteca torch-fidelity
pip install torch-fidelity

import torch_fidelity
def fidelity_metric(genereated_images_path, real_images_path):
"""
Usar el paquete fidelity para calcular todas las métricas relacionadas con la generación, introduciendo las rutas de las imágenes generadas y las imágenes reales
isc: inception score
kid: kernel inception distance
fid: frechet inception distance
"""
  metrics_dict = torch_fidelity.calculate_metrics(
    input1=genereated_images_path,
    input2=real_images_path,
    cuda=True,
    isc=True,
    fid=True,
    kid=True,
    verbose=False
  )
  return metrics_dict
```

## Obtención de datos
Los conjuntos de datos de los primeros tres temas se pueden obtener a través de los siguientes enlaces:

Descarga de Baidu Cloud: https://pan.baidu.com/s/1mOCFxATcCkHGbK8Vdtv5yQ

Descarga de DropBox: https://www.dropbox.com/sh/i79cbllw6763zxg/AAA3-jPaRlYHMvsMyRbtRRmaa?dl=0

Los archivos son iguales después de descargar por cualquiera de los dos métodos, puede elegir cualquiera de ellos.

**Dado que los métodos de descarga anteriores pueden ser inconvenientes, aquí se proporciona un método de descarga a través del disco en la nube de Beihang, la dirección es:**[https://bhpan.buaa.edu.cn/link/AA84F755C78F1F4062BB81EBD5B41D5F7A](https://bhpan.buaa.edu.cn/link/AA84F755C78F1F4062BB81EBD5B41D5F7A)

El conjunto de datos del cuarto tema se puede obtener a través del siguiente código:

```python
from torchvision.datasets import CIFAR10
dataset = CIFAR10(root='./CIFARdata', download=True, transform=transforms.ToTensor())
```

## Presentación del informe
El informe experimental debe nombrarse con **número de estudiante + nombre**, y se debe presentar un **archivo PDF** en la carpeta correspondiente del disco en la nube. El informe experimental debe incluir:
1. Descripción del problema
2. Principios y descripción general del modelo experimental
3. Estructura y parámetros del modelo experimental
4. Análisis de resultados experimentales (incluyendo resultados de pruebas en conjuntos de entrenamiento y prueba), se requiere enumerar algunos **casos fallidos** y analizarlos. Cuantas más métricas de análisis se proporcionen y más detallado sea el análisis con gráficos, mayor será la puntuación.
5. Conclusión