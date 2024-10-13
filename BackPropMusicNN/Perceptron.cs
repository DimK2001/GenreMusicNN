using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GenreMusicNN
{
    internal class Perceptron
    {
        // Количество нейронов в каждом слое
        public int[] Layers { get; private set; }
        // Значения нейронов
        public double[][] Neurons { get; private set; }
        // Веса между слоями
        public List<double[][]> Weights { get; private set; } = new List<double[][]>();
        // Смещения (bias)
        public double[][] Biases { get; private set; }
        // Скорость обучения
        private double learningRate = 0.01;
        // Генератор случайных чисел
        private static Random random = new Random(Guid.NewGuid().GetHashCode());

        // Конструктор перцептрона
        public Perceptron(int[] layers, double learningRate = 0.01)
        {
            Layers = layers;
            this.learningRate = learningRate;
            InitializeNeurons();
            InitializeWeights();
            InitializeBiases();
        }

        // Инициализация нейронов для каждого слоя
        private void InitializeNeurons()
        {
            Neurons = new double[Layers.Length][];
            for (int i = 0; i < Layers.Length; i++)
            {
                Neurons[i] = new double[Layers[i]];
            }
        }

        // Инициализация весов
        private void InitializeWeights()
        {
            for (int i = 0; i < Layers.Length - 1; i++)
            {
                double[][] layerWeights = new double[Layers[i]][];
                for (int j = 0; j < Layers[i]; j++)
                {
                    layerWeights[j] = new double[Layers[i + 1]];
                    for (int k = 0; k < Layers[i + 1]; k++)
                    {
                        layerWeights[j][k] = random.NextDouble() * 2 - 1; // Веса в диапазоне [-1, 1]
                    }
                }
                Weights.Add(layerWeights);
            }
        }

        // Инициализация смещений для каждого слоя (кроме входного)
        private void InitializeBiases()
        {
            Biases = new double[Layers.Length - 1][];
            for (int i = 1; i < Layers.Length; i++)
            {
                Biases[i - 1] = new double[Layers[i]];
                for (int j = 0; j < Layers[i]; j++)
                {
                    Biases[i - 1][j] = random.NextDouble() * 2 - 1; // Смещения в диапазоне [-1, 1]
                }
            }
        }

        // Прямое распространение (FeedForward)
        public double[] FeedForward(double[][] inputs)
        {
            // Преобразуем двумерный входной массив в одномерный вектор
            var flattenedInput = FlattenInput(inputs);
            // Устанавливаем значения для входного слоя
            for (int i = 0; i < flattenedInput.Length; i++)
            {
                Neurons[0][i] = flattenedInput[i];
            }
            // Прямое распространение для каждого слоя
            for (int i = 0; i < Layers.Length - 1; i++)
            {
                for (int j = 0; j < Layers[i + 1]; j++)
                {
                    double value = 0;
                    for (int k = 0; k < Layers[i]; k++)
                    {
                        value += Neurons[i][k] * Weights[i][k][j];
                    }
                    Neurons[i + 1][j] = ReLU(value + Biases[i][j]); // Применение функции активации
                }
            }

            return Neurons[Layers.Length - 1]; // Возвращаем выходной слой
        }

        // Преобразование двумерного массива MFCC в одномерный вектор
        private double[] FlattenInput(double[][] inputs)
        {
            int rows = inputs.Length;
            int cols = inputs[0].Length;
            double[] flattened = new double[rows * cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    flattened[i * cols + j] = inputs[i][j];
                }
            }
            return flattened;
        }

        // Реализация функции активации ReLU
        private double ReLU(double x)
        {
            return Math.Max(0, x);
        }

        // Производная ReLU для обратного распространения
        private double ReLUDerivative(double x)
        {
            return x > 0 ? 1 : 0;
        }

        // Обратное распространение ошибки (Backpropagation)
        public void Backpropagation(double[] expected)
        {
            double[][] errors = new double[Layers.Length][];

            // Инициализация ошибок на каждом слое
            for (int i = 0; i < Layers.Length; i++)
            {
                errors[i] = new double[Layers[i]];
            }
            // Ошибки на выходном слое
            for (int i = 0; i < Layers[Layers.Length - 1]; i++)
            {
                errors[Layers.Length - 1][i] = Neurons[Layers.Length - 1][i] - expected[i];
            }
            // Обратное распространение ошибки
            for (int i = Layers.Length - 2; i >= 0; i--)
            {
                for (int j = 0; j < Layers[i]; j++)
                {
                    double error = 0;
                    for (int k = 0; k < Layers[i + 1]; k++)
                    {
                        error += errors[i + 1][k] * Weights[i][j][k];
                        Weights[i][j][k] -= learningRate * errors[i + 1][k] * Neurons[i][j]; // Обновление весов
                    }
                    errors[i][j] = error * ReLUDerivative(Neurons[i][j]);
                }
                // Обновление смещений
                for (int j = 0; j < Layers[i + 1]; j++)
                {
                    Biases[i][j] -= learningRate * errors[i + 1][j];
                }
            }
        }
        // Softmax для выходного слоя
        private double[] Softmax(double[] x)
        {
            // Находим максимальное значение для числовой стабильности
            double max = x.Max();

            // Вычисляем сумму экспонент с вычитанием максимума
            double sum = 0.0;
            for (int i = 0; i < x.Length; i++)
            {
                sum += Math.Exp(x[i] - max);
            }
            // Применяем Softmax ко всем элементам
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = Math.Exp(x[i] - max) / sum;
            }
            return x;
        }
    }
}