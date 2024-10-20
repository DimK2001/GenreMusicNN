using Keras;
using Keras.Layers;
using Keras.Models;
using Numpy;
using Keras.Callbacks;

namespace GenreMusicNN
{
    internal class GenreClassifier
    {
        private int numMFCCs;
        private int numTimeSteps;
        private Sequential model;

        // Конструктор для создания модели CNN + LSTM
        public GenreClassifier(int numMFCCs = 13, int numTimeSteps = 128)
        {
            this.numMFCCs = numMFCCs;
            this.numTimeSteps = numTimeSteps;

            // Определяем архитектуру модели
            BuildModel();
        }

        // Метод построения архитектуры CNN + LSTM
        private void BuildModel()
        {
            var inputShape = new int[] { numTimeSteps, numMFCCs, 1 }; // 2D вход
            Shape shape = new Shape(inputShape);

            model = new Sequential();

            // Первый сверточный слой с 'same' padding
            model.Add(new Conv2D(32, kernel_size: new Tuple<int, int>(3, 3), activation: "relu", padding: "same", input_shape: shape));
            model.Add(new MaxPooling2D(pool_size: new Tuple<int, int>(2, 2)));

            // Второй сверточный слой с 'same' padding
            model.Add(new Conv2D(64, kernel_size: new Tuple<int, int>(3, 3), activation: "relu", padding: "same"));
            model.Add(new MaxPooling2D(pool_size: new Tuple<int, int>(2, 2)));

            // Третий сверточный слой с 'same' padding
            model.Add(new Conv2D(128, kernel_size: new Tuple<int, int>(3, 3), activation: "relu", padding: "same"));
            model.Add(new MaxPooling2D(pool_size: new Tuple<int, int>(2, 2)));

            // Преобразуем данные в последовательность для LSTM
            model.Add(new Reshape(new Shape(new int[] { numTimeSteps / 4, -1 })));

            // LSTM слой
            model.Add(new LSTM(128, return_sequences: false));

            // Полносвязный слой
            model.Add(new Dense(128, activation: "relu"));

            // Выходной слой с Softmax для классификации
            model.Add(new Dense(TrainingData.GenreMapping.Count, activation: "softmax"));

            model.Compile(optimizer: "adam", loss: "categorical_crossentropy", metrics: new string[] { "accuracy" });
            // Печать структуры модели
            //model.Summary();
        }

        // Метод для обучения модели
        public void Train(float[][][] X_train, float[][] Y_train, int batch_size = 32, int epochs = 1000)
        {
            // Преобразование 3D массива в одномерный и передача его в NDarray с правильной формой
            int dim1 = X_train.Length;
            int dim2 = X_train[0].Length;
            int dim3 = X_train[0][0].Length;

            // Преобразование трехмерного массива в одномерный
            float[] flattenedArray = new float[dim1 * dim2 * dim3];
            int index = 0;
            for (int i = 0; i < dim1; i++)
            {
                for (int j = 0; j < dim2; j++)
                {
                    for (int k = 0; k < dim3; k++)
                    {
                        flattenedArray[index++] = X_train[i][j][k];
                    }
                }
            }

            // Преобразуем одномерный массив в NDarray с указанием формы
            NDarray X_train_nd = np.array(flattenedArray).reshape(dim1, dim2, dim3);

            // Аналогично для меток (Y_train)
            int y_dim1 = Y_train.Length;
            int y_dim2 = Y_train[0].Length;

            float[] flattenedY = new float[y_dim1 * y_dim2];
            index = 0;
            for (int i = 0; i < y_dim1; i++)
            {
                for (int j = 0; j < y_dim2; j++)
                {
                    flattenedY[index++] = Y_train[i][j];
                }
            }

            NDarray Y_train_nd = np.array(flattenedY).reshape(y_dim1, y_dim2);
            // Создание callback для ранней остановки
            var earlyStopping = new EarlyStopping(monitor: "loss", patience: 30, verbose: 1, restore_best_weights: true);
            // Теперь передаем NDarray в Keras модель
            Console.WriteLine("Starting training...");
            model.Fit(X_train_nd, Y_train_nd, batch_size: batch_size, epochs: epochs, callbacks: new Callback[] { earlyStopping });
            Console.WriteLine("Ready!");
        }

        // Метод для предсказания жанра
        public float[] Predict(float[][][] mfccData)
        {
            // Преобразование 3D массива в одномерный и передача его в NDarray с правильной формой
            int dim1 = mfccData.Length;
            int dim2 = mfccData[0].Length;
            int dim3 = mfccData[0][0].Length;

            // Преобразование трехмерного массива в одномерный
            float[] flattenedArray = new float[dim1 * dim2 * dim3];
            int index = 0;
            for (int i = 0; i < dim1; i++)
            {
                for (int j = 0; j < dim2; j++)
                {
                    for (int k = 0; k < dim3; k++)
                    {
                        flattenedArray[index++] = mfccData[i][j][k];
                    }
                }
            }

            // Преобразуем одномерный массив в NDarray с указанием формы
            NDarray mfccData_nd = np.array(flattenedArray).reshape(1, dim1, dim2);
            var predictions = model.Predict(mfccData_nd);
            return predictions.GetData<float>();
        }
        // Метод для сохранения модели
        public void SaveModel(string filePath)
        {
            model.Save(filePath);
            Console.WriteLine("Model saved");
        }

        // Метод для загрузки модели
        public void LoadModel(string filePath)
        {
            model.LoadWeight(filePath);
        }
    }
}
