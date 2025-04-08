using Keras;
using Keras.Layers;
using Keras.Models;
using Numpy;
using Keras.Callbacks;
using System.Xml.Linq;

namespace GenreMusicNN
{
    internal class GenreClassifier
    {
        private int numMFCCs;
        private int numTimeSteps;
        private Sequential model;

        // Конструктор для создания модели CNN + LSTM
        public GenreClassifier(int numMFCCs = 240, int numTimeSteps = 128)
        {
            this.numMFCCs = numMFCCs;
            this.numTimeSteps = numTimeSteps;
            // Определяем архитектуру модели
            //BuildModel();
            BuildBiggerCNNModel();
        }

        // Метод построения архитектуры CNN + LSTM
        private void BuildModel()
        {
            var inputShape = new int[] { numTimeSteps, numMFCCs, 2 }; // Входной размер данных [время, коэффициенты, 2 (k, b)]
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
        }
        private void BuildCNNModel()
        {
            var inputShape = new int[] { numTimeSteps, numMFCCs, 2 };
            Shape shape = new Shape(inputShape);

            model = new Sequential();

            model.Add(new Conv2D(32, kernel_size: new Tuple<int, int>(3, 3), activation: "relu", padding: "same", input_shape: shape));
            model.Add(new MaxPooling2D(pool_size: new Tuple<int, int>(2, 2)));

            model.Add(new Conv2D(64, kernel_size: new Tuple<int, int>(3, 3), activation: "relu", padding: "same"));
            model.Add(new MaxPooling2D(pool_size: new Tuple<int, int>(2, 2)));

            model.Add(new Conv2D(128, kernel_size: new Tuple<int, int>(3, 3), activation: "relu", padding: "same"));
            model.Add(new MaxPooling2D(pool_size: new Tuple<int, int>(2, 2)));

            model.Add(new Flatten());
            model.Add(new Dense(128, activation: "relu"));

            model.Add(new Dense(TrainingData.GenreMapping.Count, activation: "softmax"));

            model.Compile(optimizer: "adam", loss: "categorical_crossentropy", metrics: new string[] { "accuracy" });
        }
        //Из статьи Traffic Density Classification Using Sound Datasets
        private void BuildBiggerCNNModel()
        {
            var inputShape = new int[] { numTimeSteps, numMFCCs, 2 };
            Shape shape = new Shape(inputShape);

            model = new Sequential();

            model.Add(new Conv2D(32, kernel_size: new Tuple<int, int>(3, 3), activation: "relu", padding: "same", input_shape: shape));
            model.Add(new MaxPooling2D(pool_size: new Tuple<int, int>(2, 2)));

            model.Add(new Conv2D(64, kernel_size: new Tuple<int, int>(3, 3), activation: "relu", padding: "same"));
            model.Add(new MaxPooling2D(pool_size: new Tuple<int, int>(2, 2)));

            model.Add(new Conv2D(128, kernel_size: new Tuple<int, int>(3, 3), activation: "relu", padding: "same"));
            model.Add(new MaxPooling2D(pool_size: new Tuple<int, int>(2, 2)));

            model.Add(new Conv2D(256, kernel_size: new Tuple<int, int>(3, 3), activation: "relu", padding: "same"));
            model.Add(new MaxPooling2D(pool_size: new Tuple<int, int>(2, 2)));

            model.Add(new Conv2D(512, kernel_size: new Tuple<int, int>(3, 3), activation: "relu", padding: "same"));
            model.Add(new MaxPooling2D(pool_size: new Tuple<int, int>(2, 2)));

            model.Add(new Flatten());
            model.Add(new Dense(256, activation: "relu"));

            model.Add(new Dense(TrainingData.GenreMapping.Count, activation: "softmax"));

            model.Compile(optimizer: "adam", loss: "categorical_crossentropy", metrics: new string[] { "accuracy" });
        }

        // Метод для обучения модели
        public void Train(float[][][][] X_train, float[][] Y_train, float[][][][] X_test = null, float[][] Y_test = null, int batch_size = 32, int epochs = 500)
        {
            // Преобразование 4D массива в одномерный и передача его в NDarray с правильной формой
            int dim1 = X_train.Length;
            int dim2 = X_train[0].Length;
            int dim3 = X_train[0][0].Length;
            int dim4 = X_train[0][0][0].Length;
            float[] flattenedArray = new float[dim1 * dim2 * dim3 * dim4];
            int index = 0;
            for (int i = 0; i < dim1; i++)
            {
                for (int j = 0; j < dim2; j++)
                {
                    for (int k = 0; k < dim3; k++)
                    {
                        for (int m = 0; m < dim4; m++)
                        {
                            flattenedArray[index++] = X_train[i][j][k][m];
                        }
                    }
                }
            }

            // Преобразуем одномерный массив в NDarray с указанием формы
            NDarray X_train_nd = np.array(flattenedArray).reshape(dim1, dim2, dim3, dim4);
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
            // Callback для ранней остановки
            var earlyStopping = new EarlyStopping(monitor: "loss", patience: 30, verbose: 1, restore_best_weights: true);
            // Начало обучения
            Console.WriteLine("Starting training...");
            model.Fit(X_train_nd, Y_train_nd, batch_size: batch_size, epochs: epochs, callbacks: new Callback[] { earlyStopping });
            Console.WriteLine("Ready!");

            // Если переданы тестовые данные – выполняем оценку
            if (X_test != null && Y_test != null)
            {
                var metrics = Evaluate(X_test, Y_test);
                Console.WriteLine($"[Post-Train Evaluation] Loss = {metrics[0]}, Accuracy = {metrics[1]}");
            }
        }

        // Метод для предсказания жанра
        public float[] Predict(float[][][][] mfccData)
        {
            // Преобразование 4D массива в одномерный и передача его в NDarray с правильной формой
            int dim1 = mfccData.Length;
            int dim2 = mfccData[0].Length;
            int dim3 = mfccData[0][0].Length;
            int dim4 = mfccData[0][0][0].Length;
            float[] flattenedArray = new float[dim1 * dim2 * dim3 * dim4];
            int index = 0;
            for (int i = 0; i < dim1; i++)
            {
                for (int j = 0; j < dim2; j++)
                {
                    for (int k = 0; k < dim3; k++)
                    {
                        for (int m = 0; m < dim4; m++)
                        {
                            flattenedArray[index++] = mfccData[i][j][k][m];
                        }
                    }
                }
            }
            // Преобразуем одномерный массив в NDarray с указанием формы
            NDarray mfccData_nd = np.array(flattenedArray).reshape(1, dim1, dim2, dim3, dim4); // Учёт нового измерения
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

        public void CompareModels(float[][][][] X_train, float[][] Y_train, float[][][][] X_test, float[][] Y_test, int epoch)
        {
            Console.WriteLine($"Training CNN + LSTM model");
            BuildModel(); // Строим CNN + LSTM
            Train(X_train, Y_train, epochs: epoch);
            var lstmAccuracy = Evaluate(X_test, Y_test);
            //Console.WriteLine($"CNN + LSTM Accuracy: {lstmAccuracy[1]}");

            Console.WriteLine($"Training CNN-only model");
            BuildCNNModel(); // Строим только CNN
            Train(X_train, Y_train, epochs: epoch);
            var cnnAccuracy = Evaluate(X_test, Y_test);
            Console.WriteLine($"CNN + LSTM Accuracy: {lstmAccuracy[1]}, loss: {lstmAccuracy[0]} \n " +
                $"CNN Accuracy: {cnnAccuracy[1]}, loss: {cnnAccuracy[0]}");
        }

        public void TestWindowSizes(float[][][][] X_train, float[][] Y_train, float[][][][] X_test, float[][] Y_test, int timeSteps, int epochs, int mfccCount, string outputPath = "test_results.txt")
        {
            Console.WriteLine($"Testing timeSteps = {timeSteps}");

            // Открываем файл для дозаписи
            using (StreamWriter writer = new StreamWriter(outputPath, append: true))
            {
                // Заголовок таблицы
                writer.WriteLine($"TimeSteps = {timeSteps}");
                writer.WriteLine("Model\tAccuracy\tLoss");
                Console.WriteLine($"TimeSteps = {timeSteps}");
                Console.WriteLine("Model\tAccuracy\tLoss");

                // --- CNN-only ---
                Console.WriteLine("Testing CNN-only model...");
                BuildBiggerCNNModel();
                this.Train(X_train, Y_train, batch_size: timeSteps, epochs: epochs);
                var cnnResults = this.Evaluate(X_test, Y_test);

                // Вывод и в консоль, и в файл
                string cnnLine = $"CNN\t{cnnResults[1]:F4}\t{cnnResults[0]:F4}";
                Console.WriteLine(cnnLine);
                writer.WriteLine(cnnLine);

                // --- Если добавишь CNN+LSTM позже ---
                /*
                BuildModel();
                Train(X_train, Y_train, batch_size: timeSteps, epochs: epochs);
                var lstmResults = Evaluate(X_test, Y_test);
                string lstmLine = $"CNN+LSTM\t{lstmResults[1]:F4}\t{lstmResults[0]:F4}";
                Console.WriteLine(lstmLine);
                writer.WriteLine(lstmLine);
                */

                writer.WriteLine(); // Пустая строка в конце блока
                Console.WriteLine(); // Для читаемости
            }
        }


        public double[] Evaluate(float[][][][] X_test, float[][] Y_test)
        {
            int dim1 = X_test.Length;
            int dim2 = X_test[0].Length;
            int dim3 = X_test[0][0].Length;
            int dim4 = X_test[0][0][0].Length;
            float[] flattenedArray = new float[dim1 * dim2 * dim3 * dim4];
            int index = 0;
            for (int i = 0; i < dim1; i++)
            {
                for (int j = 0; j < dim2; j++)
                {
                    for (int k = 0; k < dim3; k++)
                    {
                        for (int m = 0; m < dim4; m++)
                        {
                            if (X_test[i][j][k][m] == null)
                            {
                                Console.WriteLine(i + " " +  j + " " + k + " " + m);
                                break;
                            }
                            flattenedArray[index++] = X_test[i][j][k][m];
                        }
                    }
                }
            }
            NDarray X_test_nd = np.array(flattenedArray).reshape(dim1, dim2, dim3, dim4);

            int y_dim1 = Y_test.Length;
            int y_dim2 = Y_test[0].Length;

            float[] flattenedY = new float[y_dim1 * y_dim2];
            index = 0;
            for (int i = 0; i < y_dim1; i++)
            {
                for (int j = 0; j < y_dim2; j++)
                {
                    flattenedY[index++] = Y_test[i][j];
                }
            }
            NDarray Y_test_nd = np.array(flattenedY).reshape(y_dim1, y_dim2);

            var result = model.Evaluate(X_test_nd, Y_test_nd);
            return result; // Точность и потери
        }
    }
}
