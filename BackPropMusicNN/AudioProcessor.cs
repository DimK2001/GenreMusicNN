using Accord.Audio;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using NAudio.Dsp;
using Tensorflow.Keras.Utils;

namespace GenreMusicNN
{
    internal class AudioProcessor
    {
        private int sampleRate;
        private int mfccCount; // Количество MFCC коэффициентов
        private int timeSteps = 256; // Количество временных окон
        private const int stepMultyplier = 2; 
        private SpectrogramGenerator generator = new SpectrogramGenerator();
        int[] testSteps = new int[] { 64, 88, 144, 256, 320, 400 };

        // Конструктор класса
        public AudioProcessor(int sampleRate = 40000, int mfccCount = 140, int timeSteps = 256)
        {
            this.sampleRate = sampleRate;
            this.mfccCount = mfccCount;
            this.timeSteps = timeSteps;
        }
        // Метод чтения аудиофайла и подготовки данных для нейросети
        public float[][][][] ProcessAudioFileTest(string filePath)
        {
            float[][][][] preparedData = new float[3][][][];
            float[] audioData = new float[1];
            try
            {
                // Считываем аудиофайл и нормализуем его
                audioData = LoadAudioFile(filePath, out int originalSampleRate);
                // Если частота дискретизации файла отличается от целевой, пересэмплируем
                if (originalSampleRate != sampleRate)
                {
                    audioData = Resample(audioData, originalSampleRate, sampleRate);
                }
            }
            catch
            {
                throw new Exception("Can not read choosen audiofile.");
            }
            finally
            {
                // Извлекаем MFCC из аудиоданных
                var mfcc = ExtractMFCC(audioData);
                // Приводим к требуемому размеру (дополнение или обрезка)
                preparedData = PrepareDataForNeuralNetworkTest(mfcc);
            }
            return preparedData;
        }
        // Метод чтения аудиофайла и подготовки данных для нейросети
        public float[][][][] ProcessAudioFile(string filePath)
        {
            float[][][][] preparedData = new float[3][][][];
            float[] audioData = new float[1];
            try
            {
                // Считываем аудиофайл и нормализуем его
                audioData = LoadAudioFile(filePath, out int originalSampleRate);
                // Если частота дискретизации файла отличается от целевой, пересэмплируем
                if (originalSampleRate != sampleRate)
                {
                    audioData = Resample(audioData, originalSampleRate, sampleRate);
                }
            }
            catch
            {
                throw new Exception("Can not read choosen audiofile.");
            }
            finally
            {
                // Извлекаем MFCC из аудиоданных
                var mfcc = ExtractMFCC(audioData);
                // Приводим к требуемому размеру (дополнение или обрезка)
                preparedData = PrepareDataForNeuralNetwork(mfcc);
            }
            return preparedData;
        }

        // Метод загрузки аудиофайла
        private float[] LoadAudioFile(string filePath, out int actualSampleRate)
        {
            float[] audioData;
            using (var audioFileReader = new AudioFileReader(filePath))
            {
                // Получаем фактическую частоту дискретизации из аудиофайла
                actualSampleRate = audioFileReader.WaveFormat.SampleRate;
                // Вычисляем параметры аудио
                int bytesPerSample = audioFileReader.WaveFormat.BitsPerSample / 8; // Количество байт на сэмпл
                int blockAlign = audioFileReader.WaveFormat.BlockAlign;            // Размер блока (байт на все каналы)
                int totalSamples = (int)(audioFileReader.Length / bytesPerSample); // Общее количество сэмплов
                                                                                   // Создаем буфер для данных
                audioData = new float[totalSamples];
                // Читаем данные из файла блоками
                int samplesRead = 0;
                int offset = 0;
                byte[] buffer = new byte[blockAlign]; // Буфер для чтения данных блоками
                float sampleValue = 0f;
                while (samplesRead < totalSamples)
                {
                    // Чтение одного блока данных
                    int bytesRead = audioFileReader.Read(buffer, 0, blockAlign);
                    if (bytesRead == 0)
                        break; // Если больше нечего читать, выходим из цикла
                               // Преобразование байтов в сэмплы (нормализация до float от -1.0f до 1.0f)
                    for (int i = 0; i < bytesRead / bytesPerSample; i++)
                    {
                        // Преобразование сэмплов в float
                        if (audioFileReader.WaveFormat.BitsPerSample == 16)
                        {
                            // Если 16 бит, конвертируем из Int16 в float
                            sampleValue = BitConverter.ToInt16(buffer, i * bytesPerSample) / 32768f;
                        }
                        else if (audioFileReader.WaveFormat.BitsPerSample == 32)
                        {
                            // Если 32 бит, конвертируем из Int32 в float
                            sampleValue = BitConverter.ToInt32(buffer, i * bytesPerSample) / (float)Int32.MaxValue;
                        }
                        audioData[offset + i] = sampleValue; // Записываем в аудиомассив
                    }
                    offset += bytesRead / bytesPerSample; // Смещаемся по массиву данных
                    samplesRead += bytesRead / bytesPerSample; // Обновляем количество прочитанных сэмплов
                }
                // Нормализация данных
                float maxAmplitude = audioData.Max(Math.Abs);
                if (maxAmplitude > 0)
                {
                    for (int i = 0; i < audioData.Length; i++)
                    {
                        audioData[i] /= maxAmplitude;  // Нормализация амплитуды в диапазон [-1, 1]
                    }
                }
                return audioData;
            }
        }

        // Метод извлечения MFCC
        private float[][] ExtractMFCC(float[] audioData)
        {
            // Преобразуем аудиоданные в сигнал
            Signal signal = Signal.FromArray(audioData, sampleRate);
            // Создаем объект для извлечения MFCC коэффициентов
            var mfccExtractor = new MelFrequencyCepstrumCoefficient(cepstrumCount: mfccCount, samplingRate: sampleRate);
            // Извлекаем MFCC коэффициенты
            IEnumerable<MelFrequencyCepstrumCoefficientDescriptor> mfccDescriptors = mfccExtractor.Transform(signal);
            var mfccList = mfccDescriptors.ToList();
            int numFrames = mfccList.Count;
            float[][] mfcc = new float[numFrames][];

            // Заполняем массив MFCC коэффициентов
            Parallel.For(0, numFrames, i =>
            {
                mfcc[i] = new float[mfccCount];
                double[] coefficients = mfccList[i].Descriptor;
                for (int j = 0; j < mfccCount; j++)
                {
                    mfcc[i][j] = Convert.ToSingle(coefficients[j]);
                }
            });
            return mfcc;
        }

        // Метод для пересэмплирования
        private float[] Resample(float[] audioData, int originalSampleRate, int targetSampleRate)
        {
            var waveFormat = WaveFormat.CreateIeeeFloatWaveFormat(originalSampleRate, 1); // Моно формат
            var byteArray = new byte[audioData.Length * sizeof(float)];
            Buffer.BlockCopy(audioData, 0, byteArray, 0, byteArray.Length);

            using (var stream = new RawSourceWaveStream(new MemoryStream(byteArray), waveFormat))
            {
                var resampler = new WdlResamplingSampleProvider(new WaveToSampleProvider(stream), targetSampleRate);
                var resampledData = new List<float>();
                var buffer = new float[1024];
                int samplesRead;

                while ((samplesRead = resampler.Read(buffer, 0, buffer.Length)) > 0)
                {
                    resampledData.AddRange(buffer.Take(samplesRead));
                }
                return resampledData.ToArray();
            }
        }

        // Приведение данных к нужному размеру (дополнение или обрезка)
        private float[][][][] PrepareDataForNeuralNetwork(float[][] mfcc)
        {
            return CompressMFCC(mfcc, timeSteps);
        }
        // Приведение данных к нужному размеру (дополнение или обрезка)
        private float[][][][] PrepareDataForNeuralNetworkTest(float[][] mfcc)
        {
            List<float[][][]> compressed = new List<float[][][]>();
            for (int i = 0; i < testSteps.Length; i++)
            {
                int currentLength = mfcc.Length;
                if (currentLength < testSteps[i] * stepMultyplier)
                {
                    // Дополнение
                    mfcc = PadMFCC(mfcc);
                }
                else if (currentLength > testSteps[i] * stepMultyplier)
                {
                    // Обрезка
                    mfcc = TruncateMFCC(mfcc);
                }
                var tmp = CompressMFCC(mfcc, testSteps[i]);
                for (int j = 0; j < tmp.Length; j++)
                {
                    compressed.Add(tmp[j]);
                }
            }
            return compressed.ToArray();
        }
        float[][][][] CompressMFCC(float[][] mfcc, int _timeStep)
        {
            /*var spectrum = new float[512][];
            for (int i = 0; i < spectrum.Length; i++)
            {
                spectrum[i] = mfcc[i*45];
            }
            //Сохранение изначальной спектрограммы
            generator.SaveSpectrogram(spectrum, "Mel_Spectrum.png");*/

            float[][][][] compressedFactoredData = new float[3][][][];
            int m = 0;
            for (int stepFactor = 4; stepFactor <= 104; stepFactor += 50)// 4, 54, 104
            {
                int _stepMultyplier = stepMultyplier * stepFactor;
                int currentLength = mfcc.Length;
                int numCoefficients = mfcc[0].Length;

                float[][][] compressedMFCC = new float[_timeStep][][];
                for (int i = 0; i < _timeStep; i++)
                {
                    compressedMFCC[i] = new float[numCoefficients][];
                    float[] data = new float[_stepMultyplier];
                    for (int k = 0; k < numCoefficients; ++k)
                    {
                        for (int j = 0; j < _stepMultyplier; ++j)
                        {
                            int index = i * _stepMultyplier + j;
                            if (index < currentLength) // Проверка на выход за границы
                            {
                                data[j] = mfcc[index][k];
                            }
                            else
                            {
                                data[j] = 0; // Если данных меньше, чем ожидается, заполняем нулями
                            }
                        }
                        (float K, float B) = LeastSquaresMethod(data);
                        compressedMFCC[i][k] = new float[] { K, B };
                    }
                }
                compressedFactoredData[m] = compressedMFCC;
                //m++;

                /*
                if (mfcc.Length > _timeStep * _stepMultyplier)
                {
                    for (int i = 0; i < _timeStep; ++i)
                    {
                        int param = mfcc.Length - _timeStep * _stepMultyplier + i;
                        compressedMFCC[i] = new float[numCoefficients][];
                        float[] data = new float[_stepMultyplier];
                        for (int k = 0; k < numCoefficients; ++k)
                        {
                            for (int j = 0; j < _stepMultyplier; ++j)
                            {
                                int index = param + j * _stepMultyplier;
                                data[j] = mfcc[index][k];
                            }
                            (float K, float B) = LeastSquaresMethod(data);
                            compressedMFCC[i][k] = new float[] { K, B };
                        }
                    }
                    compressedFactoredData[m] = compressedMFCC;
                }
                else
                {
                    for (int i = 0; i < 3; ++i)
                    {
                        //compressedFactoredData[i + 3] = compressedFactoredData[i];
                    }
                }*/
                m++;
            }
            return compressedFactoredData;
        }
        // Метод для нахождения уравнения прямой методом наименьших квадратов
        (float k, float b) LeastSquaresMethod(float[] y)
        {
            int n = y.Length;
            float sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;

            // Значения x — это индексы элементов массива y
            for (int i = 0; i < n; i++)
            {
                float x = i;
                sumX += x;
                sumY += y[i];
                sumXY += x * y[i];
                sumX2 += x * x;
            }

            // Вычисляем коэффициенты k и b
            float k = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
            float b = (sumY - k * sumX) / n;

            return (k, b);
        }

        // Метод дополнения (Padding) MFCC до требуемой длины
        private float[][] PadMFCC(float[][] mfcc)
        {
            int currentLength = mfcc.Length;
            int numCoefficients = mfcc[0].Length;

            float[][] paddedMFCC = new float[timeSteps * stepMultyplier][];
            // Копируем исходные данные
            for (int i = 0; i < currentLength; i++)
            {
                paddedMFCC[i] = new float[numCoefficients];
                for (int j = 0; j < numCoefficients; j++)
                {
                    paddedMFCC[i][j] = mfcc[i][j];
                }
            }
            // Оставшиеся элементы заполняем нулями
            for (int i = currentLength; i < timeSteps * stepMultyplier; i++)
            {
                paddedMFCC[i] = new float[numCoefficients];
                for (int j = 0; j < numCoefficients; j++)
                {
                    paddedMFCC[i][j] = 0.0f;
                }
            }
            return paddedMFCC;
        }

        // Метод обрезки (Truncation) MFCC до требуемой длины
        private float[][] TruncateMFCC(float[][] mfcc)
        {
            int numCoefficients = mfcc[0].Length;
            float[][] truncatedMFCC = new float[timeSteps * stepMultyplier][];
            // Копируем только нужное количество временных окон
            for (int i = 0; i < timeSteps * stepMultyplier; i++)
            {
                truncatedMFCC[i] = new float[numCoefficients];
                for (int j = 0; j < numCoefficients; j++)
                {
                    truncatedMFCC[i][j] = mfcc[i][j];
                }
            }
            return truncatedMFCC;
        }

        // Метод для обработки аудио и предсказания жанра
        public float[] ProcessAndClassifyTest(string filePath)
        {
            float[] predictedGenre = new float[TrainingData.GenreMapping.Count];
            float[][][][] inputDatas = ProcessAudioFileTest(filePath);
            foreach (var inputData in inputDatas)
            {
                // Преобразуем массив для работы с нейросетью (добавляем третье измерение для каналов)
                float[][][][] inputForNetwork = inputData.Select(frame => frame.Select(coeff => coeff.Select(
                    c => new float[] { c }).ToArray()).ToArray()).ToArray();
                // Создаем экземпляр модели классификации
                var classifier = new GenreClassifier(numMFCCs: mfccCount,
                    numTimeSteps: timeSteps);
                classifier.LoadModel("ReadyModel.h5");
                // Предсказываем жанр на основе MFCC
                for (int i = 0; i < predictedGenre.Length; i++)
                {
                    predictedGenre[i] += classifier.Predict(inputForNetwork)[i] / 3f;
                }
            }
            // Возвращаем индекс жанра с наибольшей вероятностью
            return predictedGenre;
        }
        // Метод для обработки аудио и предсказания жанра
        public float[] ProcessAndClassify(string filePath)
        {
            float[] predictedGenre = new float[TrainingData.GenreMapping.Count];
            float[][][][] inputDatas = ProcessAudioFile(filePath);
            foreach (var inputData in inputDatas)
            {
                // Преобразуем массив для работы с нейросетью (добавляем третье измерение для каналов)
                float[][][][] inputForNetwork = inputData.Select(frame => frame.Select(coeff => coeff.Select(
                    c => new float[] { c }).ToArray()).ToArray()).ToArray();
                // Создаем экземпляр модели классификации
                var classifier = new GenreClassifier(numMFCCs: mfccCount,
                    numTimeSteps: timeSteps);
                classifier.LoadModel("ReadyModel.h5");
                // Предсказываем жанр на основе MFCC
                for (int i = 0; i < predictedGenre.Length; i++)
                {
                    predictedGenre[i] += classifier.Predict(inputForNetwork)[i] / 3f;
                }
            }
            // Возвращаем индекс жанра с наибольшей вероятностью
            return predictedGenre;
        }

        public void TrainModel(string[] trainingFiles, float[][] labels, string[] testFiles, float[][] testLabels)
        {
            // Инициализация массивов для входных данных и меток
            int fileAmount = trainingFiles.Length * 3;            // Учитывая изменение pitch для каждого файла (0.5, 1, 2)
            float[][][][] X_train = new float[fileAmount][][][];  // Данные в виде четырехмерного массива: {files, timeSteps, mfccCount, 2}
            float[][] Y_train = new float[fileAmount][];          // Векторы меток
            Parallel.For(0, trainingFiles.Length, i =>
            {
                float[][][][] inputDatas = ProcessAudioFile(trainingFiles[i]);
                for (int j = 0; j < 3; ++j)
                {
                    // Преобразование в нужный формат для нейросети
                    float[][][] inputDataReshaped = new float[timeSteps][][];
                    for (int t = 0; t < timeSteps; t++)
                    {
                        inputDataReshaped[t] = new float[mfccCount][];
                        for (int k = 0; k < mfccCount; k++)
                        {
                            inputDataReshaped[t][k] = new float[2];  // {k, b}
                            inputDataReshaped[t][k][0] = inputDatas[j][t][k][0]; // k
                            inputDataReshaped[t][k][1] = inputDatas[j][t][k][1]; // b
                        }
                    }
                    // Сохраняем данные в массив
                    X_train[i * 3 + j] = inputDataReshaped;
                    Y_train[i * 3 + j] = labels[i];
                }
                ClearMemory(); // Чистим память после обработки каждого файла
                Console.WriteLine(i + " Ready file " + trainingFiles[i]);
            });

            // Загрузка тестовой базы
            float[][][][] X_test = new float[testFiles.Length * 3][][][];
            float[][] Y_test = new float[testFiles.Length * 3][];

            Parallel.For(0, testFiles.Length, i =>
            {
                float[][][][] inputDatas = ProcessAudioFile(testFiles[i]);
                for (int j = 0; j < 3; ++j)
                {
                    // Используем только оригинальный pitch (inputDatas[1]) для теста
                    float[][][] inputDataReshaped = new float[timeSteps][][];

                    for (int t = 0; t < timeSteps; t++)
                    {
                        inputDataReshaped[t] = new float[mfccCount][];
                        for (int k = 0; k < mfccCount; k++)
                        {
                            inputDataReshaped[t][k] = new float[2];
                            inputDataReshaped[t][k][0] = inputDatas[j][t][k][0];
                            inputDataReshaped[t][k][1] = inputDatas[j][t][k][1];
                        }
                    }
                    X_test[i * 3 + j] = inputDataReshaped;
                    Y_test[i * 3 + j] = labels[i];
                }
                Console.WriteLine(i + " Ready test file " + testFiles[i]);
                ClearMemory(); // Чистим память после обработки каждого файла
            });

            // Обучение модели с тестовыми данными
            var classifier = new GenreClassifier(numMFCCs: mfccCount, numTimeSteps: timeSteps);
            classifier.Train(X_train, Y_train, X_test, Y_test, batch_size: 32, epochs: 500);
            classifier.SaveModel("ReadyModel.h5");
            ClearMemory(); // Чистим память после обучения
        }

        public void TestModel(string[] trainingFiles, string[] testFiles, float[][] labels, float[][] labelsTest)
        {
            // Инициализация массивов для входных данных и меток
            int fileAmount = trainingFiles.Length;
            int steps = testSteps.Length * 3;
            float[][][][][] X_train = new float[steps][][][][];  // Данные в виде четырехмерного массива: {files, timeSteps, mfccCount, 2}
            float[][][] Y_train = new float[steps][][];          // Векторы меток

            // Инициализация X_train и Y_train
            for (int stepFactor = 0; stepFactor < steps; ++stepFactor)
            {
                X_train[stepFactor] = new float[fileAmount][][][]; // Данные для каждого шага
                Y_train[stepFactor] = new float[fileAmount][];    // Метки для каждого шага
            }

            Parallel.For(0, trainingFiles.Length, i =>
            {
                float[][][][] inputDatas = ProcessAudioFileTest(trainingFiles[i]);
                for (int stepFactor = 0; stepFactor < steps; ++stepFactor) // 12 шагов (3 фактора * 4 testSteps)
                {
                    int testStepIndex = stepFactor / 3;

                    for (int j = 0; j < inputDatas.Length / testSteps.Length; ++j) // 3 варианта сжатия
                    {
                        // Преобразование в нужный формат для нейросети
                        float[][][] inputDataReshaped = new float[testSteps[testStepIndex]][][];

                        for (int t = 0; t < testSteps[testStepIndex]; t++) // 32... временных шага
                        {
                            inputDataReshaped[t] = new float[mfccCount][]; // mfccCount коэффициентов

                            for (int k = 0; k < mfccCount; k++) // 240 коэффициентов MFCC
                            {
                                inputDataReshaped[t][k] = new float[2];  // {K, B}
                                inputDataReshaped[t][k][0] = inputDatas[stepFactor][t][k][0]; // K
                                inputDataReshaped[t][k][1] = inputDatas[stepFactor][t][k][1]; // B
                            }
                        }
                        X_train[stepFactor][i] = inputDataReshaped; // Записываем в итоговый массив
                        Y_train[stepFactor][i] = labels[i]; // Заполняем метки
                    }
                }
                Console.WriteLine(i + " Ready file " + trainingFiles[i]);
            });
            // Подготовка тестовых данных
            fileAmount = testFiles.Length;
            float[][][][][] X_test = new float[steps][][][][];  // Данные для теста
            float[][][] Y_test = new float[steps][][];          // Метки для теста

            // Инициализация X_test и Y_test
            for (int stepFactor = 0; stepFactor < steps; ++stepFactor)
            {
                X_test[stepFactor] = new float[fileAmount][][][]; // Для каждого шага
                Y_test[stepFactor] = new float[fileAmount][];    // Метки для каждого шага
            }

            Parallel.For(0, testFiles.Length, i =>
            {
                float[][][][] inputDatas = ProcessAudioFileTest(testFiles[i]);
                for (int stepFactor = 0; stepFactor < steps; ++stepFactor) // 12 шагов (3 фактора * 4 testSteps)
                {
                    int testStepIndex = stepFactor / 3;

                    for (int j = 0; j < inputDatas.Length / testSteps.Length; ++j) // 3 варианта сжатия
                    {
                        // Преобразование в нужный формат для нейросети
                        float[][][] inputDataReshaped = new float[testSteps[testStepIndex]][][];

                        for (int t = 0; t < testSteps[testStepIndex]; t++) // 32 временных шага
                        {
                            inputDataReshaped[t] = new float[mfccCount][]; // mfccCount коэффициентов

                            for (int k = 0; k < mfccCount; k++) // 240 коэффициентов MFCC
                            {
                                inputDataReshaped[t][k] = new float[2];  // {K, B}
                                inputDataReshaped[t][k][0] = inputDatas[stepFactor][t][k][0]; // K
                                inputDataReshaped[t][k][1] = inputDatas[stepFactor][t][k][1]; // B
                            }
                        }
                        X_test[stepFactor][i] = inputDataReshaped; // Записываем в итоговый массив
                        Y_test[stepFactor][i] = labelsTest[i]; // Заполняем метки
                    }
                }
                Console.WriteLine(i + " Ready TEST file " + testFiles[i]);
            });

            // Тестирование моделей
            for (int i = 0; i < testSteps.Length; ++i)
            {
                // Создаем экземпляр модели классификации
                var classifier = new GenreClassifier(numMFCCs: mfccCount, numTimeSteps: testSteps[i]);
                // Применяем метод тестирования с окнами
                Console.WriteLine("!!!!!!!!!!Test of 4 factor:");
                classifier.TestWindowSizes(X_train[i * 3], Y_train[i * 3], X_test[i * 3], Y_test[i * 3], testSteps[i], 512, mfccCount);
                Console.WriteLine("!!!!!!!!!!Test of 54 factor:");
                classifier.TestWindowSizes(X_train[i * 3 + 1], Y_train[i * 3 + 1], X_test[i * 3 + 1], Y_test[i * 3 + 1], testSteps[i], 512, mfccCount);
                Console.WriteLine("!!!!!!!!!!Test of 104 factor:");
                classifier.TestWindowSizes(X_train[i * 3 + 2], Y_train[i * 3 + 2], X_test[i * 3 + 2], Y_test[i * 3 + 2], testSteps[i], 512, mfccCount);
            }
            // Четвёртый шаг — усреднение предсказаний по всем 3 масштабам
            for (int i = 0; i < testSteps.Length; ++i)
            {
                var classifier = new GenreClassifier(numMFCCs: mfccCount, numTimeSteps: testSteps[i]);
                Console.WriteLine("!!!!!!!!!!Test of ALL factors (avg):");

                // Объединение 3 версий для каждого примера (обучающего и тестового)
                int trainLen = X_train[i * 3].Length;
                float[][][][] X_train_merged = new float[trainLen * 3][][][];
                float[][] Y_train_merged = new float[trainLen * 3][];
                for (int j = 0; j < trainLen; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                        X_train_merged[j * 3 + k] = X_train[i * 3 + k][j];
                        Y_train_merged[j * 3 + k] = Y_train[i * 3 + k][j];
                    }
                }

                int testLen = X_test[i * 3].Length;
                float[][][][] X_test_merged = new float[testLen][][][];
                float[][] Y_test_merged = new float[testLen][];
                for (int j = 0; j < testLen; ++j)
                {
                    float[] predictionSum = new float[TrainingData.GenreMapping.Count];
                    float[] label = Y_test[i * 3][j]; // одинаковые метки на всех 3
                    for (int k = 0; k < 3; ++k)
                    {
                        var prediction = classifier.Predict(new float[][][][] { X_test[i * 3 + k][j] });
                        for (int g = 0; g < prediction.Length; g++)
                            predictionSum[g] += prediction[g] / 3f;
                    }
                    // Можно сохранить predictionSum или вывести accuracy на основе всех
                    X_test_merged[j] = X_test[i * 3][j]; // не важно, что тут — нужно только Y для сравнения
                    Y_test_merged[j] = label; // одна метка
                }

                // Финальный вызов тестирования
                classifier.TestWindowSizes(X_train_merged, Y_train_merged, X_test_merged, Y_test_merged, testSteps[i], 512, mfccCount);
            }
            Console.WriteLine("Test finished...............");
            // Очистка памяти после завершения
            ClearMemory();
        }


        private void ClearMemory()
        {
            GC.Collect();  // Принудительный вызов сборщика мусора
            GC.WaitForPendingFinalizers();
        }
    }
}
