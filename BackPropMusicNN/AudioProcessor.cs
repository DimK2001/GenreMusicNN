using Accord.Audio;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;

namespace GenreMusicNN
{
    internal class AudioProcessor
    {
        private int sampleRate;
        private const int mfccCount = 60; // Количество MFCC коэффициентов
        private const int timeSteps = 4096; // Количество временных окон
        private const int stepMultyplier = 6;

        // Конструктор класса
        public AudioProcessor(int sampleRate = 44100)
        {
            this.sampleRate = sampleRate;
        }

        // Метод чтения аудиофайла и подготовки данных для нейросети
        public float[][][] ProcessAudioFile(string filePath)
        {
            float[][][] preparedData = new float[1][][];
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
            using (var audioFileReader = new AudioFileReader(filePath))
            {
                // Получаем фактическую частоту дискретизации из аудиофайла
                actualSampleRate = audioFileReader.WaveFormat.SampleRate;

                // Вычисляем параметры аудио
                int bytesPerSample = audioFileReader.WaveFormat.BitsPerSample / 8; // Количество байт на сэмпл
                int blockAlign = audioFileReader.WaveFormat.BlockAlign;            // Размер блока (байт на все каналы)
                int totalSamples = (int)(audioFileReader.Length / bytesPerSample); // Общее количество сэмплов

                // Создаем буфер для данных
                float[] audioData = new float[totalSamples];

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
            for (int i = 0; i < numFrames; ++i)
            {
                mfcc[i] = new float[mfccCount];
                double[] coefficients = mfccList[i].Descriptor;
                for (int j = 0; j < mfccCount; j++)
                {
                    mfcc[i][j] = Convert.ToSingle(coefficients[j]);
                }
            }
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
        private float[][][] PrepareDataForNeuralNetwork(float[][] mfcc)
        {
            int currentLength = mfcc.Length;
            int numCoefficients = mfcc[0].Length;
            
            if (currentLength < timeSteps * stepMultyplier)
            {
                // Дополнение (Padding)
                mfcc = PadMFCC(mfcc);
            }
            else if (currentLength > timeSteps * stepMultyplier)
            {
                // Обрезка (Truncation)
                mfcc = TruncateMFCC(mfcc);
            }

            return CompressMFCC(mfcc, timeSteps);
        }
        float[][][] CompressMFCC(float[][] mfcc, int targetLength)
        {
            int currentLength = mfcc.Length;
            int numCoefficients = mfcc[0].Length;

            float[][][] compressedMFCC = new float[targetLength][][];
            for (int i = 0; i < targetLength; i++)
            {
                compressedMFCC[i] = new float[numCoefficients][];
                float[] data = new float[stepMultyplier];
                for (int k = 0; k < numCoefficients; ++k)
                {
                    for (int j = 0; j < stepMultyplier; ++j)
                    {
                        int index = i * stepMultyplier + j;
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
                    compressedMFCC[i][k] = new float[]{K, B};
                }
            }
            return compressedMFCC;
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
        public float[] ProcessAndClassify(string filePath)
        {
            // Считываем аудиофайл и нормализуем его
             float[][][] inputData = ProcessAudioFile(filePath);

            // Преобразуем массив для работы с нейросетью (добавляем третье измерение для каналов)
            float[][][][] inputForNetwork = inputData.Select(frame => frame.Select(coeff => coeff.Select(c => new float[] { c }).ToArray()).ToArray()).ToArray();

            // Создаем экземпляр модели классификации
            var classifier = new GenreClassifier(numMFCCs: mfccCount, 
                numTimeSteps: timeSteps);
            classifier.LoadModel("ReadyModel.h5");
            // Предсказываем жанр на основе MFCC
            float[] predictedGenre = classifier.Predict(inputForNetwork);

            // Возвращаем индекс жанра с наибольшей вероятностью
            return predictedGenre;
        }

        public void TrainModel(List<string> trainingFiles, List<float[]> labels)
        {
            // Инициализация списков для входных данных и меток
            List<float[][][]> X_train = new List<float[][][]>(); // Данные в виде трехмерного массива: {timeSteps, numCoefficients, 2}
            List<float[]> Y_train = new List<float[]>();         // Векторы меток

            for (int i = 0; i < trainingFiles.Count; i++)
            {
                // Обработка каждого аудиофайла
                float[][][] inputData = ProcessAudioFile(trainingFiles[i]);
                // Преобразование в нужный формат для нейросети
                float[][][] inputDataReshaped = new float[timeSteps][][];
                for (int t = 0; t < timeSteps; t++)
                {
                    inputDataReshaped[t] = new float[mfccCount][];
                    // Копируем данные для каждого коэффициента (k и b)
                    for (int k = 0; k < mfccCount; k++)
                    {
                        inputDataReshaped[t][k] = new float[2];  // {k, b}
                        // Копируем коэффициенты k и b
                        inputDataReshaped[t][k][0] = inputData[t][k][0]; // k
                        inputDataReshaped[t][k][1] = inputData[t][k][1]; // b
                    }
                }
                // Добавляем подготовленные данные в список
                X_train.Add(inputDataReshaped);
                // Добавляем вектор меток для соответствующего аудиофайла
                Y_train.Add(labels[i]);

                ClearMemory(); // Чистим память после обработки каждого файла
            }

            // Конвертируем списки в массивы
            var X_trainArray = X_train.ToArray();
            var Y_trainArray = Y_train.ToArray();

            // Создаем экземпляр модели классификации
            var classifier = new GenreClassifier(numMFCCs: mfccCount, numTimeSteps: timeSteps);

            // Обучаем модель с новыми данными
            classifier.Train(X_trainArray, Y_trainArray, batch_size: 32, epochs: 500);

            // Сохраняем обученную модель
            classifier.SaveModel("ReadyModel.h5");

            ClearMemory(); // Чистим память после обучения
        }

        private void ClearMemory()
        {
            GC.Collect();  // Принудительный вызов сборщика мусора
            GC.WaitForPendingFinalizers();
        }
    }
}
