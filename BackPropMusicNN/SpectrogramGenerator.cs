using System;
using System.Drawing;
using System.Drawing.Imaging;

namespace GenreMusicNN
{
    internal class SpectrogramGenerator
    {
        public void SaveSpectrogram(float[][] mfccData, string filePath)
        {
            int width = mfccData.Length;       // Время (количество окон)
            int height = mfccData[0].Length;   // Частоты (количество коэффициентов)

            using (Bitmap bmp = new Bitmap(width, height))
            {
                for (int x = 0; x < width; x++)
                {
                    for (int y = 0; y < height; y++)
                    {
                        // Нормализация значений в диапазон 0-255
                        float value = mfccData[x][y];
                        int intensity = (int)(255 * (value - MinValue(mfccData)) / (MaxValue(mfccData) - MinValue(mfccData)));
                        intensity = Math.Max(0, Math.Min(255, intensity));

                        // Генерация оттенка серого
                        Color color = Color.FromArgb(intensity, intensity, intensity);
                        bmp.SetPixel(x, height - y - 1, color); // Инверсия Y для корректного отображения
                    }
                }
                bmp.Save(filePath, ImageFormat.Png);
            }
        }

        private float MinValue(float[][] data)
        {
            float min = float.MaxValue;
            foreach (var row in data)
                foreach (var val in row)
                    if (val < min) min = val;
            return min;
        }

        private float MaxValue(float[][] data)
        {
            float max = float.MinValue;
            foreach (var row in data)
                foreach (var val in row)
                    if (val > max) max = val;
            return max;
        }
    }
}
