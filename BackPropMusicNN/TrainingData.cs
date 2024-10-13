using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GenreMusicNN
{
    internal static class TrainingData
    {
        public static List<string> AudioFiles { get; private set; } = new List<string>();
        // Список векторов меток для мультижанровых песен
        public static List<float[]> Labels { get; private set; } = new List<float[]>();
        public static Dictionary<int, string> GenreMapping { get; private set; } = new Dictionary<int, string>();

        // Метод для добавления жанра
        public static void AddGenre(int label, string genreName)
        {
            if (!GenreMapping.ContainsKey(label))
            {
                GenreMapping[label] = genreName;
            }
        }

        // Метод для добавления аудиофайла с вектором жанров
        public static void AddAudioFile(string filePath, float[] labelVector)
        {
            AudioFiles.Add(filePath);
            Labels.Add(labelVector);
        }
    }
}