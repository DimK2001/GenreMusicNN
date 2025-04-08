using System.Text.Json;

namespace GenreMusicNN
{
    internal static class TrainingData
    {
        public static List<string> AudioFiles { get; private set; } = new List<string>();
        public static List<string> TestFiles { get; private set; } = new List<string>();
        // Список векторов меток для мультижанровых песен
        public static List<float[]> Labels { get; private set; } = new List<float[]>();
        public static List<float[]> TestLabels { get; private set; } = new List<float[]>();
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
        public static void AddTestFile(string filePath, float[] labelVector)
        {
            TestFiles.Add(filePath);
            TestLabels.Add(labelVector);
        }

        public static async Task LoadSongBase()
        {
            var songs = Directory.GetFiles("SongBase");
            foreach (var songPath in songs)
            {
                using (FileStream fs = new FileStream(songPath, FileMode.Open))
                {
                    Song? song = await JsonSerializer.DeserializeAsync<Song>(fs);
                    if (song != null) AddAudioFile(song.filePath, song.labelVector);
                }
            }
            return;
        }
        public static async Task LoadTestBase()
        {
            var songs = Directory.GetFiles("TestBase");
            foreach (var songPath in songs)
            {
                using (FileStream fs = new FileStream(songPath, FileMode.Open))
                {
                    Song? song = await JsonSerializer.DeserializeAsync<Song>(fs);
                    if (song != null) AddTestFile(song.filePath, song.labelVector);
                }
            }
            return;
        }

        public static void AddToSongBase(string filePath, float[] labelVector)
        {
            var song = new Song
            {
                filePath = filePath,
                labelVector = labelVector
            };
            string fileName = "SongBase\\" + filePath.Split("\\")[filePath.Split("\\").Length - 1] + ".json";
            string jsonString = JsonSerializer.Serialize(song);
            File.WriteAllText(fileName, jsonString);
        }

        public static void CountSongsByGenre()
        {
            // Создаем словарь для подсчета количества песен по жанрам
            var genreCount = new Dictionary<int, int>();

            // Проходим по меткам жанров для всех песен
            foreach (var labelVector in Labels)
            {
                // Для каждой песни проверяем, какой жанр активен (где значение больше 0)
                for (int i = 0; i < labelVector.Length; i++)
                {
                    if (labelVector[i] >= 0.5f)
                    {
                        if (genreCount.ContainsKey(i))
                        {
                            genreCount[i]++;
                        }
                        else
                        {
                            genreCount[i] = 1;
                        }
                    }
                }
            }

            // Выводим количество песен для каждого жанра
            foreach (var genre in genreCount)
            {
                string genreName = GenreMapping.ContainsKey(genre.Key)
                    ? GenreMapping[genre.Key]
                    : "Unknown Genre";

                Console.WriteLine($"{genreName}: {genre.Value} songs");
            }
        }

    public class Song
        {
            public string filePath { get; set; }
            public float[] labelVector { get; set; }
        }
    }
}