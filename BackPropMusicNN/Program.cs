namespace GenreMusicNN
{
    internal static class Program
    {
        /// <summary>
        ///  The main entry point for the application.
        /// </summary>
        [STAThread]
        static void Main()
        {
            Keras.Keras.DisablePySysConsoleLog = true;
            var audioProcessor = new AudioProcessor(44100);
            // Добавляем жанры
            TrainingData.AddGenre(0, "Rock/Metal");
            TrainingData.AddGenre(1, "Pop");
            TrainingData.AddGenre(2, "Jazz/Blues");
            TrainingData.AddGenre(3, "Classical");
            TrainingData.AddGenre(4, "Chanson");
            TrainingData.AddGenre(5, "Rap/Hip-Hop");
            TrainingData.AddGenre(6, "Electronic");
            TrainingData.AddGenre(7, "Country");

            #region Songs
            // Добавляем аудиофайлы с соответствующими жанрами
            TrainingData.AddAudioFile("Songs\\The Rolling Stones - Paint It, Black.wav", new float[] {
                0.90f, //Rock/Metal
                0.05f, //Pop
                0.05f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Led Zeppelin - Stairway To Heaven.wav", new float[] {
                0.83f, //Rock/Metal
                0.10f, //Pop
                0.02f, //Jazz/Blues
                0.05f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Guns N' Roses - Sweet Child O' Mine.wav", new float[] {
                1.00f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Chuck Berry - Johnny B. Goode.mp3", new float[] {
                0.70f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.05f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.25f }); //Country
            TrainingData.AddAudioFile("Songs\\Creedence Clearwater Revival - Fortunate Son.mp3", new float[] {
                0.95f, //Rock/Metal
                0.00f, //Pop
                0.05f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Creedence Clearwater Revival - Suzie Q.mp3", new float[] {
                0.90f, //Rock/Metal
                0.00f, //Pop
                0.10f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Ozzy Osbourne - Crazy Train.wav", new float[] {
                1.00f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Madonna - Like A Prayer.wav", new float[] {
                0.00f, //Rock/Metal
                1.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Gorillaz - Feel Good Inc..wav", new float[] {
                0.00f, //Rock/Metal
                0.40f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.40f, //Rap/Hip-Hop
                0.20f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Miles Davis - So What.wav", new float[] {
                0.00f, //Rock/Metal
                0.00f, //Pop
                1.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Beethoven - Symphony No. 5 (Proms 2012).wav", new float[] {
                0.00f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                1.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Вивальди - Времена года Весна allegro.mp3", new float[] {
                0.00f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                1.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Вивальди - Времена года Лето.mp3", new float[] {
                0.00f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                1.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Прокофьев - Танец Рыцарей.mp3", new float[] {
                0.00f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                1.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Robert Johnson - Cross Road Blues.mp3", new float[] {
                0.00f, //Rock/Metal
                0.00f, //Pop
                0.80f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.20f }); //Country
            TrainingData.AddAudioFile("Songs\\Robert Johnson - Me and the Devil Blues (Take 2).mp3", new float[] {
                0.00f, //Rock/Metal
                0.00f, //Pop
                0.90f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.10f }); //Country
            TrainingData.AddAudioFile("Songs\\b.b.-king-blues-boys-tune.mp3", new float[] {
                0.03f, //Rock/Metal
                0.02f, //Pop
                0.95f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\b.b.-king-the-thrill-is-gone.mp3", new float[] {
                0.00f, //Rock/Metal
                0.00f, //Pop
                1.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Maximum the Hormone - Zetsubou Billy.mp3", new float[] {
                0.90f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.10f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\nobody.one - Heroin.mp3", new float[] {
                0.85f, //Rock/Metal
                0.05f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.10f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\AC-DC-Highway to Hell.mp3", new float[] {
                0.98f, //Rock/Metal
                0.02f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\ac-dc-thunderstruck.mp3", new float[] {
                0.98f, //Rock/Metal
                0.02f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Metallica-All Nightmare Long.mp3", new float[] {
                1.00f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Metallica-Enter Sandman.mp3", new float[] {
                0.90f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.10f }); //Country
            TrainingData.AddAudioFile("Songs\\Metallica-Muster Of Puppets.mp3", new float[] {
                1.00f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Metallica-One.mp3", new float[] {
                0.80f, //Rock/Metal
                0.10f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.10f }); //Country
            TrainingData.AddAudioFile("Songs\\Metallica-Seek&Destroy.mp3", new float[] {
                1.00f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Pantera - Walk.mp3", new float[] {
                1.00f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Любэ-Давай за.mp3", new float[] {
                0.20f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.80f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Любэ-Комбат.mp3", new float[] {
                0.20f, //Rock/Metal
                0.10f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.70f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Любэ-Конь.mp3", new float[] {
                0.20f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.60f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.20f }); //Country
            TrainingData.AddAudioFile("Songs\\Михаил Круг-Владимерский централ.mp3", new float[] {
                0.00f, //Rock/Metal
                0.05f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.95f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Михаил Шафутинский-Третье сентября.mp3", new float[] {
                0.00f, //Rock/Metal
                0.15f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.85f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Михаил Шафутинский-Третье сентября.mp3", new float[] {
                0.00f, //Rock/Metal
                0.40f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.60f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Михаил Елизаров-Одноместное сердце.mp3", new float[] {
                0.00f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.95f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.05f }); //Country
            TrainingData.AddAudioFile("Songs\\Михаил Елизаров-Рагнарёк.mp3", new float[] {
                0.00f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.95f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.05f }); //Country
            TrainingData.AddAudioFile("Songs\\Михаил Елизаров-Советская.mp3", new float[] {
                0.05f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.90f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.05f }); //Country
            TrainingData.AddAudioFile("Songs\\Воровайки - Хоп мусорок.mp3", new float[] {
                0.00f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                1.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Михаил Круг - Магадан.mp3", new float[] {
                0.00f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                1.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Михаил Круг - Доброго пути.mp3", new float[] {
                0.00f, //Rock/Metal
                0.00f, //Pop
                0.05f, //Jazz/Blues
                0.00f, //Classic
                0.95f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Михаил Круг - На юга.mp3", new float[] {
                0.00f, //Rock/Metal
                0.00f, //Pop
                0.10f, //Jazz/Blues
                0.00f, //Classic
                0.90f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Григорий Лепс и Михаил Круг - Кольщик.mp3", new float[] {
                0.10f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.90f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Группа Сентябрь - Лето возвратится.mp3", new float[] {
                0.00f, //Rock/Metal
                0.30f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.70f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Nirvana-Heart Shaped Box.mp3", new float[] {
                0.90f, //Rock/Metal
                0.00f, //Pop
                0.10f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Nirvana-Smells like teen spirit.mp3", new float[] {
                1.00f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Пионер лагерь пыльная радуга-Гранж.mp3", new float[] {
                0.90f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.10f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Пионер лагерь пыльная радуга-Дорога добра.mp3", new float[] {
                0.90f, //Rock/Metal
                0.10f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Пионер лагерь пыльная радуга-Мало.mp3", new float[] {
                0.70f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.30f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Пионер лагерь пыльная радуга-Панк и Рок-н-Ролл.mp3", new float[] {
                0.90f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.10f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\ЖЩ - 7000000000.mp3", new float[] {
                0.85f, //Rock/Metal
                0.15f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\ЖЩ - Боль всего мира.mp3", new float[] {
                0.95f, //Rock/Metal
                0.05f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\ЖЩ - Гранж.mp3", new float[] {
                1.00f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\ЖЩ - У меня есть мечта.mp3", new float[] {
                1.00f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\ЖЩ - Запал.mp3", new float[] {
                0.90f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.10f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Многосольный - Пост Instrumental.mp3", new float[] {
                0.80f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.10f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.10f }); //Country
            TrainingData.AddAudioFile("Songs\\Eminem – Lose Yourself.mp3", new float[] {
                0.00f, //Rock/Metal
                0.10f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.90f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Coolio - Gangsta's Paradise.mp3", new float[] {
                0.00f, //Rock/Metal
                0.40f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.60f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\50 Cent – In da Club.mp3", new float[] {
                0.00f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.90f, //Rap/Hip-Hop
                0.10f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Lil Jon & The East Side Boyz - Get Low.mp3", new float[] {
                0.00f, //Rock/Metal
                0.10f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.90f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Eminem – Without Me.mp3", new float[] {
                0.00f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.95f, //Rap/Hip-Hop
                0.05f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Eminem - Not afraid.mp3", new float[] {
                0.00f, //Rock/Metal
                0.30f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.65f, //Rap/Hip-Hop
                0.05f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\noize-mc-Выдыхай.mp3", new float[] {
                0.20f, //Rock/Metal
                0.20f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.60f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\НТР - HelloWorld.mp3", new float[] {
                0.00f, //Rock/Metal
                0.10f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.70f, //Rap/Hip-Hop
                0.20f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\НТР - Костыль и велосипед.mp3", new float[] {
                0.00f, //Rock/Metal
                0.10f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.80f, //Rap/Hip-Hop
                0.10f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\НТР - Х.х и в продакшн.mp3", new float[] {
                0.00f, //Rock/Metal
                0.10f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.80f, //Rap/Hip-Hop
                0.10f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Adana Twins feat. Upercent - Darkness.mp3", new float[] {
                0.00f, //Rock/Metal
                0.10f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.90f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\borne-control.mp3", new float[] {
                0.00f, //Rock/Metal
                0.10f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.10f, //Rap/Hip-Hop
                0.80f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\M.I.A. - Paper Planes.mp3", new float[] {
                0.00f, //Rock/Metal
                0.30f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.20f, //Rap/Hip-Hop
                0.50f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\imanbek-feat.-taichu-ella-quiere-techno.mp3", new float[] {
                0.00f, //Rock/Metal
                0.10f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.20f, //Rap/Hip-Hop
                0.70f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Klur - Waterfall.mp3", new float[] {
                0.00f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                1.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\PLVTO - Maybe.mp3", new float[] {
                0.00f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                1.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Rednex - Cotton Eye Joe.mp3", new float[] {
                0.00f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                1.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Mitchell Tenpenny - I Won't.mp3", new float[] {
                0.00f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                1.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Johnny Cash - a Boy Named Sue.mp3", new float[] {
                0.00f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                1.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Johnny Cash - Hurt.mp3", new float[] {
                0.48f, //Rock/Metal
                0.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.52f }); //Country
            TrainingData.AddAudioFile("Songs\\Celine Dion - My Heart Will Go On - Love Theme from Titanic.mp3", new float[] {
                0.00f, //Rock/Metal
                1.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Алла погачева - Айсберг.mp3", new float[] {
                0.00f, //Rock/Metal
                1.00f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Глюкоза - Невеста.mp3", new float[] {
                0.20f, //Rock/Metal
                0.60f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.20f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Глюкоза - Швайне.mp3", new float[] {
                0.20f, //Rock/Metal
                0.60f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.20f, //Electronic
                0.00f }); //Country
            TrainingData.AddAudioFile("Songs\\Bonnie Tyler - I Need A Hero.mp3", new float[] {
                0.10f, //Rock/Metal
                0.88f, //Pop
                0.00f, //Jazz/Blues
                0.00f, //Classic
                0.00f, //Chanson
                0.00f, //Rap/Hip-Hop
                0.00f, //Electronic
                0.02f }); //Country
            #endregion

            // To customize application configuration such as set high DPI settings or default font,
            // see https://aka.ms/applicationconfiguration.
            ApplicationConfiguration.Initialize();
            Application.Run(new MainWindow(audioProcessor));
            // Создаем экземпляр AudioProcessor  
        }
    }
}