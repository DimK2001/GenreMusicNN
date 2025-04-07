using System;
using System.Runtime.InteropServices;
using System.Windows.Forms;

namespace GenreMusicNN
{
    internal static class Program
    {
        [DllImport("kernel32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        static extern bool AllocConsole();
        [DllImport("kernel32.dll", SetLastError = true, ExactSpelling = true)]
        static extern bool FreeConsole();
        /// <summary>
        ///  The main entry point for the application.
        /// </summary>
        [STAThread]
        static void Main()
        {
            Keras.Keras.DisablePySysConsoleLog = true;
            var audioProcessor = new AudioProcessor();
            // Добавляем жанры
            TrainingData.AddGenre(0, "Rock/Metal");
            TrainingData.AddGenre(1, "Pop");
            TrainingData.AddGenre(2, "Jazz/Blues");
            TrainingData.AddGenre(3, "Classical");
            TrainingData.AddGenre(4, "Chanson");
            TrainingData.AddGenre(5, "Rap/Hip-Hop");
            TrainingData.AddGenre(6, "Electronic");
            TrainingData.AddGenre(7, "Country");

            // To customize application configuration such as set high DPI settings or default font,
            // see https://aka.ms/applicationconfiguration.
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
//#if DEBUG
            AllocConsole();
            Console.WriteLine("Debug Console");
//#endif
            Application.Run(new MainWindow(audioProcessor));
//#if DEBUG
            FreeConsole();
//#endif
        }
    }
}