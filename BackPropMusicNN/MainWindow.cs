namespace GenreMusicNN
{
    public partial class MainWindow : Form
    {
        AudioProcessor audioProcessor;
        internal MainWindow(AudioProcessor processor)
        {
            audioProcessor = processor;
            InitializeComponent();
            string strExeFilePath = System.Reflection.Assembly.GetExecutingAssembly().Location;
            var sub = strExeFilePath.Split("\\");
            FileName.Text = strExeFilePath.Substring(0, strExeFilePath.Length - sub[sub.Length - 1].Length) + "Search";
        }

        protected void RetrainNetwork()
        {
            // trainingData.AudioFiles и TrainingData.Labels для обучения модели
            audioProcessor.TrainModel(TrainingData.AudioFiles, TrainingData.Labels);
        }
        protected string GenreClassification(string filePath)
        {
            string output = "";
            // Классификация музыкального жанра
            var vector = audioProcessor.ProcessAndClassify(filePath);
            for (int i = 0; i < vector.Length; ++i)
            {
                output += vector[i] + " " + TrainingData.GenreMapping[i] + "\n";
            }
            string name = Path.GetFileName(filePath);
            output += name + " " + TrainingData.GenreMapping[Array.IndexOf(vector, vector.Max())];
            return output;
        }

        private void MainWindow_Load(object sender, EventArgs e)
        {

        }

        private void ChooseFile_Click(object sender, EventArgs e)
        {
            openFileDialog1.InitialDirectory = FileName.Text;
            openFileDialog1.Filter = "mp3 files (*.mp3)|*.mp3|wave files (*.wav)|*.wav|ogg files (*.ogg)|*.ogg|All files (*.*)|*.*";
            openFileDialog1.FilterIndex = 1;
            openFileDialog1.RestoreDirectory = true;
            if (openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                //Get the path of specified file
                FileName.Text = openFileDialog1.FileName;
            }
        }

        private void Start_Click(object sender, EventArgs e)
        {
            Result.Text = GenreClassification(FileName.Text);
        }

        private void Retrain_Click(object sender, EventArgs e)
        {
            RetrainNetwork();
        }
    }
}
