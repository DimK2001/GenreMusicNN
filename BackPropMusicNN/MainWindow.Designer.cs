namespace GenreMusicNN
{
    partial class MainWindow
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(MainWindow));
            FileName = new TextBox();
            label1 = new Label();
            openFileDialog1 = new OpenFileDialog();
            ChooseFile = new Button();
            Start = new Button();
            ResultsText = new Label();
            Result = new Label();
            Retrain = new Button();
            WarningLabel = new Label();
            ModelTest = new Button();
            SuspendLayout();
            // 
            // FileName
            // 
            FileName.Location = new Point(100, 6);
            FileName.Name = "FileName";
            FileName.Size = new Size(421, 23);
            FileName.TabIndex = 0;
            // 
            // label1
            // 
            label1.AutoSize = true;
            label1.Location = new Point(12, 9);
            label1.Name = "label1";
            label1.Size = new Size(82, 15);
            label1.TabIndex = 1;
            label1.Text = "Выбор файла";
            // 
            // openFileDialog1
            // 
            openFileDialog1.FileName = "openFileDialog1";
            // 
            // ChooseFile
            // 
            ChooseFile.Location = new Point(527, 6);
            ChooseFile.Name = "ChooseFile";
            ChooseFile.Size = new Size(65, 23);
            ChooseFile.TabIndex = 2;
            ChooseFile.Text = "Выбрать";
            ChooseFile.UseVisualStyleBackColor = true;
            ChooseFile.Click += ChooseFile_Click;
            // 
            // Start
            // 
            Start.Font = new Font("Segoe UI", 12F);
            Start.Location = new Point(12, 52);
            Start.Name = "Start";
            Start.Size = new Size(186, 55);
            Start.TabIndex = 3;
            Start.Text = "Запуск";
            Start.UseVisualStyleBackColor = true;
            Start.Click += Start_Click;
            // 
            // ResultsText
            // 
            ResultsText.AutoSize = true;
            ResultsText.Location = new Point(12, 135);
            ResultsText.Name = "ResultsText";
            ResultsText.Size = new Size(114, 15);
            ResultsText.TabIndex = 4;
            ResultsText.Text = "Результаты поиска:";
            // 
            // Result
            // 
            Result.AutoSize = true;
            Result.Location = new Point(31, 161);
            Result.Name = "Result";
            Result.Size = new Size(0, 15);
            Result.TabIndex = 5;
            // 
            // Retrain
            // 
            Retrain.Font = new Font("Segoe UI", 12F);
            Retrain.Location = new Point(231, 52);
            Retrain.Name = "Retrain";
            Retrain.Size = new Size(186, 55);
            Retrain.TabIndex = 6;
            Retrain.Text = "Переобучение";
            Retrain.UseVisualStyleBackColor = true;
            Retrain.Click += Retrain_Click;
            // 
            // WarningLabel
            // 
            WarningLabel.Location = new Point(423, 52);
            WarningLabel.Name = "WarningLabel";
            WarningLabel.Size = new Size(177, 141);
            WarningLabel.TabIndex = 7;
            WarningLabel.Text = resources.GetString("WarningLabel.Text");
            // 
            // ModelTest
            // 
            ModelTest.Font = new Font("Segoe UI", 12F);
            ModelTest.Location = new Point(423, 196);
            ModelTest.Name = "ModelTest";
            ModelTest.Size = new Size(169, 55);
            ModelTest.TabIndex = 8;
            ModelTest.Text = "Тест моделей";
            ModelTest.UseVisualStyleBackColor = true;
            ModelTest.Click += ModelTest_Click;
            // 
            // MainWindow
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(599, 380);
            Controls.Add(ModelTest);
            Controls.Add(WarningLabel);
            Controls.Add(Retrain);
            Controls.Add(Result);
            Controls.Add(ResultsText);
            Controls.Add(Start);
            Controls.Add(ChooseFile);
            Controls.Add(label1);
            Controls.Add(FileName);
            Name = "MainWindow";
            Text = "Определитель жанра музыки";
            Load += MainWindow_Load;
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private TextBox FileName;
        private Label label1;
        private OpenFileDialog openFileDialog1;
        private Button ChooseFile;
        private Button Start;
        private Label ResultsText;
        private Label Result;
        private Button Retrain;
        private Label WarningLabel;
        private Button ModelTest;
    }
}