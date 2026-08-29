using VisionStudioAI.Core.Services;
using VisionStudioAI.Core.Models;
using static VisionStudioAI.Core.Utils.CoreUtils;

namespace VisionStudioAI.App.Forms
{
    public partial class TrainedModelsForm : Form
    {
        required public ProjectPaths Paths { get; set; }
        public AppUseCaseMode UseCaseMode { get; set; } = AppUseCaseMode.Segmentation;
        public string? SelectedModelFileName { get; private set; }

        public TrainedModelsForm()
        {
            InitializeComponent();
        }

        private void btnChooseModel_Click(object sender, EventArgs e)
        {
            if (lbTrainedModels.SelectedItem is string path)
            {
                this.SelectedModelFileName = path;
                DialogResult = DialogResult.OK;
                Close();
            }
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            DialogResult = DialogResult.Cancel;
            Close();
        }

        private void TrainedModelsForm_Load(object sender, EventArgs e)
        {
            if (Paths == null)
            {
                MessageBox.Show(
                    "Project paths are not set.",
                    "Error",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error);
                Close();
                return;
            }

            if (string.IsNullOrWhiteSpace(Paths.Models) || !Directory.Exists(Paths.Models))
            {
                MessageBox.Show(
                    "Models directory is not configured.",
                    "Error",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error);
                Close();
                return;
            }

            RefreshModels();
        }

        private void btnDeleteModel_Click(object sender, EventArgs e)
        {
            if (lbTrainedModels.SelectedItem is not string modelName)
                return;

            var confirm = MessageBox.Show(
                this,
               "Delete model?",
               "Warning",
               MessageBoxButtons.YesNo,
               MessageBoxIcon.Question);

            if (confirm != DialogResult.Yes)
                return;

            if (UseCaseMode == AppUseCaseMode.AnomalyDetection)
            {
                DeleteAnomalyModel(modelName);
            }
            else
            {
                var modelFile = Path.Combine(Paths.Models, modelName + Paths.ModelExt);
                var settingsFile = Path.Combine(
                    Paths.Models,
                    modelName.Replace(Paths.ModelSub, Paths.ModelSettingsSub) + ".json");

                TryDeleteFile(modelFile);
                TryDeleteFile(settingsFile);
            }

            RefreshModels();
        }

        private void RefreshModels()
        {
            List<string> models;

            if (UseCaseMode == AppUseCaseMode.AnomalyDetection)
            {
                models = Directory.EnumerateDirectories(Paths.Models)
                    .Where(directory => File.Exists(Path.Combine(directory, "anomaly-model.json")))
                    .OrderByDescending(Directory.GetLastWriteTimeUtc)
                    .Select(directory => new DirectoryInfo(directory).Name)
                    .ToList();
            }
            else
            {
                models = Directory.GetFiles(Paths.Models, Paths.ModelSub + "*" + Paths.ModelExt)
                    .Select(file => Path.GetFileNameWithoutExtension(file)!)
                    .OrderByDescending(path => File.GetCreationTime(
                        Path.Combine(Paths.Models, path + Paths.ModelExt)))
                    .ToList();
            }

            lbTrainedModels.DataSource = models;
        }

        private void DeleteAnomalyModel(string modelName)
        {
            var modelsRoot = Path.GetFullPath(Paths.Models)
                .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) +
                Path.DirectorySeparatorChar;
            var artifactDirectory = Path.GetFullPath(Path.Combine(Paths.Models, modelName));

            if (!artifactDirectory.StartsWith(modelsRoot, StringComparison.OrdinalIgnoreCase) ||
                !File.Exists(Path.Combine(artifactDirectory, "anomaly-model.json")))
            {
                MessageBox.Show(
                    "The selected anomaly model directory is invalid.",
                    "Delete model",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error);
                return;
            }

            try
            {
                Directory.Delete(artifactDirectory, recursive: true);
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    $"Could not delete anomaly model: {ex.Message}",
                    "Delete model",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error);
            }
        }
    }
}
