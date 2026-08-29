using VisionStudioAI.Core.Models;
using System.Drawing.Drawing2D;
using System.Text.Json;

namespace VisionStudioAI.App.Forms
{
    public partial class StartupModeForm : Form
    {
        private readonly Image segmentationDefaultImage;
        private readonly Image segmentationHoverImage;
        private readonly Image objectDetectionDefaultImage;
        private readonly Image objectDetectionHoverImage;
        private readonly Image anomalyDetectionDefaultImage;
        private readonly Image anomalyDetectionHoverImage;

        private readonly JsonSerializerOptions jsonOptions;


        public AppUseCaseMode? SelectedMode { get; private set; }

        public string? StartupProjectPath { get; private set; }

        public StartupModeForm(JsonSerializerOptions jsonOptions)
        {
            this.jsonOptions = jsonOptions;

            InitializeComponent();
            FormBorderStyle = FormBorderStyle.FixedSingle;
            MaximizeBox = false;

            segmentationDefaultImage = Properties.Resources.Segmentation;
            objectDetectionDefaultImage = Properties.Resources.ObjectDetection;
            anomalyDetectionDefaultImage = Properties.Resources.AnomalyDetection;
            segmentationHoverImage = CreateHoverImage(segmentationDefaultImage);
            objectDetectionHoverImage = CreateHoverImage(objectDetectionDefaultImage);
            anomalyDetectionHoverImage = CreateHoverImage(anomalyDetectionDefaultImage);
            btnSegmentation.BackgroundImage = segmentationDefaultImage;
            btnObjectDetection.BackgroundImage = objectDetectionDefaultImage;
            btnAnomalyDetection.BackgroundImage = anomalyDetectionDefaultImage;
            btnSegmentation.Cursor = Cursors.Hand;
            btnObjectDetection.Cursor = Cursors.Hand;
            btnAnomalyDetection.Cursor = Cursors.Hand;

            btnSegmentation.MouseEnter += (_, _) => btnSegmentation.BackgroundImage = segmentationHoverImage;
            btnSegmentation.MouseLeave += (_, _) => btnSegmentation.BackgroundImage = segmentationDefaultImage;
            btnObjectDetection.MouseEnter += (_, _) => btnObjectDetection.BackgroundImage = objectDetectionHoverImage;
            btnObjectDetection.MouseLeave += (_, _) => btnObjectDetection.BackgroundImage = objectDetectionDefaultImage;
            btnAnomalyDetection.MouseEnter += (_, _) => btnAnomalyDetection.BackgroundImage = anomalyDetectionHoverImage;
            btnAnomalyDetection.MouseLeave += (_, _) => btnAnomalyDetection.BackgroundImage = anomalyDetectionDefaultImage;
        }

        private void btnSegmentation_Click(object sender, EventArgs e)
        {
            SelectedMode = AppUseCaseMode.Segmentation;
            DialogResult = DialogResult.OK;
            Close();
        }

        private void btnObjectDetection_Click(object sender, EventArgs e)
        {
            SelectedMode = AppUseCaseMode.ObjectDetection;
            DialogResult = DialogResult.OK;
            Close();
        }

        private void btnAnomalyDetection_Click(object sender, EventArgs e)
        {
            SelectedMode = AppUseCaseMode.AnomalyDetection;
            DialogResult = DialogResult.OK;
            Close();
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            DialogResult = DialogResult.Cancel;
            Close();
        }

        private void btnLoadProject_Click(object sender, EventArgs e)
        {
            using var ofd = new OpenFileDialog();
            ofd.Title = "Load Project File";
            ofd.Filter = "Project JSON|*.json|All files|*.*";

            if (ofd.ShowDialog(this) != DialogResult.OK)
                return;

            try
            {
                var json = File.ReadAllText(ofd.FileName);
                var project = JsonSerializer.Deserialize<DeepLearningProject>(json, jsonOptions);

                if (project == null)
                {
                    MessageBox.Show(this, "Could not read project file.", "Error",
                        MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }

                StartupProjectPath = ofd.FileName;
                SelectedMode = project.UseCaseMode;

                DialogResult = DialogResult.OK;
                Close();
            }
            catch (Exception ex)
            {
                MessageBox.Show(this, $"Error loading project: {ex.Message}", "Error",
                    MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private static Bitmap CreateHoverImage(Image baseImage)
        {
            var hoverImage = new Bitmap(baseImage.Width, baseImage.Height);

            using (var graphics = Graphics.FromImage(hoverImage))
            using (var overlayBrush = new SolidBrush(Color.FromArgb(72, 30, 144, 255)))
            using (var borderPen = new Pen(Color.FromArgb(255, 0, 102, 204), 10))
            {
                graphics.SmoothingMode = SmoothingMode.AntiAlias;
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

                graphics.DrawImage(baseImage, 0, 0, baseImage.Width, baseImage.Height);
                graphics.FillRectangle(overlayBrush, 0, 0, baseImage.Width, baseImage.Height);
                graphics.DrawRectangle(borderPen, 5, 5, baseImage.Width - 10, baseImage.Height - 10);
            }

            return hoverImage;
        }


    }
}
