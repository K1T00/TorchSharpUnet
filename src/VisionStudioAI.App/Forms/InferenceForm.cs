using VisionStudioAI.ML.Inference;
using VisionStudioAI.ML.Inference.AnomalyDetection;
using VisionStudioAI.Core.Services;
using VisionStudioAI.Core.Models;
using System.ComponentModel;

namespace VisionStudioAI.App.Forms
{
    public partial class InferenceForm : Form
    {
        private CancellationTokenSource? cts;
        private readonly ISegmentationInferencePipeline pipeline = null!;
        private readonly IBinaryAnomalyInferencePipeline anomalyPipeline = null!;
        private readonly IProjectOptionsService projectOptionsService = null!;

        public InferenceForm(
            ISegmentationInferencePipeline pipeline,
            IBinaryAnomalyInferencePipeline anomalyPipeline,
            IProjectOptionsService projectOptionsService)
        {
            // Allows designer to open the form without DI
            if (LicenseManager.UsageMode == LicenseUsageMode.Designtime)
            {
                InitializeComponent();
                return;
            }
            InitializeComponent();

            this.pipeline = pipeline!;
            this.anomalyPipeline = anomalyPipeline!;
            this.projectOptionsService = projectOptionsService!;
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            btnCancel.Enabled = false;
            cts?.Cancel();
        }

        public async Task StartInferenceRun(IProjectPresenter projectPresenter, string selectedModelPath)
        {
            var keepDeviceFromUi = projectPresenter.Project.Settings.TrainModelSettings.Device;

            progressBar.Style = ProgressBarStyle.Continuous;
            cts = new CancellationTokenSource();
            var progress = new Progress<int>(percent => progressBar.Value = percent);

            try
            {
                if (projectPresenter.Project.UseCaseMode == AppUseCaseMode.AnomalyDetection)
                {
                    await Task.Run(
                        () => anomalyPipeline.RunInference(
                            projectPresenter,
                            selectedModelPath,
                            progress,
                            cts.Token),
                        cts.Token);
                }
                else
                {
                    projectPresenter.UpdateTrainingSettings(
                        projectOptionsService.ExtractMetadataFilePath(selectedModelPath));
                    projectPresenter.Project.Settings.TrainModelSettings.Device = keepDeviceFromUi;

                    await Task.Run(
                        () => pipeline.RunInference(
                            projectPresenter,
                            selectedModelPath,
                            progress,
                            cts.Token),
                        cts.Token);
                }

                btnCancel.Enabled = false;
                this.Close();
            }
            catch (OperationCanceledException)
            {
                // User canceled inference
                this.Close();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.ToString());
            }
            finally
            {
                cts?.Dispose();
                cts = null;
            }
        }
    }
}
