using VisionStudioAI.ML.Training;
using VisionStudioAI.Core.Logging;
using VisionStudioAI.Core.Models;
using VisionStudioAI.Core.Services;
using Microsoft.Extensions.Logging;
using ScottPlot;
using ScottPlot.AxisPanels;
using ScottPlot.Plottables;
using System.ComponentModel;
using static VisionStudioAI.Core.Utils.CoreUtils;
using static VisionStudioAI.ML.Training.TrainingTileGenerator;
using VisionStudioAI.ML.Models.AnomalyDetection;
using VisionStudioAI.ML.Training.AnomalyDetection;

namespace VisionStudioAI.App.Forms
{
    public partial class TrainingForm : Form, ITrainingLogBridge
    {
        private readonly SegmentationTrainingPipeline segmentationPipeline = null!;
        private readonly BinaryAnomalyTrainingPipeline anomalyPipeline = null!;
        private readonly IBinaryAnomalyComplexityConfigProvider anomalyComplexityProvider = null!;
        private readonly IProjectOptionsService projectOptionsService = null!;
        private readonly ILogger<TrainingForm> logger = null!;
        private CancellationTokenSource? cts;
        private DataLogger? trainLossLogger;
        private DataLogger? validationLossLogger;

        public TrainingForm(
            SegmentationTrainingPipeline segmentationPipeline,
            BinaryAnomalyTrainingPipeline anomalyPipeline,
            IBinaryAnomalyComplexityConfigProvider anomalyComplexityProvider,
            IProjectOptionsService projectOptionsService,
            ILogger<TrainingForm> logger)
        {
            // Allows designer to open the form without DI
            if (LicenseManager.UsageMode == LicenseUsageMode.Designtime)
            {
                InitializeComponent();
                return;
            }

            InitializeComponent();
            InitializeDataPlots();

            this.segmentationPipeline = segmentationPipeline;
            this.anomalyPipeline = anomalyPipeline;
            this.anomalyComplexityProvider = anomalyComplexityProvider;
            this.projectOptionsService = projectOptionsService;
            this.logger = logger;
        }

        public async Task StartTrainingRun(
            IProjectPresenter projectPresenter,
            long cpuMemoryBudgetBytes,
            long gpuMemoryBudgetBytes,
            string? anomalyPretrainedWeightsPath = null)
        {
            btnClose.Enabled = false;
            btnStopTraining.Enabled = true;

            pBTrainingForms.Style = ProgressBarStyle.Continuous;

            logger.LogInformation("Preprocessing images...");

            cts?.Dispose();
            cts = new CancellationTokenSource();

            var progress = new Progress<int>(percent =>
            {
                if (!IsDisposed && pBTrainingForms.IsHandleCreated)
                    pBTrainingForms.Value = Math.Clamp(percent, pBTrainingForms.Minimum, pBTrainingForms.Maximum);
            });

            try
            {
                var isAnomalyDetection =
                    projectPresenter.Project.UseCaseMode == AppUseCaseMode.AnomalyDetection;

                // Create folder structure if missing and get paths and clean up any old slices
                projectOptionsService.EnsureAll(projectPresenter.ProjectPath);

                var paths = projectPresenter.Paths;

                await Task.Run(() =>
                {
                    PrepareOutputDirectory(paths.SlicedImages);
                    if (isAnomalyDetection)
                    {
                        GenerateAnomalyTrainingTiles(projectPresenter, progress, cts.Token);
                    }
                    else
                    {
                        PrepareOutputDirectory(paths.SlicedMasks);
                        GenerateTrainingTiles(projectPresenter, progress, cts.Token);
                    }
                }, cts.Token);
                
                logger.LogInformation("Preprocessing complete! Ready to train.");

                cts.Token.ThrowIfCancellationRequested();

                logger.LogInformation("Starting training ...");

                if (isAnomalyDetection)
                {
                    var config = anomalyComplexityProvider.GetConfig(projectPresenter.Project.Settings.TrainModelSettings.ModelComplexity);

                    ConfigureAnomalyWeightInitialization(config, anomalyPretrainedWeightsPath);

                    var artifactDirectory = CreateAnomalyArtifactDirectory(paths.Models);
                    var anomalyProgress = new Progress<BinaryAnomalyTrainingProgress>();

                    anomalyProgress.ProgressChanged += ProcessAnomalyLossData;

                    logger.LogInformation(
                        "Training {Approach} anomaly model with {Complexity} complexity and {Initialization} initialization.",
                        config.Approach,
                        projectPresenter.Project.Settings.TrainModelSettings.ModelComplexity,
                        config.Initialization);

                    await Task.Run(() => anomalyPipeline.RunTraining(
                        projectPresenter,
                        config,
                        artifactDirectory,
                        anomalyProgress,
                        cts.Token,
                        cpuMemoryBudgetBytes,
                        gpuMemoryBudgetBytes), cts.Token);
                }
                else
                {
                    var trainLossProgressReport = new Progress<LossReport>();
                    trainLossProgressReport.ProgressChanged += ProcessLossData;

                    await Task.Run(() => segmentationPipeline.RunTraining(
                        projectPresenter,
                        trainLossProgressReport,
                        cts.Token,
                        cpuMemoryBudgetBytes,
                        gpuMemoryBudgetBytes), cts.Token);
                }

                logger.LogInformation("Training finished!");
            }
            catch (OperationCanceledException)
            {
                logger.LogInformation("Training stopped.");
                if (!IsDisposed && pBTrainingForms.IsHandleCreated)
                    pBTrainingForms.Value = 0;
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Training failed.");
                logger.LogInformation(ex.ToString());
            }
            finally
            {
                btnStopTraining.Enabled = false;
                btnClose.Enabled = true;
            }
        }

        private void InitializeDataPlots()
        {
            trainLossLogger = formsPlotTrainLoss.Plot.Add.DataLogger();
            validationLossLogger = formsPlotTrainLoss.Plot.Add.DataLogger();

            var axisL = (LeftAxis)formsPlotTrainLoss.Plot.Axes.Left;
            var axisB = (BottomAxis)formsPlotTrainLoss.Plot.Axes.Bottom;

            trainLossLogger.Axes.YAxis = axisL;
            validationLossLogger.Axes.YAxis = axisL;

            trainLossLogger.Axes.XAxis = axisB;
            validationLossLogger.Axes.XAxis = axisB;

            if (Application.SystemColorMode == SystemColorMode.Dark)
            {
                formsPlotTrainLoss.Plot.FigureBackground.Color = ScottPlot.Color.FromColor(System.Drawing.Color.Empty);
                formsPlotTrainLoss.Plot.Axes.Color(ScottPlot.Color.FromColor(System.Drawing.Color.White));
            }

            formsPlotTrainLoss.Plot.XLabel("Epochs");
            formsPlotTrainLoss.Plot.YLabel("Loss");

            trainLossLogger.LegendText = "Train";
            validationLossLogger.LegendText = "Validation";

            formsPlotTrainLoss.Plot.Legend.Alignment = Alignment.UpperRight;

            trainLossLogger.Color = ScottPlot.Color.FromColor(System.Drawing.Color.Blue);
            validationLossLogger.Color = ScottPlot.Color.FromColor(System.Drawing.Color.Brown);

            trainLossLogger.LineWidth = 2;
            validationLossLogger.LineWidth = 2;

            trainLossLogger.ManageAxisLimits = true;
            validationLossLogger.ManageAxisLimits = false;
        }

        // UI-safe forwarding
        private async void ProcessLossData(object? sender, LossReport e)
        {
            if (IsDisposed || Disposing || !formsPlotTrainLoss.IsHandleCreated)
                return;

            try
            {
                await formsPlotTrainLoss.InvokeAsync(() =>
                {
                    if (IsDisposed || Disposing)
                        return;

                    trainLossLogger?.Add(e.Epoch, e.TrainLoss > 1 ? 1 : e.TrainLoss);
                    validationLossLogger?.Add(e.Epoch, e.ValidationLoss > 1 ? 1 : e.ValidationLoss);

                    formsPlotTrainLoss.Plot.Axes.AutoScale();
                    formsPlotTrainLoss.Refresh();
                }, cts?.Token ?? CancellationToken.None);
            }
            catch (OperationCanceledException)
            {
            }
            catch (ObjectDisposedException)
            {
            }
        }

        private void ProcessAnomalyLossData(object? sender, BinaryAnomalyTrainingProgress e)
        {
            ProcessLossData(sender, new LossReport
            {
                Epoch = e.Epoch,
                TrainLoss = e.TrainingLoss,
                ValidationLoss = e.ValidationLoss
            });
        }

        private static string CreateAnomalyArtifactDirectory(string modelsDirectory)
        {
            var baseName = "Anomaly_" + DateTime.Now.ToString("yyyy_MM_dd_HH_mm_ss_fff");
            var directory = Path.Combine(modelsDirectory, baseName);
            var suffix = 1;

            while (Directory.Exists(directory))
            {
                directory = Path.Combine(modelsDirectory, baseName + "_" + suffix);
                suffix++;
            }

            return directory;
        }

        private static void ConfigureAnomalyWeightInitialization(
            BinaryAnomalyModelConfig config,
            string? pretrainedWeightsPath)
        {
            if (string.IsNullOrWhiteSpace(pretrainedWeightsPath))
            {
                config.Initialization = WeightInitialization.Random;
                config.PretrainedWeightsPath = null;
                config.FrozenBackboneEpochs = 0;
                return;
            }

            config.Initialization = WeightInitialization.Pretrained;
            config.PretrainedWeightsPath = Path.GetFullPath(pretrainedWeightsPath);
        }


        // UI-safe forwarding (called by TrainingFormLoggerProvider)
        public async void Append(string message)
        {
            if (IsDisposed || Disposing || tbLogTrainForm == null || tbLogTrainForm.IsDisposed || !IsHandleCreated)
                return;

            try
            {
                await InvokeAsync(() =>
                {
                    if (IsDisposed || Disposing || tbLogTrainForm.IsDisposed)
                        return;

                    tbLogTrainForm.AppendText(message + Environment.NewLine);
                }, CancellationToken.None);
            }
            catch (OperationCanceledException)
            {
            }
            catch (ObjectDisposedException)
            {
                // Form is closing, ignore
            }
        }


        private void btnStopTraining_Click(object sender, EventArgs e)
        {
            if (!tbLogTrainForm.IsDisposed)
                tbLogTrainForm.AppendText(Environment.NewLine + "Stopping, please wait ...");

            btnStopTraining.Enabled = false;
            cts?.Cancel();
            btnClose.Enabled = true;
        }

        private void btnClose_Click(object sender, EventArgs e)
        {
            cts?.Cancel();
            this.Close();
        }
    }
}
