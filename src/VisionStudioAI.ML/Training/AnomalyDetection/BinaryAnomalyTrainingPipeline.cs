using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TorchSharp;
using VisionStudioAI.Core.Models;
using VisionStudioAI.Core.Services;
using VisionStudioAI.ML.Inference.AnomalyDetection;
using VisionStudioAI.ML.IO;
using VisionStudioAI.ML.Models.AnomalyDetection;
using static TorchSharp.torch;
using static VisionStudioAI.ML.Utils.BatchSizeEstimator;
using static VisionStudioAI.ML.Utils.DatasetStatistics;

namespace VisionStudioAI.ML.Training.AnomalyDetection
{
    /// <summary>
    /// Trains an OK-only self-supervised feature extractor, builds a normal
    /// descriptor memory bank, calibrates it, and persists one inference package.
    /// </summary>
    public sealed class BinaryAnomalyTrainingPipeline
    {
        private readonly IBinaryAnomalyModelFactory modelFactory;
        private readonly BinaryAnomalyTrainer trainer;
        private readonly SimilarityMemoryBankBuilder memoryBankBuilder;
        private readonly AnomalyCalibrationBuilder calibrationBuilder;
        private readonly ILogger<BinaryAnomalyTrainingPipeline> logger;
        private readonly JsonSerializerOptions jsonOptions;

        public BinaryAnomalyTrainingPipeline(
            IBinaryAnomalyModelFactory modelFactory,
            BinaryAnomalyTrainer trainer,
            SimilarityMemoryBankBuilder memoryBankBuilder,
            AnomalyCalibrationBuilder calibrationBuilder,
            ILogger<BinaryAnomalyTrainingPipeline> logger,
            JsonSerializerOptions jsonOptions)
        {
            this.modelFactory = modelFactory;
            this.trainer = trainer;
            this.memoryBankBuilder = memoryBankBuilder;
            this.calibrationBuilder = calibrationBuilder;
            this.logger = logger;
            this.jsonOptions = jsonOptions;
        }

        public async Task RunTraining(
            IProjectPresenter project,
            BinaryAnomalyModelConfig config,
            string artifactDirectory,
            IProgress<BinaryAnomalyTrainingProgress> progress,
            CancellationToken cancellationToken,
            long cpuMemoryBudgetBytes = 0,
            long gpuMemoryBudgetBytes = 0)
        {
            if (project == null) throw new ArgumentNullException(nameof(project));
            if (string.IsNullOrWhiteSpace(artifactDirectory))
                throw new ArgumentException("Artifact directory is required.", nameof(artifactDirectory));
            config.Validate();
            EnsurePretrainedWeights(config);
            if (config.Approach != AnomalyDetectionApproach.SimilarityBased)
            {
                throw new NotSupportedException(
                    "Golden Sample anomaly detection is reserved but not implemented yet.");
            }

            var training = project.GetAnomalyTrainingSamples(DatasetSplit.Train);
            var validation = project.GetAnomalyTrainingSamples(DatasetSplit.Validate);
            EnsureOkSamples(training, "training");
            EnsureOkSamples(validation, "validation");

            var settings = project.Project.Settings;
            EnsureNormalization(settings.PreprocessingSettings, training, config);
            var device = new Device(settings.TrainModelSettings.Device == ComputeDevice.Gpu
                ? DeviceType.CUDA
                : DeviceType.CPU);
            var channels = settings.PreprocessingSettings.TrainAsGreyscale ? 1 : 3;
            var maxOriginalTilesPerBatch = EstimateTileBatchSize(
                training,
                settings,
                config,
                channels,
                cpuMemoryBudgetBytes,
                gpuMemoryBudgetBytes);

            logger?.LogInformation(
                "Similarity training capacity: {TileBatchSize} original OK tiles " +
                "({SyntheticBatchSize} tensors after CutPaste expansion).",
                maxOriginalTilesPerBatch,
                maxOriginalTilesPerBatch * SimilarityAnomalyModel.PretextClassCount);

            Directory.CreateDirectory(artifactDirectory);
            using (var model = modelFactory.Create(channels, config, device))
            {
                var maxEpochs = Math.Max(1, settings.TrainingStoppingSettings.MaxIterationCount);
                try
                {
                    await trainer.TrainAsync(
                        model,
                        training,
                        validation,
                        settings.PreprocessingSettings,
                        config,
                        device,
                        maxEpochs,
                        maxOriginalTilesPerBatch,
                        progress,
                        cancellationToken).ConfigureAwait(false);

                    BuildAndSavePackage(
                        model,
                        training,
                        validation,
                        settings,
                        config,
                        device,
                        maxOriginalTilesPerBatch,
                        artifactDirectory,
                        cancellationToken);
                }
                catch (OperationCanceledException)
                {
                    logger?.LogInformation(
                        "Similarity training canceled. Saving the current encoder state.");
                    BuildAndSavePackage(
                        model,
                        training,
                        validation,
                        settings,
                        config,
                        device,
                        maxOriginalTilesPerBatch,
                        artifactDirectory,
                        CancellationToken.None);
                    throw;
                }
            }
        }

        private void BuildAndSavePackage(
            SimilarityAnomalyModel model,
            IReadOnlyList<AnomalyImageSample> training,
            IReadOnlyList<AnomalyImageSample> validation,
            DeepLearningSettings settings,
            BinaryAnomalyModelConfig config,
            Device device,
            int tileBatchSize,
            string artifactDirectory,
            CancellationToken cancellationToken)
        {
            using (var memoryBank = memoryBankBuilder.Build(
                model,
                training,
                settings.PreprocessingSettings,
                config,
                device,
                tileBatchSize,
                cancellationToken))
            {
                var calibration = calibrationBuilder.Build(
                    model,
                    memoryBank,
                    validation,
                    settings.PreprocessingSettings,
                    config,
                    device,
                    cancellationToken);
                var package = new BinaryAnomalyModelPackage
                {
                    ModelConfig = config,
                    Settings = settings,
                    Calibration = calibration
                };

                model.save(Path.Combine(
                    artifactDirectory,
                    package.FeatureExtractorWeightsFile));
                memoryBank.Save(Path.Combine(
                    artifactDirectory,
                    package.MemoryBankFile));
                File.WriteAllText(
                    Path.Combine(artifactDirectory, "anomaly-model.json"),
                    JsonSerializer.Serialize(package, jsonOptions));
            }

            logger?.LogInformation(
                "Similarity anomaly model saved to {ArtifactDirectory}.",
                artifactDirectory);
        }

        private static void EnsureNormalization(
            PreprocessingSettings preprocessing,
            IReadOnlyList<AnomalyImageSample> samples,
            BinaryAnomalyModelConfig config)
        {
            if (config.Initialization == WeightInitialization.Pretrained &&
                config.UseImageNetNormalizationForPretrained)
            {
                preprocessing.Normalization = preprocessing.TrainAsGreyscale
                    ? new NormalizationSettings { Mean = new[] { 0.449f }, Std = new[] { 0.226f } }
                    : new NormalizationSettings
                    {
                        Mean = new[] { 0.485f, 0.456f, 0.406f },
                        Std = new[] { 0.229f, 0.224f, 0.225f }
                    };
                return;
            }

            var okPaths = samples
                .Where(sample => sample.Label == AnomalyLabel.Ok)
                .SelectMany(sample => sample.TilePaths);
            preprocessing.Normalization = preprocessing.TrainAsGreyscale
                ? ComputeGreyStats(okPaths)
                : ComputeRgbStats(okPaths);
        }

        private int EstimateTileBatchSize(
            IReadOnlyList<AnomalyImageSample> training,
            DeepLearningSettings settings,
            BinaryAnomalyModelConfig config,
            int channels,
            long cpuMemoryBudgetBytes,
            long gpuMemoryBudgetBytes)
        {
            var okTileCount = training
                .Where(sample => sample.Label == AnomalyLabel.Ok)
                .Sum(sample => sample.TilePaths.Count);
            var availableMemory = settings.TrainModelSettings.Device == ComputeDevice.Gpu
                ? gpuMemoryBudgetBytes
                : cpuMemoryBudgetBytes;
            if (availableMemory <= 0)
                return Math.Max(1, Math.Min(okTileCount, 16));

            var estimated = EstimateAnomalyTileBatchSize(
                config,
                settings.PreprocessingSettings.SliceSize,
                settings.PreprocessingSettings.SliceSize,
                channels,
                availableMemory,
                settings.TrainModelSettings.Device,
                OptimizerType.AdamW);
            return Math.Max(1, Math.Min(estimated, okTileCount));
        }

        private static void EnsureOkSamples(
            IReadOnlyList<AnomalyImageSample> samples,
            string splitName)
        {
            if (!samples.Any(sample =>
                sample.Label == AnomalyLabel.Ok && sample.TilePaths.Count > 0))
            {
                throw new InvalidOperationException(
                    "The " + splitName + " split requires at least one OK image.");
            }
        }

        private void EnsurePretrainedWeights(BinaryAnomalyModelConfig config)
        {
            if (config.Initialization != WeightInitialization.Pretrained)
                return;

            var weightsPath = Path.GetFullPath(config.PretrainedWeightsPath);
            if (!File.Exists(weightsPath))
            {
                throw new FileNotFoundException(
                    "The pretrained anomaly backbone weights were not found.",
                    weightsPath);
            }

            config.PretrainedWeightsPath = weightsPath;
            logger?.LogInformation(
                "Initializing the anomaly ResNet-18 backbone from {WeightsPath}.",
                weightsPath);
        }
    }
}
