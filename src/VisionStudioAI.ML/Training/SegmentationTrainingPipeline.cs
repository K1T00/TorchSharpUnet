using VisionStudioAI.ML.Models;
using VisionStudioAI.ML.Utils;
using VisionStudioAI.Core.Models;
using VisionStudioAI.Core.Services;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;
using static VisionStudioAI.ML.IO.ModelMetaData;
using static VisionStudioAI.ML.Utils.AiUtils;
using static VisionStudioAI.ML.Utils.BatchSizeEstimator;
using static VisionStudioAI.ML.Utils.DatasetStatistics;
using static VisionStudioAI.ML.Utils.CudaOps.NativeTorchCudaOps;
using static VisionStudioAI.ML.Utils.ImageProcessing.ImageAnalysis;
using static TorchSharp.torch;
using static TorchSharp.torch.utils.data;


namespace VisionStudioAI.ML.Training
{
    /// <summary>
    /// High-level pipeline that orchestrates training for TorchSharp modules produced by an ISegmentationModelFactory.
    /// </summary>
    public class SegmentationTrainingPipeline
    {
        private const int DefaultWorkerCount = 4;
        // Note: Batch size variation (450+50 -> 250+250) does not skew results significantly, so we use a fixed optimizer type for estimation
        // But BatchNorm2d should be used with a minimum batch size
        private const int MinBatchForBatchNorm = 8;

        private readonly ILogger<SegmentationTrainingPipeline> logger;
        private readonly ISegmentationModelFactory modelFactory;
        private readonly SegmentationTrainer trainer;
        private readonly IProjectOptionsService projectOptionsService;
        private readonly JsonSerializerOptions jsonOptions;
        private Device device;
        private readonly IModelComplexityConfigProvider complexityProvider;


        public SegmentationTrainingPipeline(
            ILogger<SegmentationTrainingPipeline> logger,
            ILoggerFactory loggerFactory,
            ISegmentationModelFactory modelFactory,
            IProjectOptionsService projectOptionsService,
            JsonSerializerOptions jsonOptions,
            IModelComplexityConfigProvider complexityProvider)
        {
            this.logger = logger;
            this.modelFactory = modelFactory;
            this.trainer = new SegmentationTrainer(loggerFactory.CreateLogger<SegmentationTrainer>());
            this.projectOptionsService = projectOptionsService;
            this.jsonOptions = jsonOptions;
            this.complexityProvider = complexityProvider;
        }

        public async Task RunTraining(IProjectPresenter project, IProgress<LossReport> lossProgress, CancellationToken ct, long cpuMemoryBudgetBytes, long gpuMemoryBudgetBytes)
        {
            try
            {
                device = ResolveDevice(project.Project.Settings);
                logger.LogInformation("Starting training on: {Device}", device);

                // Build training config based on model complexity
                var cfg = BuildModelConfig(project);

                using (var loaderInfo = BuildDataLoaders(project, cfg, cpuMemoryBudgetBytes, gpuMemoryBudgetBytes))
                {
                    var segmentationMode = GetSegmentationMode(project);

                    var inChannels = project.Project.Settings.PreprocessingSettings.TrainAsGreyscale ? 1 : 3;
                    var featureCount = project.Project.Features.Count;

                    using (var model = modelFactory.Create(inChannels, featureCount, cfg, device))
                    using (var optimization = TrainingOptimizationFactory.Build(model, project.Project.Settings, cfg, loaderInfo.BatchSize))
                    {
                        var ctx = new TrainingContext
                        {
                            Device = device,
                            Model = model,
                            Optimization = optimization,
                            TrainLoader = loaderInfo.TrainLoader,
                            ValLoader = loaderInfo.ValLoader,
                            Settings = project.Project.Settings,
                            StoppingMonitor = new TrainingStopMonitor(project.Project.Settings.TrainingStoppingSettings),
                            SegmentationMode = segmentationMode
                        };

                        try
                        {
                            // Training loop
                            await trainer.RunTrainer(ctx, lossProgress, ct).ConfigureAwait(false);

                            logger.LogInformation("Training done.");

                            // Save model and metadata
                            SaveModelAndMetadata(model, project, projectOptionsService, jsonOptions, logger);
                        }
                        catch (OperationCanceledException)
                        {
                            logger.LogInformation("Training canceled. Saving current model state.");

                            SaveModelAndMetadata(model, project, projectOptionsService, jsonOptions, logger);

                            throw;
                        }
                    }
                }
            }
            finally
            {
                if (device.type == DeviceType.CUDA)
                {
                    EmptyCudaCache();
                }
            }
        }

        private DataLoaderBuildResult BuildDataLoaders(IProjectPresenter project, SegmentationModelConfig cfg, long cpuMemoryBudgetBytes, long gpuMemoryBudgetBytes)
        {
            var settings = project.Project.Settings;
            var preprocessing = settings.PreprocessingSettings;

            var availableMemory =
                project.Project.Settings.TrainModelSettings.Device == ComputeDevice.Gpu
                ? gpuMemoryBudgetBytes
                : cpuMemoryBudgetBytes;

            var sliceSize = preprocessing.SliceSize;
            var numChannels = preprocessing.TrainAsGreyscale ? 1 : 3;

            var trainPairs = project.GetSlicedTrainingPairs(DatasetSplit.Train);
            var valPairs = project.GetSlicedTrainingPairs(DatasetSplit.Validate);

            if (trainPairs.Count == 0)
                throw new InvalidOperationException("No sliced training image/mask pairs found.");

            EnsureNormalization(project, trainPairs.Select(p => p.imagePath));

            var batchSize = EstimateBatchSize(
                cfg,
                sliceSize,
                sliceSize,
                numChannels,
                availableMemory,
                settings.TrainModelSettings.Device,
                OptimizerType.AdamW);

            // BatchNorm should not run with very small batches.
            if (!cfg.UseInstanceNorm)
            {
                batchSize = AdjustBatchSizeIfNecessary(batchSize, trainPairs.Count, MinBatchForBatchNorm);
            }

            var augmentations = ImageAugmentations.BuildAugmentations(settings.AugmentationSettings);
            var trainDataset = BuildTrainingDataset(project, trainPairs, augmentations, cfg);

            // Validation dataset is not augmented in any mode, but we still need to apply preprocessing (e.g. normalization)
            var validationDataset = new SegmentationDataset(valPairs, project, null, cfg);

            return new DataLoaderBuildResult
            {
                BatchSize = batchSize,
                TrainDataset = trainDataset,
                ValDataset = validationDataset,
                TrainLoader = DataLoader(trainDataset, batchSize, shuffle: true, device: CPU, num_worker: DefaultWorkerCount),
                ValLoader = DataLoader(validationDataset, batchSize, shuffle: false, device: CPU, num_worker: DefaultWorkerCount)
            };
        }

        private void EnsureNormalization(IProjectPresenter project, IEnumerable<string> trainingImagePaths)
        {
            var preprocessing = project.Project.Settings.PreprocessingSettings;

            preprocessing.Normalization = preprocessing.TrainAsGreyscale
                ? ComputeGreyStats(trainingImagePaths)
                : ComputeRgbStats(trainingImagePaths);
        }

        private SegmentationModelConfig BuildModelConfig(IProjectPresenter project)
        {
            var settings = project.Project.Settings;
            var preprocessing = settings.PreprocessingSettings;
            var trainSettings = settings.TrainModelSettings;

            return complexityProvider.GetConfig(trainSettings.ModelComplexity);
        }

        private SegmentationMode GetSegmentationMode(IProjectPresenter project)
        {
            return project.Project.Features.Count == 1
                ? SegmentationMode.Binary
                : SegmentationMode.Multiclass;
        }

        private static Dataset BuildTrainingDataset(IProjectPresenter project, IReadOnlyList<(string imagePath, string maskPath)> trainPairs, IPairedTransform augmentations, SegmentationModelConfig cfg)
        {
            switch (project.Project.Settings.AugmentationSettings.AugmentationMode)
            {
                case AugmentationMode.Standard:
                    return new SegmentationDataset(trainPairs, project, augmentations, cfg);

                case AugmentationMode.Duplication:
                    return new ConcatSegmentationDataset(
                        new SegmentationDataset(trainPairs, project, null, cfg),
                        new SegmentationDataset(trainPairs, project, augmentations, cfg));

                case AugmentationMode.FeatureAware:
                    return new ConcatSegmentationDataset(
                        new SegmentationDataset(trainPairs, project, null, cfg),
                        new SegmentationDataset(trainPairs, project, augmentations, cfg, IsBlobInImage));

                default:
                    return new SegmentationDataset(trainPairs, project, augmentations, cfg);
            }
        }

        private sealed class DataLoaderBuildResult : IDisposable
        {
            private bool disposed;

            public int BatchSize { get; set; }
            public Dataset TrainDataset { get; set; }
            public Dataset ValDataset { get; set; }
            public DataLoader TrainLoader { get; set; }
            public DataLoader ValLoader { get; set; }

            public void Dispose()
            {
                if (disposed)
                    return;

                DisposeIfSupported(TrainLoader);
                DisposeIfSupported(ValLoader);
                TrainDataset?.Dispose();
                ValDataset?.Dispose();

                disposed = true;
            }

            private static void DisposeIfSupported(object value)
            {
                var disposable = value as IDisposable;
                if (disposable != null)
                {
                    disposable.Dispose();
                }
            }
        }

    }
}
