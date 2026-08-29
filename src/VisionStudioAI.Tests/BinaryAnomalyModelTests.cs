using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using TorchSharp;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using OpenCvSharp;
using VisionStudioAI.Core;
using VisionStudioAI.Core.Configuration;
using VisionStudioAI.Core.Models;
using VisionStudioAI.Core.Services;
using VisionStudioAI.ML;
using VisionStudioAI.ML.Inference.AnomalyDetection;
using VisionStudioAI.ML.IO;
using VisionStudioAI.ML.Models.AnomalyDetection;
using VisionStudioAI.ML.Training;
using VisionStudioAI.ML.Training.AnomalyDetection;
using VisionStudioAI.ML.Utils;
using Xunit;
using static TorchSharp.torch;

namespace VisionStudioAI.Tests
{
    public sealed class BinaryAnomalyModelTests
    {
        [Fact]
        public void AnomalyBatchEstimator_UsesAdditionalAvailableMemory()
        {
            var config = CreateSmallConfig();
            var twoGb = BatchSizeEstimator.EstimateAnomalyTileBatchSize(
                config, 256, 256, 3, 2L * 1024 * 1024 * 1024,
                ComputeDevice.Cpu, BatchSizeEstimator.OptimizerType.AdamW);
            var fourGb = BatchSizeEstimator.EstimateAnomalyTileBatchSize(
                config, 256, 256, 3, 4L * 1024 * 1024 * 1024,
                ComputeDevice.Cpu, BatchSizeEstimator.OptimizerType.AdamW);

            Assert.True(twoGb > 0);
            Assert.True(fourGb > twoGb);
        }

        [Fact]
        public void AnomalyBatchEstimator_DefaultGpuTargetUsesMoreCapacityThanLegacyConservativeTarget()
        {
            var config = CreateSmallConfig();
            const long eightGb = 8L * 1024 * 1024 * 1024;
            var current = BatchSizeEstimator.EstimateAnomalyTileBatchSize(
                config, 128, 128, 3, eightGb,
                ComputeDevice.Gpu, BatchSizeEstimator.OptimizerType.AdamW);
            var legacy = BatchSizeEstimator.EstimateAnomalyTileBatchSize(
                config, 128, 128, 3, eightGb,
                ComputeDevice.Gpu, BatchSizeEstimator.OptimizerType.AdamW,
                targetUtilization: 0.78);

            Assert.True(current > legacy);
        }

        [Fact]
        public async Task AnomalyTrainer_TrainsOnlyOkTilesWithSyntheticPretextClasses()
        {
            var directory = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(directory);

            try
            {
                var samples = new List<AnomalyImageSample>();
                for (var index = 0; index < 2; index++)
                {
                    var path = Path.Combine(directory, index + ".png");
                    using (var image = new Mat(64, 64, MatType.CV_8UC1))
                    {
                        Cv2.Randu(
                            image,
                            OpenCvSharp.Scalar.All(0),
                            OpenCvSharp.Scalar.All(255));
                        Cv2.ImWrite(path, image);
                    }

                    samples.Add(new AnomalyImageSample
                    {
                        ImageId = Guid.NewGuid(),
                        Label = index == 0 ? AnomalyLabel.Ok : AnomalyLabel.NotOk,
                        TilePaths = new List<string> { path }
                    });
                }

                var config = CreateSmallConfig();
                var preprocessing = new PreprocessingSettings
                {
                    TrainAsGreyscale = true,
                    Normalization = new NormalizationSettings
                    {
                        Mean = new[] { 0.5f },
                        Std = new[] { 0.25f }
                    }
                };

                using (var model = new BinaryAnomalyModelFactory().Create(1, config, CPU))
                {
                    var trainer = new BinaryAnomalyTrainer(
                        NullLogger<BinaryAnomalyTrainer>.Instance);
                    await trainer.TrainAsync(
                        model,
                        samples,
                        samples,
                        preprocessing,
                        config,
                        CPU,
                        maxEpochs: 1,
                        maxOriginalTilesPerBatch: 2,
                        progress: null,
                        CancellationToken.None);
                }
            }
            finally
            {
                if (Directory.Exists(directory))
                    Directory.Delete(directory, recursive: true);
            }
        }

        [Fact]
        public void TopKPooling_UsesHighestSpatialValues()
        {
            using (var logits = tensor(new[] { 1f, 2f, 3f, 4f }).reshape(1, 1, 2, 2))
            using (var pooled = AnomalyTopKPooling.PoolImageBag(logits, 0.5))
            {
                Assert.Equal(3.5f, pooled.ToSingle(), 5);
            }
        }

        [Fact]
        public void RandomModel_ReturnsPretextLogitsAndPatchEmbeddings()
        {
            var config = CreateSmallConfig();
            using (var model = new BinaryAnomalyModelFactory().Create(3, config, CPU))
            using (var input = rand(2, 3, 64, 64))
            using (var inference = inference_mode())
            using (var logits = model.call(input))
            using (var embeddings = model.ExtractPatchEmbeddings(input))
            {
                Assert.Equal(new long[] { 2, SimilarityAnomalyModel.PretextClassCount }, logits.shape);
                Assert.Equal(new long[] { 2, 32, 8, 8 }, embeddings.shape);
            }
        }

        [Fact]
        public void Calibration_SeparatesValidationClasses()
        {
            var config = CreateSmallConfig();
            var calibration = SimilarityAnomalyCalibration.Fit(
                new List<SimilarityValidationScore>
                {
                    new SimilarityValidationScore { IsNotOk = false, ImageScore = 0.1f, PatchScores = new[] { 0.01f, 0.02f } },
                    new SimilarityValidationScore { IsNotOk = false, ImageScore = 0.2f, PatchScores = new[] { 0.02f, 0.03f } },
                    new SimilarityValidationScore { IsNotOk = true, ImageScore = 0.8f },
                    new SimilarityValidationScore { IsNotOk = true, ImageScore = 0.9f }
                },
                config);

            Assert.True(0.2f < calibration.RawDecisionThreshold);
            Assert.True(0.8f >= calibration.RawDecisionThreshold);
            Assert.InRange(calibration.Heatmap.Normalize(0.01f), 0, 1);
        }

        [Fact]
        public void OkOnlyCalibration_PlacesThresholdAboveObservedOkScores()
        {
            var config = CreateSmallConfig();
            var calibration = SimilarityAnomalyCalibration.Fit(
                new List<SimilarityValidationScore>
                {
                    new SimilarityValidationScore
                    {
                        IsNotOk = false,
                        ImageScore = 0.2f,
                        PatchScores = new[] { 0.01f, 0.02f, 0.03f }
                    }
                },
                config);

            Assert.True(calibration.RawDecisionThreshold > 0.2f);
        }

        [Fact]
        public void GoldenSampleApproach_IsPersistableButNotImplementedYet()
        {
            var config = CreateSmallConfig();
            config.Approach = AnomalyDetectionApproach.GoldenSample;
            config.Validate();

            Assert.Throws<NotSupportedException>(() =>
                new BinaryAnomalyModelFactory().Create(3, config, CPU));
        }

        [Fact]
        public void SimilarityScorer_ScoresMemoryPatchesNearZero()
        {
            var config = CreateSmallConfig();
            using (var model = new BinaryAnomalyModelFactory().Create(3, config, CPU))
            using (var input = rand(1, 3, 64, 64))
            using (var inference = inference_mode())
            using (var embeddingMap = model.ExtractPatchEmbeddings(input))
            using (var descriptors = embeddingMap.permute(0, 2, 3, 1).contiguous().reshape(-1, embeddingMap.shape[1]))
            using (var bank = new SimilarityMemoryBank(descriptors))
            using (var scorer = new SimilarityNearestNeighbourScorer(bank, config, CPU))
            using (var result = scorer.Score(model, input))
            {
                Assert.InRange(result.ImageScore, 0, 1e-3f);
                Assert.Equal(new long[] { 1, 1, 8, 8 }, result.TileScoreMaps.shape);
            }
        }

        [Fact]
        public void PretextClassifier_BackpropagatesFromSyntheticClasses()
        {
            var config = CreateSmallConfig();
            using (var model = new BinaryAnomalyModelFactory().Create(3, config, CPU))
            using (var input = rand(2, 3, 64, 64))
            using (var synthetic = CutPasteBatchGenerator.Create(input, 0.02, 0.15, 42))
            using (var logits = model.call(synthetic.Inputs))
            using (var loss = TorchSharp.torch.nn.functional.cross_entropy(logits, synthetic.Targets))
            using (var optimizer = torch.optim.AdamW(model.parameters(), 1e-3))
            {
                optimizer.zero_grad();
                loss.backward();
                optimizer.step();
            }
        }

        [Fact]
        public void MemoryBank_RoundTrips()
        {
            var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".bin");
            try
            {
                using (var embeddings = rand(5, 7))
                using (var bank = new SimilarityMemoryBank(embeddings))
                {
                    bank.Save(path);
                }

                using (var loaded = SimilarityMemoryBank.Load(path))
                {
                    Assert.Equal(5, loaded.Count);
                    Assert.Equal(7, loaded.Dimensions);
                }
            }
            finally
            {
                if (File.Exists(path)) File.Delete(path);
            }
        }

        [Fact]
        public void SimilarityPackage_RoundTripsAndRuns()
        {
            var directory = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(directory);
            var config = CreateSmallConfig();

            try
            {
                using (var model = new BinaryAnomalyModelFactory().Create(3, config, CPU))
                using (var memory = rand(10, 32))
                using (var bank = new SimilarityMemoryBank(memory))
                {
                    var package = new BinaryAnomalyModelPackage
                    {
                        ModelConfig = config,
                        Settings = new DeepLearningSettings(),
                        Calibration = new SimilarityAnomalyCalibration
                        {
                            ImageScore = new LinearScoreCalibration { Low = 0, High = 1 },
                            Heatmap = new LinearScoreCalibration { Low = 0, High = 2 },
                            RawDecisionThreshold = 0.5f
                        }
                    };
                    model.save(Path.Combine(directory, package.FeatureExtractorWeightsFile));
                    bank.Save(Path.Combine(directory, package.MemoryBankFile));
                    File.WriteAllText(
                        Path.Combine(directory, "anomaly-model.json"),
                        JsonSerializer.Serialize(package));
                }

                using (var session = BinaryAnomalyInferenceSession.Load(directory, CPU))
                using (var input = rand(1, 3, 64, 64))
                using (var prediction = session.Run(input))
                {
                    Assert.InRange(prediction.Score, 0, 1);
                    Assert.Equal(new long[] { 1, 1, 8, 8 }, prediction.TileHeatmaps.shape);
                }
            }
            finally
            {
                if (Directory.Exists(directory)) Directory.Delete(directory, true);
            }
        }

        [Fact]
        public void RandomInitialization_CannotFreezeUntrainedBackbone()
        {
            var config = CreateSmallConfig();
            config.FrozenBackboneEpochs = 1;
            Assert.Throws<InvalidOperationException>(() => config.Validate());
        }

        [Theory]
        [InlineData(ModelComplexity.L0, 32, 0.02, 2_000, 256)]
        [InlineData(ModelComplexity.L1, 64, 0.05, 5_000, 512)]
        [InlineData(ModelComplexity.L2, 96, 0.075, 10_000, 512)]
        [InlineData(ModelComplexity.L3, 128, 0.10, 20_000, 1_024)]
        [InlineData(ModelComplexity.L4, 192, 0.15, 35_000, 1_024)]
        [InlineData(ModelComplexity.L5, 256, 0.20, 50_000, 2_048)]
        public void ComplexityProvider_ReturnsExpectedValidPreset(
            ModelComplexity complexity,
            int embeddingChannels,
            double samplingRatio,
            int maxMemoryEntries,
            int distanceChunkSize)
        {
            var services = new ServiceCollection();
            services.AddAiServices();

            using (var serviceProvider = services.BuildServiceProvider())
            {
                var provider = serviceProvider.GetRequiredService<IBinaryAnomalyComplexityConfigProvider>();
                var config = provider.GetConfig(complexity);

                Assert.Equal(embeddingChannels, config.EmbeddingChannels);
                Assert.Equal(samplingRatio, config.MemoryBankSamplingRatio, 6);
                Assert.Equal(maxMemoryEntries, config.MaxMemoryEntries);
                Assert.Equal(distanceChunkSize, config.DistanceChunkSize);
                Assert.Equal(AnomalyDetectionApproach.SimilarityBased, config.Approach);
                Assert.Equal(WeightInitialization.Random, config.Initialization);
                Assert.Null(config.PretrainedWeightsPath);
                config.Validate();
            }
        }

        [Fact]
        public void ComplexityProvider_ReturnsIndependentConfigs()
        {
            var services = new ServiceCollection();
            services.AddAiServices();

            using (var serviceProvider = services.BuildServiceProvider())
            {
                var provider = serviceProvider.GetRequiredService<IBinaryAnomalyComplexityConfigProvider>();
                var first = provider.GetConfig(ModelComplexity.L3);
                first.EmbeddingChannels = 1;

                var second = provider.GetConfig(ModelComplexity.L3);

                Assert.Equal(128, second.EmbeddingChannels);
            }
        }

        [Fact]
        public void ComplexityProvider_RejectsUnknownLevel()
        {
            var services = new ServiceCollection();
            services.AddAiServices();

            using (var serviceProvider = services.BuildServiceProvider())
            {
                var provider = serviceProvider.GetRequiredService<IBinaryAnomalyComplexityConfigProvider>();
                Assert.Throws<ArgumentOutOfRangeException>(() => provider.GetConfig((ModelComplexity)999));
            }
        }

        [Fact]
        public void AnomalyTileGenerator_WritesImageTilesWithoutMasks()
        {
            var directory = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
            var optionsService = new ProjectOptionsService(Options.Create(new ProjectOptions()));
            var presenter = new ProjectPresenter(new JsonSerializerOptions(), optionsService)
            {
                ProjectPath = directory
            };
            presenter.Project.Name = "anomaly-tiles";
            presenter.Project.UseCaseMode = AppUseCaseMode.AnomalyDetection;
            presenter.Project.Settings.PreprocessingSettings.SliceSize = 32;
            presenter.Project.Settings.PreprocessingSettings.DownSample = 1;
            presenter.Project.Settings.PreprocessingSettings.BorderPadding = false;
            presenter.Project.Settings.PreprocessingSettings.TrainAsGreyscale = true;
            presenter.Project.Settings.PreprocessingSettings.TrainOnlyFeatures = true;
            optionsService.EnsureAll(directory);

            var imageItem = new ImageItem
            {
                ImageSize = new System.Drawing.Size(64, 64),
                Roi = new System.Drawing.Rectangle(0, 0, 64, 64),
                Split = DatasetSplit.Train,
                AnomalyLabel = AnomalyLabel.Ok
            };
            presenter.Project.Images.Add(imageItem);

            try
            {
                using (var image = new Mat(
                    64,
                    64,
                    MatType.CV_8UC1,
                    OpenCvSharp.Scalar.All(127)))
                {
                    Cv2.ImWrite(
                        Path.Combine(presenter.Paths.Images, imageItem.Guid + presenter.Paths.ImagesExt),
                        image);
                }

                TrainingTileGenerator.GenerateAnomalyTrainingTiles(
                    presenter,
                    progress: null,
                    CancellationToken.None);

                var samples = presenter.GetAnomalyTrainingSamples(DatasetSplit.Train);
                var sample = Assert.Single(samples);
                Assert.NotEmpty(sample.TilePaths);
                Assert.Empty(Directory.EnumerateFiles(presenter.Paths.SlicedMasks));
            }
            finally
            {
                if (Directory.Exists(directory)) Directory.Delete(directory, true);
            }
        }

        [Fact]
        public async Task CanceledAnomalyTraining_SavesLoadableSimilarityPackage()
        {
            var directory = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
            var artifactDirectory = Path.Combine(directory, "canceled-model");
            var services = new ServiceCollection();
            services.AddCoreServices();
            services.AddAiServices();

            try
            {
                using (var serviceProvider = services.BuildServiceProvider())
                {
                    var presenter = serviceProvider.GetRequiredService<IProjectPresenter>();
                    presenter.ProjectPath = directory;
                    presenter.Project.Name = "canceled-anomaly-training";
                    presenter.Project.UseCaseMode = AppUseCaseMode.AnomalyDetection;
                    presenter.Project.Settings.PreprocessingSettings.TrainAsGreyscale = true;
                    serviceProvider
                        .GetRequiredService<IProjectOptionsService>()
                        .EnsureAll(directory);

                    AddAnomalyTile(presenter, DatasetSplit.Train, AnomalyLabel.Ok);
                    AddAnomalyTile(presenter, DatasetSplit.Train, AnomalyLabel.NotOk);
                    AddAnomalyTile(presenter, DatasetSplit.Validate, AnomalyLabel.Ok);
                    AddAnomalyTile(presenter, DatasetSplit.Validate, AnomalyLabel.NotOk);

                    var pipeline = serviceProvider.GetRequiredService<BinaryAnomalyTrainingPipeline>();
                    var config = CreateSmallConfig();
                    using (var cancellation = new CancellationTokenSource())
                    {
                        cancellation.Cancel();
                        await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
                            pipeline.RunTraining(
                                presenter,
                                config,
                                artifactDirectory,
                                progress: null,
                                cancellation.Token));
                    }

                    Assert.True(File.Exists(Path.Combine(artifactDirectory, "similarity-encoder.bin")));
                    Assert.True(File.Exists(Path.Combine(artifactDirectory, "similarity-memory.bin")));
                    Assert.True(File.Exists(Path.Combine(artifactDirectory, "anomaly-model.json")));

                    using (var session = BinaryAnomalyInferenceSession.Load(
                        artifactDirectory,
                        CPU,
                        serviceProvider.GetRequiredService<JsonSerializerOptions>()))
                    {
                        Assert.NotNull(session.Package.Calibration);
                    }
                }
            }
            finally
            {
                if (Directory.Exists(directory)) Directory.Delete(directory, true);
            }
        }

        private static void AddAnomalyTile(
            IProjectPresenter presenter,
            DatasetSplit split,
            AnomalyLabel label)
        {
            var imageItem = new ImageItem
            {
                ImageSize = new System.Drawing.Size(64, 64),
                Roi = new System.Drawing.Rectangle(0, 0, 64, 64),
                Split = split,
                AnomalyLabel = label
            };
            presenter.Project.Images.Add(imageItem);

            using (var image = new Mat(64, 64, MatType.CV_8UC1))
            {
                Cv2.Randu(
                    image,
                    OpenCvSharp.Scalar.All(0),
                    OpenCvSharp.Scalar.All(255));
                Cv2.ImWrite(
                    Path.Combine(
                        presenter.Paths.SlicedImages,
                        imageItem.Guid + "_0000" + presenter.Paths.ImagesExt),
                    image);
            }
        }

        private static BinaryAnomalyModelConfig CreateSmallConfig()
        {
            return new BinaryAnomalyModelConfig
            {
                Initialization = WeightInitialization.Random,
                Approach = AnomalyDetectionApproach.SimilarityBased,
                EmbeddingChannels = 16,
                Dropout = 0,
                FrozenBackboneEpochs = 0,
                MaxMemoryEntries = 100,
                DistanceChunkSize = 16
            };
        }
    }
}
