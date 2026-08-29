using System;
using System.IO;
using System.Text.Json;
using TorchSharp;
using VisionStudioAI.ML.IO;
using VisionStudioAI.ML.Models.AnomalyDetection;
using VisionStudioAI.ML.Training.AnomalyDetection;
using static TorchSharp.torch;

namespace VisionStudioAI.ML.Inference.AnomalyDetection
{
    /// <summary>
    /// Owns the feature extractor, normal memory bank, and calibrated scorer.
    /// Input tensors must use the normalization stored in Package.Settings.
    /// </summary>
    public sealed class BinaryAnomalyInferenceSession : IDisposable
    {
        private readonly SimilarityAnomalyScorer scorer;
        public BinaryAnomalyModelPackage Package { get; }

        private BinaryAnomalyInferenceSession(SimilarityAnomalyScorer scorer, BinaryAnomalyModelPackage package)
        {
            this.scorer = scorer;
            Package = package;
        }

        public static BinaryAnomalyInferenceSession Load(
            string artifactDirectory,
            Device device,
            JsonSerializerOptions jsonOptions = null)
        {
            if (string.IsNullOrWhiteSpace(artifactDirectory))
                throw new ArgumentException("Artifact directory is required.", nameof(artifactDirectory));

            var metadataPath = Path.Combine(artifactDirectory, "anomaly-model.json");
            if (!File.Exists(metadataPath))
                throw new FileNotFoundException("Anomaly model metadata not found.", metadataPath);

            var package = JsonSerializer.Deserialize<BinaryAnomalyModelPackage>(
                File.ReadAllText(metadataPath), jsonOptions ?? new JsonSerializerOptions());
            if (package == null)
                throw new InvalidOperationException("Invalid anomaly model package.");
            if (package.FormatVersion != BinaryAnomalyModelPackage.CurrentFormatVersion)
            {
                throw new InvalidOperationException(
                    "This anomaly artifact uses the removed hybrid model format. " +
                    "Retrain it with the similarity-based pipeline.");
            }
            if (package.ModelConfig == null ||
                package.ModelConfig.Approach != AnomalyDetectionApproach.SimilarityBased)
            {
                throw new NotSupportedException(
                    "Only similarity-based anomaly artifacts are currently supported.");
            }

            var runtimeConfig = CopyForFullModelLoading(package.ModelConfig);
            var channels = package.Settings.PreprocessingSettings.TrainAsGreyscale ? 1 : 3;
            var model = new BinaryAnomalyModelFactory().Create(channels, runtimeConfig, device);

            try
            {
                model.load(Path.Combine(artifactDirectory, package.FeatureExtractorWeightsFile));
                model.to(runtimeConfig.TrainPrecision, true).eval();

                using (var bank = SimilarityMemoryBank.Load(Path.Combine(artifactDirectory, package.MemoryBankFile)))
                {
                    var nearest = new SimilarityNearestNeighbourScorer(bank, runtimeConfig, device);
                    return new BinaryAnomalyInferenceSession(
                        new SimilarityAnomalyScorer(model, nearest, package.Calibration),
                        package);
                }
            }
            catch
            {
                model.Dispose();
                throw;
            }
        }

        public AnomalyPrediction Run(Tensor normalizedImageTiles)
        {
            return scorer.Score(normalizedImageTiles);
        }

        public void Dispose()
        {
            scorer.Dispose();
        }

        private static BinaryAnomalyModelConfig CopyForFullModelLoading(BinaryAnomalyModelConfig source)
        {
            if (source == null) throw new InvalidOperationException("Anomaly model configuration is missing.");
            return new BinaryAnomalyModelConfig
            {
                Approach = source.Approach,
                Backbone = source.Backbone,
                Initialization = WeightInitialization.Random,
                EmbeddingChannels = source.EmbeddingChannels,
                ImageScoreTopKFraction = source.ImageScoreTopKFraction,
                Dropout = source.Dropout,
                TrainPrecision = source.TrainPrecision,
                FrozenBackboneEpochs = 0,
                LearningRate = source.LearningRate,
                WeightDecay = source.WeightDecay,
                FreezePretrainedBatchNormalization = source.FreezePretrainedBatchNormalization,
                UseImageNetNormalizationForPretrained = source.UseImageNetNormalizationForPretrained,
                MemoryBankSamplingRatio = source.MemoryBankSamplingRatio,
                MaxMemoryEntries = source.MaxMemoryEntries,
                DistanceChunkSize = source.DistanceChunkSize,
                PatchAggregationKernelSize = source.PatchAggregationKernelSize,
                CutPasteMinAreaFraction = source.CutPasteMinAreaFraction,
                CutPasteMaxAreaFraction = source.CutPasteMaxAreaFraction,
                OkOnlyThresholdMargin = source.OkOnlyThresholdMargin
            };
        }
    }
}
