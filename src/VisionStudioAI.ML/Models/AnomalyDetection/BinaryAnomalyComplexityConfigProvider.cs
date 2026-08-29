using System;
using VisionStudioAI.Core.Models;
using static TorchSharp.torch;

namespace VisionStudioAI.ML.Models.AnomalyDetection
{
    /// <summary>
    /// Maps the common user-facing complexity levels to similarity-feature
    /// capacity and normal memory-bank size.
    ///
    /// All current levels use ResNet18. Higher levels increase the learned
    /// descriptor width and retain more normal patch descriptors.
    ///
    /// Initialization, pretrained weight paths, and frozen-backbone epochs are
    /// deployment/training choices and must be applied by the caller after the
    /// preset has been created.
    /// </summary>
    internal sealed class BinaryAnomalyComplexityConfigProvider : IBinaryAnomalyComplexityConfigProvider
    {
        public BinaryAnomalyModelConfig GetConfig(ModelComplexity complexity)
        {
            var config = new BinaryAnomalyModelConfig
            {
                Approach = AnomalyDetectionApproach.SimilarityBased,
                Backbone = AnomalyBackbone.ResNet18,
                Initialization = WeightInitialization.Random,
                PretrainedWeightsPath = null,
                ImageScoreTopKFraction = 0.01,
                Dropout = 0.2,
                TrainPrecision = ScalarType.Float32,
                FrozenBackboneEpochs = 0,
                LearningRate = 3e-3,
                WeightDecay = 1e-4,
                FreezePretrainedBatchNormalization = true,
                UseImageNetNormalizationForPretrained = true,
                PatchAggregationKernelSize = 3,
                CutPasteMinAreaFraction = 0.02,
                CutPasteMaxAreaFraction = 0.15,
                OkOnlyThresholdMargin = 0.10
            };

            switch (complexity)
            {
                case ModelComplexity.L0:
                    config.EmbeddingChannels = 32;
                    config.MemoryBankSamplingRatio = 0.02;
                    config.MaxMemoryEntries = 2_000;
                    config.DistanceChunkSize = 256;
                    break;

                case ModelComplexity.L1:
                    config.EmbeddingChannels = 64;
                    config.MemoryBankSamplingRatio = 0.05;
                    config.MaxMemoryEntries = 5_000;
                    config.DistanceChunkSize = 512;
                    break;

                case ModelComplexity.L2:
                    config.EmbeddingChannels = 96;
                    config.MemoryBankSamplingRatio = 0.075;
                    config.MaxMemoryEntries = 10_000;
                    config.DistanceChunkSize = 512;
                    break;

                case ModelComplexity.L3:
                    config.EmbeddingChannels = 128;
                    config.MemoryBankSamplingRatio = 0.10;
                    config.MaxMemoryEntries = 20_000;
                    config.DistanceChunkSize = 1_024;
                    break;

                case ModelComplexity.L4:
                    config.EmbeddingChannels = 192;
                    config.MemoryBankSamplingRatio = 0.15;
                    config.MaxMemoryEntries = 35_000;
                    config.DistanceChunkSize = 1_024;
                    break;

                case ModelComplexity.L5:
                    config.EmbeddingChannels = 256;
                    config.MemoryBankSamplingRatio = 0.20;
                    config.MaxMemoryEntries = 50_000;
                    config.DistanceChunkSize = 2_048;
                    break;

                default:
                    throw new ArgumentOutOfRangeException(
                        nameof(complexity),
                        complexity,
                        "Model complexity must be between L0 and L5.");
            }

            config.Validate();
            return config;
        }
    }
}
