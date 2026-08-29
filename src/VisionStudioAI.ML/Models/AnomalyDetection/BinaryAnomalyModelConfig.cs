using System;
using static TorchSharp.torch;

namespace VisionStudioAI.ML.Models.AnomalyDetection
{
    public enum WeightInitialization
    {
        Random,
        Pretrained
    }

    public enum AnomalyBackbone
    {
        ResNet18
    }

    /// <summary>
    /// Selects the anomaly-detection strategy. GoldenSample is reserved so a
    /// position-dependent implementation can be added without changing the
    /// persisted model contract.
    /// </summary>
    public enum AnomalyDetectionApproach
    {
        SimilarityBased = 0,
        GoldenSample = 1
    }

    /// <summary>
    /// Configuration for OK-only, similarity-based anomaly detection.
    /// The feature extractor is trained with generated CutPaste pretext tasks;
    /// only unmodified OK descriptors are stored in the final memory bank.
    /// </summary>
    public sealed class BinaryAnomalyModelConfig
    {
        public AnomalyDetectionApproach Approach { get; set; } =
            AnomalyDetectionApproach.SimilarityBased;

        public AnomalyBackbone Backbone { get; set; } = AnomalyBackbone.ResNet18;
        public WeightInitialization Initialization { get; set; } = WeightInitialization.Random;
        public string PretrainedWeightsPath { get; set; }
        public int EmbeddingChannels { get; set; } = 128;
        public double Dropout { get; set; } = 0.2;
        public ScalarType TrainPrecision { get; set; } = ScalarType.Float32;
        public int FrozenBackboneEpochs { get; set; } = 0;
        public double LearningRate { get; set; } = 1e-3;
        public double WeightDecay { get; set; } = 1e-4;
        public bool FreezePretrainedBatchNormalization { get; set; } = true;
        public bool UseImageNetNormalizationForPretrained { get; set; } = true;

        /// <summary>Fraction of normal descriptors retained before the hard cap.</summary>
        public double MemoryBankSamplingRatio { get; set; } = 0.1;
        public int MaxMemoryEntries { get; set; } = 50_000;
        public int DistanceChunkSize { get; set; } = 1_024;
        public int PatchAggregationKernelSize { get; set; } = 3;

        /// <summary>Fraction of the highest local distances used for the image score.</summary>
        public double ImageScoreTopKFraction { get; set; } = 0.01;

        /// <summary>Minimum and maximum synthetic CutPaste area relative to a tile.</summary>
        public double CutPasteMinAreaFraction { get; set; } = 0.02;
        public double CutPasteMaxAreaFraction { get; set; } = 0.15;

        /// <summary>
        /// Margin added above the observed OK score when no NOK validation
        /// images are available for supervised threshold selection.
        /// </summary>
        public double OkOnlyThresholdMargin { get; set; } = 0.10;

        public void Validate()
        {
            if (!Enum.IsDefined(typeof(AnomalyDetectionApproach), Approach))
                throw new ArgumentOutOfRangeException(nameof(Approach));
            if (!Enum.IsDefined(typeof(AnomalyBackbone), Backbone))
                throw new ArgumentOutOfRangeException(nameof(Backbone));
            if (!Enum.IsDefined(typeof(WeightInitialization), Initialization))
                throw new ArgumentOutOfRangeException(nameof(Initialization));
            if (Initialization == WeightInitialization.Pretrained &&
                string.IsNullOrWhiteSpace(PretrainedWeightsPath))
            {
                throw new InvalidOperationException(
                    "PretrainedWeightsPath is required for pretrained initialization.");
            }
            if (Initialization == WeightInitialization.Random && FrozenBackboneEpochs != 0)
                throw new InvalidOperationException("A randomly initialized backbone cannot be frozen.");
            if (EmbeddingChannels <= 0) throw new ArgumentOutOfRangeException(nameof(EmbeddingChannels));
            if (Dropout < 0 || Dropout >= 1) throw new ArgumentOutOfRangeException(nameof(Dropout));
            if (FrozenBackboneEpochs < 0) throw new ArgumentOutOfRangeException(nameof(FrozenBackboneEpochs));
            if (LearningRate <= 0) throw new ArgumentOutOfRangeException(nameof(LearningRate));
            if (WeightDecay < 0) throw new ArgumentOutOfRangeException(nameof(WeightDecay));
            if (MemoryBankSamplingRatio <= 0 || MemoryBankSamplingRatio > 1)
                throw new ArgumentOutOfRangeException(nameof(MemoryBankSamplingRatio));
            if (MaxMemoryEntries <= 0) throw new ArgumentOutOfRangeException(nameof(MaxMemoryEntries));
            if (DistanceChunkSize <= 0) throw new ArgumentOutOfRangeException(nameof(DistanceChunkSize));
            if (PatchAggregationKernelSize <= 0 || PatchAggregationKernelSize % 2 == 0)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(PatchAggregationKernelSize),
                    "Patch aggregation kernel size must be a positive odd number.");
            }
            if (ImageScoreTopKFraction <= 0 || ImageScoreTopKFraction > 1)
                throw new ArgumentOutOfRangeException(nameof(ImageScoreTopKFraction));
            if (CutPasteMinAreaFraction <= 0 || CutPasteMinAreaFraction >= 1)
                throw new ArgumentOutOfRangeException(nameof(CutPasteMinAreaFraction));
            if (CutPasteMaxAreaFraction <= CutPasteMinAreaFraction || CutPasteMaxAreaFraction >= 1)
                throw new ArgumentOutOfRangeException(nameof(CutPasteMaxAreaFraction));
            if (OkOnlyThresholdMargin < 0)
                throw new ArgumentOutOfRangeException(nameof(OkOnlyThresholdMargin));
        }
    }
}
