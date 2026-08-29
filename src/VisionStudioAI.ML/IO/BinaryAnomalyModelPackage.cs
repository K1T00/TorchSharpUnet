using VisionStudioAI.Core.Models;
using VisionStudioAI.ML.Inference.AnomalyDetection;
using VisionStudioAI.ML.Models.AnomalyDetection;

namespace VisionStudioAI.ML.IO
{
    public sealed class BinaryAnomalyModelPackage
    {
        public const int CurrentFormatVersion = 2;

        public int FormatVersion { get; set; } = CurrentFormatVersion;
        public BinaryAnomalyModelConfig ModelConfig { get; set; }
        public DeepLearningSettings Settings { get; set; }
        public SimilarityAnomalyCalibration Calibration { get; set; }
        public string FeatureExtractorWeightsFile { get; set; } = "similarity-encoder.bin";
        public string MemoryBankFile { get; set; } = "similarity-memory.bin";
    }
}
