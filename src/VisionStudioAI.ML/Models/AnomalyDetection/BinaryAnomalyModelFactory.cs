using static TorchSharp.torch;

namespace VisionStudioAI.ML.Models.AnomalyDetection
{
    public sealed class BinaryAnomalyModelFactory : IBinaryAnomalyModelFactory
    {
        public string Name => "SimilarityBasedAnomaly";

        public SimilarityAnomalyModel Create(int inChannels, BinaryAnomalyModelConfig cfg, Device device)
        {
            return new SimilarityAnomalyModel(inChannels, cfg, device);
        }
    }
}
