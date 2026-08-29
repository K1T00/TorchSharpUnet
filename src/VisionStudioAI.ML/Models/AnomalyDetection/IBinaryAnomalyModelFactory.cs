using static TorchSharp.torch;

namespace VisionStudioAI.ML.Models.AnomalyDetection
{
    public interface IBinaryAnomalyModelFactory
    {
        string Name { get; }

        /// <summary>Creates the model selected by the anomaly approach configuration.</summary>
        SimilarityAnomalyModel Create(int inChannels, BinaryAnomalyModelConfig cfg, Device device);
    }
}
