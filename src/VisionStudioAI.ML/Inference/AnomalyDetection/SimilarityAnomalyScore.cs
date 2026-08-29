using System;
using static TorchSharp.torch;

namespace VisionStudioAI.ML.Inference.AnomalyDetection
{
    public sealed class SimilarityAnomalyScore : IDisposable
    {
        public float ImageScore { get; set; }
        public Tensor TileScoreMaps { get; set; }

        public void Dispose()
        {
            TileScoreMaps?.Dispose();
            TileScoreMaps = null;
        }
    }
}
