using System;
using static TorchSharp.torch;

namespace VisionStudioAI.ML.Inference.AnomalyDetection
{
    public sealed class AnomalyPrediction : IDisposable
    {
        public float RawScore { get; set; }
        public float Score { get; set; }
        public bool IsNotOk { get; set; }

        /// <summary>Calibrated similarity heatmaps [tiles,1,h,w].</summary>
        public Tensor TileHeatmaps { get; set; }

        public void Dispose()
        {
            TileHeatmaps?.Dispose();
            TileHeatmaps = null;
        }
    }
}
