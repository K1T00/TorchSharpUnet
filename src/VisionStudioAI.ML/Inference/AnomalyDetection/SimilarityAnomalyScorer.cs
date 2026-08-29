using System;
using VisionStudioAI.ML.Models.AnomalyDetection;
using static TorchSharp.torch;

namespace VisionStudioAI.ML.Inference.AnomalyDetection
{
    public sealed class SimilarityAnomalyScorer : IDisposable
    {
        private readonly SimilarityAnomalyModel model;
        private readonly SimilarityNearestNeighbourScorer nearestNeighbour;
        private readonly SimilarityAnomalyCalibration calibration;

        public SimilarityAnomalyScorer(
            SimilarityAnomalyModel model,
            SimilarityNearestNeighbourScorer nearestNeighbour,
            SimilarityAnomalyCalibration calibration)
        {
            this.model = model ?? throw new ArgumentNullException(nameof(model));
            this.nearestNeighbour = nearestNeighbour ?? throw new ArgumentNullException(nameof(nearestNeighbour));
            this.calibration = calibration ?? throw new ArgumentNullException(nameof(calibration));
        }

        public AnomalyPrediction Score(Tensor imageTiles)
        {
            model.eval();
            using (var scope = NewDisposeScope())
            using (var inference = inference_mode())
            using (var raw = nearestNeighbour.Score(model, imageTiles))
            {
                return new AnomalyPrediction
                {
                    RawScore = raw.ImageScore,
                    Score = calibration.ImageScore.Normalize(raw.ImageScore),
                    IsNotOk = raw.ImageScore >= calibration.RawDecisionThreshold,
                    TileHeatmaps = calibration.Heatmap
                        .Normalize(raw.TileScoreMaps)
                        .MoveToOuterDisposeScope()
                };
            }
        }

        public void Dispose()
        {
            nearestNeighbour.Dispose();
            model.Dispose();
        }
    }
}
