using System;
using System.Collections.Generic;
using TorchSharp;
using VisionStudioAI.ML.Models.AnomalyDetection;
using VisionStudioAI.ML.Training.AnomalyDetection;
using static TorchSharp.torch;

namespace VisionStudioAI.ML.Inference.AnomalyDetection
{
    /// <summary>Scores every local descriptor by its nearest normal neighbour.</summary>
    public sealed class SimilarityNearestNeighbourScorer : IDisposable
    {
        private readonly Tensor memoryBank;
        private readonly int chunkSize;
        private readonly double topKFraction;

        public SimilarityNearestNeighbourScorer(
            SimilarityMemoryBank memoryBank,
            BinaryAnomalyModelConfig config,
            Device device)
        {
            if (memoryBank == null) throw new ArgumentNullException(nameof(memoryBank));
            config.Validate();
            this.memoryBank = memoryBank.Embeddings.clone().to(device).contiguous();
            chunkSize = config.DistanceChunkSize;
            topKFraction = config.ImageScoreTopKFraction;
        }

        public SimilarityAnomalyScore Score(
            SimilarityAnomalyModel model,
            Tensor imageTiles)
        {
            if (model == null) throw new ArgumentNullException(nameof(model));

            using (var scope = NewDisposeScope())
            using (var inference = inference_mode())
            {
                var map = model.ExtractPatchEmbeddings(imageTiles);
                if (map.shape[1] != memoryBank.shape[1])
                    throw new InvalidOperationException("Descriptor dimension does not match the memory bank.");

                var queries = map
                    .permute(0, 2, 3, 1)
                    .contiguous()
                    .reshape(-1, map.shape[1])
                    .to_type(ScalarType.Float32);
                var chunks = new List<Tensor>();
                for (long start = 0; start < queries.shape[0]; start += chunkSize)
                {
                    var count = Math.Min(chunkSize, queries.shape[0] - start);
                    var queryChunk = queries.narrow(0, start, count);
                    chunks.Add(cdist(queryChunk, memoryBank).min(1).Item1);
                }

                var scores = cat(chunks.ToArray(), 0)
                    .reshape(map.shape[0], 1, map.shape[2], map.shape[3]);
                var imageScore = AnomalyTopKPooling
                    .PoolImageBag(scores, topKFraction)
                    .ToSingle();
                return new SimilarityAnomalyScore
                {
                    ImageScore = imageScore,
                    TileScoreMaps = scores.MoveToOuterDisposeScope()
                };
            }
        }

        public void Dispose()
        {
            memoryBank.Dispose();
        }
    }
}
