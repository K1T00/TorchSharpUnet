using System;
using static TorchSharp.torch;

namespace VisionStudioAI.ML.Models.AnomalyDetection
{
    public static class AnomalyTopKPooling
    {
        /// <summary>
        /// Pools each item in [B,1,H,W] independently and returns [B].
        /// </summary>
        public static Tensor PoolBatch(Tensor spatialScores, double fraction)
        {
            Validate(spatialScores, fraction);
            var flat = spatialScores.flatten(start_dim: 1);
            var k = GetK(flat.shape[1], fraction);
            return flat.topk(k, dim: 1).Item1.mean(new long[] { 1 });
        }

        /// <summary>
        /// Treats every tile as part of one image bag and returns shape [1].
        /// </summary>
        public static Tensor PoolImageBag(Tensor spatialScores, double fraction)
        {
            Validate(spatialScores, fraction);
            var flat = spatialScores.flatten().reshape(1, -1);
            var k = GetK(flat.shape[1], fraction);
            return flat.topk(k, dim: 1).Item1.mean(new long[] { 1 });
        }

        private static int GetK(long elementCount, double fraction)
        {
            return checked((int)Math.Max(1, Math.Min(elementCount, Math.Ceiling(elementCount * fraction))));
        }

        private static void Validate(Tensor scores, double fraction)
        {
            if (ReferenceEquals(scores, null)) throw new ArgumentNullException(nameof(scores));
            if (scores.Dimensions != 4 || scores.shape[1] != 1)
                throw new ArgumentException("Expected spatial scores with shape [B,1,H,W].", nameof(scores));
            if (fraction <= 0 || fraction > 1)
                throw new ArgumentOutOfRangeException(nameof(fraction));
        }
    }
}
