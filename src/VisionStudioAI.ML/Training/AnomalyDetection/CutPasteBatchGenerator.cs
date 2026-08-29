using System;
using System.Linq;
using static TorchSharp.torch;

namespace VisionStudioAI.ML.Training.AnomalyDetection
{
    /// <summary>
    /// Creates a three-class self-supervised batch from OK tiles:
    /// original, rectangular CutPaste, and thin scar CutPaste.
    /// </summary>
    internal static class CutPasteBatchGenerator
    {
        public static SyntheticPretextBatch Create(
            Tensor okTiles,
            double minAreaFraction,
            double maxAreaFraction,
            int seed)
        {
            if (ReferenceEquals(okTiles, null)) throw new ArgumentNullException(nameof(okTiles));
            if (okTiles.Dimensions != 4) throw new ArgumentException("Expected [N,C,H,W].", nameof(okTiles));
            if (okTiles.shape[0] <= 0) throw new ArgumentException("The OK tile batch is empty.", nameof(okTiles));

            using (var scope = NewDisposeScope())
            {
                var regular = okTiles.clone();
                var scars = okTiles.clone();
                var random = new Random(seed);

                for (long index = 0; index < okTiles.shape[0]; index++)
                {
                    ApplyRegularCutPaste(regular, index, minAreaFraction, maxAreaFraction, random);
                    ApplyScarCutPaste(scars, index, random);
                }

                var inputs = cat(new[] { okTiles, regular, scars }, 0);
                var count = checked((int)okTiles.shape[0]);
                var targets = Enumerable.Repeat(0L, count)
                    .Concat(Enumerable.Repeat(1L, count))
                    .Concat(Enumerable.Repeat(2L, count))
                    .ToArray();
                var targetTensor = tensor(
                    targets,
                    dtype: ScalarType.Int64,
                    device: okTiles.device);

                return new SyntheticPretextBatch(
                    inputs.MoveToOuterDisposeScope(),
                    targetTensor.MoveToOuterDisposeScope());
            }
        }

        private static void ApplyRegularCutPaste(
            Tensor batch,
            long batchIndex,
            double minAreaFraction,
            double maxAreaFraction,
            Random random)
        {
            var height = checked((int)batch.shape[2]);
            var width = checked((int)batch.shape[3]);
            var areaFraction = minAreaFraction +
                               random.NextDouble() * (maxAreaFraction - minAreaFraction);
            var targetArea = Math.Max(4.0, height * width * areaFraction);
            var aspect = Math.Exp(Math.Log(0.5) + random.NextDouble() * Math.Log(4.0));
            var patchWidth = Clamp((int)Math.Round(Math.Sqrt(targetArea * aspect)), 2, Math.Max(2, width - 1));
            var patchHeight = Clamp((int)Math.Round(Math.Sqrt(targetArea / aspect)), 2, Math.Max(2, height - 1));
            PastePatch(batch, batchIndex, patchHeight, patchWidth, random);
        }

        private static void ApplyScarCutPaste(
            Tensor batch,
            long batchIndex,
            Random random)
        {
            var height = checked((int)batch.shape[2]);
            var width = checked((int)batch.shape[3]);
            var longSide = random.Next(
                Math.Max(3, Math.Min(height, width) / 8),
                Math.Max(4, Math.Min(height, width) / 2 + 1));
            var shortSide = random.Next(2, Math.Max(3, Math.Min(height, width) / 20 + 1));
            var horizontal = random.Next(2) == 0;
            var patchHeight = horizontal ? shortSide : longSide;
            var patchWidth = horizontal ? longSide : shortSide;
            PastePatch(
                batch,
                batchIndex,
                Math.Min(patchHeight, Math.Max(2, height - 1)),
                Math.Min(patchWidth, Math.Max(2, width - 1)),
                random);
        }

        private static void PastePatch(
            Tensor batch,
            long batchIndex,
            int patchHeight,
            int patchWidth,
            Random random)
        {
            var height = checked((int)batch.shape[2]);
            var width = checked((int)batch.shape[3]);
            var sourceY = random.Next(0, height - patchHeight + 1);
            var sourceX = random.Next(0, width - patchWidth + 1);
            var targetY = random.Next(0, height - patchHeight + 1);
            var targetX = random.Next(0, width - patchWidth + 1);

            for (var retry = 0; retry < 4 && targetY == sourceY && targetX == sourceX; retry++)
            {
                targetY = random.Next(0, height - patchHeight + 1);
                targetX = random.Next(0, width - patchWidth + 1);
            }

            using (var tile = batch[batchIndex])
            using (var sourceView = tile
                .narrow(1, sourceY, patchHeight)
                .narrow(2, sourceX, patchWidth))
            using (var source = sourceView.clone())
            using (var target = tile
                .narrow(1, targetY, patchHeight)
                .narrow(2, targetX, patchWidth))
            {
                var contrast = 0.85 + random.NextDouble() * 0.30;
                source.mul_(contrast);
                target.copy_(source);
            }
        }

        private static int Clamp(int value, int minimum, int maximum)
        {
            return Math.Max(minimum, Math.Min(maximum, value));
        }
    }

    internal sealed class SyntheticPretextBatch : IDisposable
    {
        public SyntheticPretextBatch(Tensor inputs, Tensor targets)
        {
            Inputs = inputs;
            Targets = targets;
        }

        public Tensor Inputs { get; private set; }
        public Tensor Targets { get; private set; }

        public void Dispose()
        {
            Inputs?.Dispose();
            Targets?.Dispose();
            Inputs = null;
            Targets = null;
        }
    }
}
