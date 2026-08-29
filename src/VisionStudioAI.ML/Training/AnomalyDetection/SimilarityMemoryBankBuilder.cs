using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using VisionStudioAI.Core.Models;
using VisionStudioAI.ML.Models.AnomalyDetection;
using static TorchSharp.torch;

namespace VisionStudioAI.ML.Training.AnomalyDetection
{
    public sealed class SimilarityMemoryBankBuilder
    {
        public SimilarityMemoryBank Build(
            SimilarityAnomalyModel model,
            IReadOnlyList<AnomalyImageSample> samples,
            PreprocessingSettings preprocessing,
            BinaryAnomalyModelConfig config,
            Device device,
            int tileBatchSize,
            CancellationToken cancellationToken)
        {
            if (model == null) throw new ArgumentNullException(nameof(model));
            if (samples == null) throw new ArgumentNullException(nameof(samples));
            if (preprocessing == null) throw new ArgumentNullException(nameof(preprocessing));
            if (tileBatchSize <= 0) throw new ArgumentOutOfRangeException(nameof(tileBatchSize));
            config.Validate();

            var okPaths = samples
                .Where(sample => sample.Label == AnomalyLabel.Ok)
                .SelectMany(sample => sample.TilePaths)
                .ToList();
            if (okPaths.Count == 0)
                throw new InvalidOperationException("At least one OK image is required.");

            var retained = new List<Tensor>();
            model.eval();
            try
            {
                using (var inference = inference_mode())
                {
                    for (var offset = 0; offset < okPaths.Count; offset += tileBatchSize)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        var paths = okPaths.Skip(offset).Take(tileBatchSize).ToList();
                        using (var scope = NewDisposeScope())
                        using (var tiles = AnomalyImageBagLoader.LoadTilePaths(
                            paths,
                            preprocessing.TrainAsGreyscale,
                            preprocessing.Normalization,
                            device,
                            config.TrainPrecision))
                        {
                            var map = model.ExtractPatchEmbeddings(tiles);
                            var descriptors = map
                                .permute(0, 2, 3, 1)
                                .contiguous()
                                .reshape(-1, map.shape[1]);
                            var desired = Math.Min(
                                descriptors.shape[0],
                                Math.Max(1, (long)Math.Ceiling(
                                    descriptors.shape[0] * config.MemoryBankSamplingRatio)));
                            var indices = BuildUniformIndices(descriptors.shape[0], desired);
                            var selected = descriptors
                                .index_select(0, tensor(indices, dtype: ScalarType.Int64, device: device))
                                .detach()
                                .to_type(ScalarType.Float32)
                                .cpu()
                                .contiguous()
                                .MoveToOuterDisposeScope();
                            retained.Add(selected);
                        }
                    }
                }

                using (var scope = NewDisposeScope())
                {
                    var all = cat(retained.ToArray(), 0);
                    if (all.shape[0] > config.MaxMemoryEntries)
                    {
                        var indices = BuildUniformIndices(all.shape[0], config.MaxMemoryEntries);
                        all = all.index_select(0, tensor(indices, dtype: ScalarType.Int64));
                    }

                    var memoryBank = new SimilarityMemoryBank(all);
                    memoryBank.Embeddings.MoveToOuterDisposeScope();
                    return memoryBank;
                }
            }
            finally
            {
                foreach (var descriptor in retained) descriptor.Dispose();
            }
        }

        private static long[] BuildUniformIndices(long count, long desired)
        {
            if (desired >= count)
                return Enumerable.Range(0, checked((int)count)).Select(index => (long)index).ToArray();

            var result = new long[checked((int)desired)];
            var scale = (double)count / desired;
            for (var index = 0; index < result.Length; index++)
                result[index] = Math.Min(count - 1, (long)Math.Floor((index + 0.5) * scale));
            return result;
        }
    }
}
