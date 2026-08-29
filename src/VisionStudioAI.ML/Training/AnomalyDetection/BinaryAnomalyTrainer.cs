using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TorchSharp;
using VisionStudioAI.Core.Models;
using VisionStudioAI.ML.Models.AnomalyDetection;
using static TorchSharp.torch;
using static TorchSharp.torch.nn.functional;
using static TorchSharp.torch.optim;

namespace VisionStudioAI.ML.Training.AnomalyDetection
{
    public sealed class BinaryAnomalyTrainingProgress
    {
        public int Epoch { get; set; }
        public float TrainingLoss { get; set; }
        public float ValidationLoss { get; set; }
    }

    /// <summary>
    /// Trains the similarity encoder from scratch on OK tiles. Two synthetic
    /// CutPaste variants provide the learning signal; real NOK images are never
    /// passed to the optimizer.
    /// </summary>
    public sealed class BinaryAnomalyTrainer
    {
        private readonly ILogger<BinaryAnomalyTrainer> logger;

        public BinaryAnomalyTrainer(ILogger<BinaryAnomalyTrainer> logger)
        {
            this.logger = logger;
        }

        public async Task TrainAsync(
            SimilarityAnomalyModel model,
            IReadOnlyList<AnomalyImageSample> trainingSamples,
            IReadOnlyList<AnomalyImageSample> validationSamples,
            PreprocessingSettings preprocessing,
            BinaryAnomalyModelConfig config,
            Device device,
            int maxEpochs,
            int maxOriginalTilesPerBatch,
            IProgress<BinaryAnomalyTrainingProgress> progress,
            CancellationToken cancellationToken)
        {
            if (model == null) throw new ArgumentNullException(nameof(model));
            if (trainingSamples == null) throw new ArgumentNullException(nameof(trainingSamples));
            if (validationSamples == null) throw new ArgumentNullException(nameof(validationSamples));
            if (preprocessing == null) throw new ArgumentNullException(nameof(preprocessing));
            if (maxEpochs <= 0) throw new ArgumentOutOfRangeException(nameof(maxEpochs));
            if (maxOriginalTilesPerBatch <= 0)
                throw new ArgumentOutOfRangeException(nameof(maxOriginalTilesPerBatch));
            config.Validate();

            var trainingPaths = GetOkTilePaths(trainingSamples, "training");
            var validationPaths = GetOkTilePaths(validationSamples, "validation");

            using (var optimizer = AdamW(
                model.parameters(),
                lr: config.LearningRate,
                weight_decay: config.WeightDecay))
            {
                for (var epoch = 1; epoch <= maxEpochs; epoch++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    model.train(true);
                    model.ConfigureForEpoch(epoch);

                    var shuffled = trainingPaths.ToList();
                    Shuffle(shuffled, new Random(epoch));
                    var trainingLoss = RunTrainingEpoch(
                        model,
                        optimizer,
                        shuffled,
                        preprocessing,
                        config,
                        device,
                        maxOriginalTilesPerBatch,
                        epoch,
                        cancellationToken);
                    var validationLoss = RunValidationEpoch(
                        model,
                        validationPaths,
                        preprocessing,
                        config,
                        device,
                        maxOriginalTilesPerBatch,
                        cancellationToken);

                    progress?.Report(new BinaryAnomalyTrainingProgress
                    {
                        Epoch = epoch,
                        TrainingLoss = trainingLoss,
                        ValidationLoss = validationLoss
                    });

                    logger?.LogInformation(
                        Environment.NewLine + "Epoch {Epoch}: " +
                        Environment.NewLine + "train={TrainLoss:F4}, " + 
                        "validation={ValidationLoss:F4}",
                        epoch,
                        trainingLoss,
                        validationLoss);
                    await Task.Yield();
                }
            }
        }

        private static float RunTrainingEpoch(
            SimilarityAnomalyModel model,
            Optimizer optimizer,
            IReadOnlyList<string> tilePaths,
            PreprocessingSettings preprocessing,
            BinaryAnomalyModelConfig config,
            Device device,
            int batchSize,
            int epoch,
            CancellationToken cancellationToken)
        {
            double totalLoss = 0;
            var processed = 0;
            var batchIndex = 0;

            foreach (var paths in Batch(tilePaths, batchSize))
            {
                cancellationToken.ThrowIfCancellationRequested();
                using (var scope = NewDisposeScope())
                using (var okTiles = AnomalyImageBagLoader.LoadTilePaths(
                    paths,
                    preprocessing.TrainAsGreyscale,
                    preprocessing.Normalization,
                    device,
                    config.TrainPrecision))
                using (var synthetic = CutPasteBatchGenerator.Create(
                    okTiles,
                    config.CutPasteMinAreaFraction,
                    config.CutPasteMaxAreaFraction,
                    unchecked(epoch * 100_003 + batchIndex)))
                {
                    optimizer.zero_grad();
                    var logits = model.call(synthetic.Inputs);
                    var loss = cross_entropy(logits, synthetic.Targets);
                    loss.backward();
                    optimizer.step();

                    totalLoss += loss.ToSingle() * paths.Count;
                    processed += paths.Count;
                }

                batchIndex++;
            }

            return processed == 0 ? 0 : (float)(totalLoss / processed);
        }

        private static float RunValidationEpoch(
            SimilarityAnomalyModel model,
            IReadOnlyList<string> tilePaths,
            PreprocessingSettings preprocessing,
            BinaryAnomalyModelConfig config,
            Device device,
            int batchSize,
            CancellationToken cancellationToken)
        {
            model.eval();
            double totalLoss = 0;
            var processed = 0;
            var batchIndex = 0;

            using (var inference = inference_mode())
            {
                foreach (var paths in Batch(tilePaths, batchSize))
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    using (var scope = NewDisposeScope())
                    using (var okTiles = AnomalyImageBagLoader.LoadTilePaths(
                        paths,
                        preprocessing.TrainAsGreyscale,
                        preprocessing.Normalization,
                        device,
                        config.TrainPrecision))
                    using (var synthetic = CutPasteBatchGenerator.Create(
                        okTiles,
                        config.CutPasteMinAreaFraction,
                        config.CutPasteMaxAreaFraction,
                        17_123 + batchIndex))
                    {
                        var logits = model.call(synthetic.Inputs);
                        var loss = cross_entropy(logits, synthetic.Targets);
                        totalLoss += loss.ToSingle() * paths.Count;
                        processed += paths.Count;
                    }

                    batchIndex++;
                }
            }

            return processed == 0 ? 0 : (float)(totalLoss / processed);
        }

        private static IReadOnlyList<string> GetOkTilePaths(
            IReadOnlyList<AnomalyImageSample> samples,
            string splitName)
        {
            var paths = samples
                .Where(sample => sample.Label == AnomalyLabel.Ok)
                .SelectMany(sample => sample.TilePaths)
                .ToList();
            if (paths.Count == 0)
                throw new InvalidOperationException(
                    "The " + splitName + " split requires at least one OK image with tiles.");
            return paths;
        }

        private static IEnumerable<IReadOnlyList<string>> Batch(
            IReadOnlyList<string> paths,
            int batchSize)
        {
            for (var offset = 0; offset < paths.Count; offset += batchSize)
            {
                var count = Math.Min(batchSize, paths.Count - offset);
                var batch = new List<string>(count);
                for (var index = 0; index < count; index++)
                    batch.Add(paths[offset + index]);
                yield return batch;
            }
        }

        private static void Shuffle<T>(IList<T> items, Random random)
        {
            for (var index = items.Count - 1; index > 0; index--)
            {
                var other = random.Next(index + 1);
                var value = items[index];
                items[index] = items[other];
                items[other] = value;
            }
        }
    }
}
