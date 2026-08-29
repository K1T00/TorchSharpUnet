using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using VisionStudioAI.Core.Models;
using VisionStudioAI.ML.Models.AnomalyDetection;
using VisionStudioAI.ML.Training.AnomalyDetection;
using static TorchSharp.torch;

namespace VisionStudioAI.ML.Inference.AnomalyDetection
{
    public sealed class AnomalyCalibrationBuilder
    {
        public SimilarityAnomalyCalibration Build(
            SimilarityAnomalyModel model,
            SimilarityMemoryBank memoryBank,
            IReadOnlyList<AnomalyImageSample> validationSamples,
            PreprocessingSettings preprocessing,
            BinaryAnomalyModelConfig config,
            Device device,
            CancellationToken cancellationToken)
        {
            if (validationSamples == null) throw new ArgumentNullException(nameof(validationSamples));

            var values = new List<SimilarityValidationScore>(validationSamples.Count);
            model.eval();
            using (var nearest = new SimilarityNearestNeighbourScorer(memoryBank, config, device))
            using (var inference = inference_mode())
            {
                foreach (var sample in validationSamples)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    using (var scope = NewDisposeScope())
                    using (var tiles = AnomalyImageBagLoader.Load(
                        sample,
                        preprocessing.TrainAsGreyscale,
                        preprocessing.Normalization,
                        device,
                        config.TrainPrecision))
                    using (var score = nearest.Score(model, tiles))
                    {
                        var patchScores = sample.Label == AnomalyLabel.Ok
                            ? score.TileScoreMaps
                                .detach()
                                .to_type(ScalarType.Float32)
                                .cpu()
                                .contiguous()
                                .data<float>()
                                .ToArray()
                            : new float[0];
                        values.Add(new SimilarityValidationScore
                        {
                            IsNotOk = sample.Label == AnomalyLabel.NotOk,
                            ImageScore = score.ImageScore,
                            PatchScores = patchScores
                        });
                    }
                }
            }

            return SimilarityAnomalyCalibration.Fit(values, config);
        }
    }
}
