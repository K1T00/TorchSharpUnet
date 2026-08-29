using System;
using System.Collections.Generic;
using System.Linq;
using VisionStudioAI.ML.Models.AnomalyDetection;
using static TorchSharp.torch;

namespace VisionStudioAI.ML.Inference.AnomalyDetection
{
    public sealed class LinearScoreCalibration
    {
        public float Low { get; set; }
        public float High { get; set; } = 1;

        public float Normalize(float value)
        {
            var range = Math.Max(1e-8f, High - Low);
            return Math.Max(0, Math.Min(1, (value - Low) / range));
        }

        public Tensor Normalize(Tensor values)
        {
            var range = Math.Max(1e-8f, High - Low);
            return ((values - Low) / range).clamp(0.0, 1.0);
        }
    }

    public sealed class SimilarityValidationScore
    {
        public bool IsNotOk { get; set; }
        public float ImageScore { get; set; }
        public float[] PatchScores { get; set; } = new float[0];
    }

    /// <summary>
    /// Keeps image classification and heatmap display calibration separate.
    /// The decision is made in raw distance space so visualization scaling can
    /// never change the OK/NOK result.
    /// </summary>
    public sealed class SimilarityAnomalyCalibration
    {
        public LinearScoreCalibration ImageScore { get; set; } = new LinearScoreCalibration();
        public LinearScoreCalibration Heatmap { get; set; } = new LinearScoreCalibration();
        public float RawDecisionThreshold { get; set; }

        public static SimilarityAnomalyCalibration Fit(
            IReadOnlyList<SimilarityValidationScore> validation,
            BinaryAnomalyModelConfig config)
        {
            if (validation == null) throw new ArgumentNullException(nameof(validation));
            if (validation.Count == 0)
                throw new ArgumentException("Validation scores are empty.", nameof(validation));
            if (config == null) throw new ArgumentNullException(nameof(config));
            config.Validate();

            var ok = validation.Where(value => !value.IsNotOk).ToList();
            if (ok.Count == 0)
                throw new InvalidOperationException("Calibration requires at least one OK validation image.");

            var okImageScores = ok.Select(value => value.ImageScore).OrderBy(value => value).ToArray();
            var okPatchScores = ok
                .SelectMany(value => value.PatchScores ?? new float[0])
                .OrderBy(value => value)
                .ToArray();
            if (okPatchScores.Length == 0)
                okPatchScores = okImageScores;

            var hasNotOk = validation.Any(value => value.IsNotOk);
            var threshold = hasNotOk
                ? SelectBalancedThreshold(validation)
                : BuildOkOnlyThreshold(okImageScores, config.OkOnlyThresholdMargin);

            var imageLow = Percentile(okImageScores, 0.05);
            var allImageScores = validation.Select(value => value.ImageScore).OrderBy(value => value).ToArray();
            var imageHigh = Math.Max(
                Percentile(allImageScores, 0.95),
                Math.Max(threshold * 1.25f, imageLow + 1e-6f));
            var heatmapLow = Percentile(okPatchScores, 0.95);
            var heatmapHigh = Math.Max(
                Percentile(okPatchScores, 0.999),
                heatmapLow + Math.Max(1e-6f, Math.Abs(heatmapLow) * 0.05f));

            return new SimilarityAnomalyCalibration
            {
                ImageScore = new LinearScoreCalibration { Low = imageLow, High = imageHigh },
                Heatmap = new LinearScoreCalibration { Low = heatmapLow, High = heatmapHigh },
                RawDecisionThreshold = threshold
            };
        }

        private static float BuildOkOnlyThreshold(float[] okScores, double margin)
        {
            var observedUpper = Percentile(okScores, 0.995);
            var added = Math.Max(1e-6f, Math.Abs(observedUpper) * (float)margin);
            return observedUpper + added;
        }

        private static float SelectBalancedThreshold(
            IReadOnlyList<SimilarityValidationScore> values)
        {
            var distinct = values.Select(value => value.ImageScore)
                .Distinct()
                .OrderBy(value => value)
                .ToArray();
            var candidates = new List<float>();
            var epsilon = Math.Max(1e-6f, Math.Abs(distinct[0]) * 1e-6f);
            candidates.Add(distinct[0] - epsilon);
            for (var index = 1; index < distinct.Length; index++)
                candidates.Add((distinct[index - 1] + distinct[index]) / 2f);
            candidates.Add(distinct[distinct.Length - 1] + epsilon);

            var positives = values.Count(value => value.IsNotOk);
            var negatives = values.Count - positives;
            var bestThreshold = candidates[0];
            var bestBalancedAccuracy = double.NegativeInfinity;
            var bestSpecificity = double.NegativeInfinity;

            foreach (var candidate in candidates)
            {
                var truePositives = values.Count(value => value.IsNotOk && value.ImageScore >= candidate);
                var trueNegatives = values.Count(value => !value.IsNotOk && value.ImageScore < candidate);
                var recall = (double)truePositives / positives;
                var specificity = (double)trueNegatives / negatives;
                var balancedAccuracy = (recall + specificity) / 2.0;

                if (balancedAccuracy > bestBalancedAccuracy ||
                    (Math.Abs(balancedAccuracy - bestBalancedAccuracy) < 1e-12 &&
                     specificity > bestSpecificity))
                {
                    bestBalancedAccuracy = balancedAccuracy;
                    bestSpecificity = specificity;
                    bestThreshold = candidate;
                }
            }

            return bestThreshold;
        }

        private static float Percentile(float[] sorted, double percentile)
        {
            if (sorted.Length == 0) throw new ArgumentException("Values are empty.", nameof(sorted));
            var position = percentile * (sorted.Length - 1);
            var lower = (int)Math.Floor(position);
            var upper = (int)Math.Ceiling(position);
            if (lower == upper) return sorted[lower];
            var fraction = position - lower;
            return (float)(sorted[lower] * (1 - fraction) + sorted[upper] * fraction);
        }
    }
}
