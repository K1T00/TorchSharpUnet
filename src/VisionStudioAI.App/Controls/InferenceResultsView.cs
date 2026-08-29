using ScottPlot;
using VisionStudioAI.Core.Models;
using static VisionStudioAI.Core.Utils.CoreUtils;

namespace VisionStudioAI.App.Controls
{
    public partial class InferenceResultsView : UserControl
    {
        private SegmentationStats? segmentationStats;
        private IReadOnlyList<DisplayMetric> displayMetrics = Array.Empty<DisplayMetric>();

        // WinForms tooltip for metric descriptions
        private readonly ToolTip radarTooltip = new ToolTip();

        // Remember which bar we are currently over (null = none)
        private int? lastHoveredBarIndex;

        private sealed record DisplayMetric(
            string Name,
            string Description,
            double ValuePercent,
            ScottPlot.Color Color);

        public InferenceResultsView()
        {
            InitializeComponent();

            radarTooltip.AutoPopDelay = 6000;
            radarTooltip.InitialDelay = 400; // slightly longer to make it feel stable
            radarTooltip.ReshowDelay = 100;
            radarTooltip.ShowAlways = true;
        }

        public SegmentationStats? SegmentationStats
        {
            get => segmentationStats;
            set
            {
                segmentationStats = value;

                if (value == null)
                {
                    ClearPlot();
                    return;
                }

                SetMetrics(
                    new DisplayMetric(
                        "Dice",
                        "Measures overlap between prediction and mask. 100% = perfect match.",
                        value.Dice * 100,
                        Colors.Green),
                    new DisplayMetric(
                        "Precision",
                        "Of all predicted defects, how many were correct. High = fewer false alarms.",
                        value.Precision * 100,
                        Colors.Green),
                    new DisplayMetric(
                        "Recall",
                        "Of all real defects, how many were detected. High = fewer missed defects.",
                        value.Recall * 100,
                        Colors.Green),
                    new DisplayMetric(
                        "FPR",
                        "False-positive rate. Lower is better.",
                        value.FPR * 100,
                        Colors.Red),
                    new DisplayMetric(
                        "DiceLoss",
                        "1 - Dice score. Lower means better segmentation.",
                        (1 - value.Dice) * 100,
                        Colors.Red));
            }
        }

        /// <summary>
        /// Shows the anomaly result for a single image. The score is expected to be normalized to [0, 1].
        /// </summary>
        public void ShowAnomalyImageResult(
            double anomalyScore,
            AnomalyLabel predictedLabel,
            AnomalyLabel annotatedLabel)
        {
            segmentationStats = null;

            var scorePercent = NormalizePercent(anomalyScore * 100);
            var predictionIsCorrect = predictedLabel == annotatedLabel;
            var predictedText = ToDisplayText(predictedLabel);
            var annotatedText = ToDisplayText(annotatedLabel);

            SetMetrics(
                new DisplayMetric(
                    "Anomaly",
                    $"Model anomaly score. Predicted class: {predictedText}.",
                    scorePercent,
                    predictedLabel == AnomalyLabel.NotOk ? Colors.Red : Colors.Green),
                new DisplayMetric(
                    "Normality",
                    "Complement of the anomaly score. Higher values indicate a stronger OK result.",
                    100 - scorePercent,
                    Colors.Green),
                new DisplayMetric(
                    "Correct",
                    $"Prediction: {predictedText}. Image annotation: {annotatedText}.",
                    predictionIsCorrect ? 100 : 0,
                    predictionIsCorrect ? Colors.Green : Colors.Red));
        }

        /// <summary>
        /// Shows image-level anomaly classification metrics for all supplied images.
        /// NOK is treated as the positive class.
        /// </summary>
        public void ShowAnomalyAggregateResults(IEnumerable<ImageItem> images)
        {
            ArgumentNullException.ThrowIfNull(images);

            segmentationStats = null;
            var evaluatedImages = images.ToList();
            if (evaluatedImages.Count == 0)
            {
                ClearPlot();
                return;
            }

            var truePositive = evaluatedImages.Count(image =>
                image.AnomalyLabel == AnomalyLabel.NotOk &&
                image.PredictedAnomalyLabel == AnomalyLabel.NotOk);
            var trueNegative = evaluatedImages.Count(image =>
                image.AnomalyLabel == AnomalyLabel.Ok &&
                image.PredictedAnomalyLabel == AnomalyLabel.Ok);
            var falsePositive = evaluatedImages.Count(image =>
                image.AnomalyLabel == AnomalyLabel.Ok &&
                image.PredictedAnomalyLabel == AnomalyLabel.NotOk);
            var falseNegative = evaluatedImages.Count(image =>
                image.AnomalyLabel == AnomalyLabel.NotOk &&
                image.PredictedAnomalyLabel == AnomalyLabel.Ok);

            var accuracy = Divide(truePositive + trueNegative, evaluatedImages.Count);
            var precision = Divide(truePositive, truePositive + falsePositive);
            var recall = Divide(truePositive, truePositive + falseNegative);
            var falsePositiveRate = Divide(falsePositive, falsePositive + trueNegative);
            var f1 = Divide(2 * truePositive, 2 * truePositive + falsePositive + falseNegative);
            var counts = $"TP: {truePositive}, TN: {trueNegative}, FP: {falsePositive}, FN: {falseNegative}.";

            SetMetrics(
                new DisplayMetric(
                    "Accuracy",
                    $"Correct image classifications across the complete set. {counts}",
                    accuracy * 100,
                    Colors.Green),
                new DisplayMetric(
                    "Precision",
                    $"Of all images predicted as NOK, how many were annotated NOK. {counts}",
                    precision * 100,
                    Colors.Green),
                new DisplayMetric(
                    "Recall",
                    $"Of all images annotated NOK, how many were detected. {counts}",
                    recall * 100,
                    Colors.Green),
                new DisplayMetric(
                    "FPR",
                    $"Of all images annotated OK, how many were incorrectly predicted NOK. {counts}",
                    falsePositiveRate * 100,
                    Colors.Red),
                new DisplayMetric(
                    "F1",
                    $"Harmonic mean of NOK precision and recall. {counts}",
                    f1 * 100,
                    Colors.Green));
        }

        private void InferenceResultsControl_Load(object? sender, EventArgs e)
        {
            UpdatePlot();
        }

        public void ClearPlot()
        {
            segmentationStats = null;
            displayMetrics = Array.Empty<DisplayMetric>();
            lastHoveredBarIndex = null;
            radarTooltip.Hide(radarPlotResults);
            radarPlotResults.Plot.Clear();
            radarPlotResults.Refresh();
        }

        public void UpdatePlot()
        {
            radarPlotResults.Plot.Clear();

            if (displayMetrics.Count == 0)
            {
                radarPlotResults.Refresh();
                return;
            }

            var ticks = new Tick[displayMetrics.Count];
            for (var i = 0; i < displayMetrics.Count; i++)
            {
                var metric = displayMetrics[i];
                var position = i + 1;
                var bar = radarPlotResults.Plot.Add.Bar(position, metric.ValuePercent);
                bar.Color = metric.Color;
                ticks[i] = new Tick(position, metric.Name);
            }

            radarPlotResults.Plot.Axes.Bottom.TickGenerator = new ScottPlot.TickGenerators.NumericManual(ticks);
            radarPlotResults.Plot.Axes.Bottom.MajorTickStyle.Length = 0;
            radarPlotResults.Plot.HideGrid();

            // Auto-scale with no padding beneath the bars
            radarPlotResults.Plot.Axes.Margins(bottom: 0);
            radarPlotResults.Plot.Axes.SetLimitsY(0, 100);

            if (IsDarkMode())
            {
                radarPlotResults.Plot.Axes.Bottom.TickLabelStyle.ForeColor = Colors.White;
                radarPlotResults.Plot.Axes.Left.TickLabelStyle.ForeColor = Colors.White;
            }
            else
            {
                radarPlotResults.Plot.Axes.Bottom.TickLabelStyle.ForeColor = Colors.Black;
                radarPlotResults.Plot.Axes.Left.TickLabelStyle.ForeColor = Colors.Black;
            }

            radarPlotResults.Refresh();
        }

        private void SetMetrics(params DisplayMetric[] metrics)
        {
            displayMetrics = metrics
                .Select(metric => metric with { ValuePercent = NormalizePercent(metric.ValuePercent) })
                .ToArray();
            lastHoveredBarIndex = null;
            radarTooltip.Hide(radarPlotResults);
            UpdatePlot();
        }

        private void radarPlotResults_MouseMove(object? sender, MouseEventArgs e)
        {
            if (displayMetrics.Count == 0)
                return;

            var coordinates = radarPlotResults.Plot.GetCoordinates(e.X, e.Y);
            int? hoveredIndex = null;

            for (var i = 0; i < displayMetrics.Count; i++)
            {
                var center = i + 1;
                var barHeight = displayMetrics[i].ValuePercent;

                if (Math.Abs(coordinates.X - center) < 0.5 &&
                    coordinates.Y >= 0 &&
                    coordinates.Y <= barHeight)
                {
                    hoveredIndex = i;
                    break;
                }
            }

            // If hover target hasn't changed, do nothing - prevents flicker.
            if (hoveredIndex == lastHoveredBarIndex)
                return;

            lastHoveredBarIndex = hoveredIndex;

            if (hoveredIndex is int index)
            {
                var metric = displayMetrics[index];
                var text = $"{metric.Name}: {metric.ValuePercent:0.0}%\r\n{metric.Description}";
                radarTooltip.Show(text, radarPlotResults, e.X + 15, e.Y + 15);
            }
            else
            {
                radarTooltip.Hide(radarPlotResults);
            }
        }

        private static double Divide(int numerator, int denominator) =>
            denominator == 0 ? 0 : (double)numerator / denominator;

        private static double NormalizePercent(double value) =>
            double.IsFinite(value) ? Math.Clamp(value, 0, 100) : 0;

        private static string ToDisplayText(AnomalyLabel label) =>
            label == AnomalyLabel.NotOk ? "NOK" : "OK";
    }
}
