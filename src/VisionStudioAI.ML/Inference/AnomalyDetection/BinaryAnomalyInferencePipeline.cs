using System;
using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using OpenCvSharp;
using TorchSharp;
using VisionStudioAI.Core.Models;
using VisionStudioAI.Core.Services;
using VisionStudioAI.ML.Geometry;
using VisionStudioAI.ML.Processing;
using static TorchSharp.torch;
using static VisionStudioAI.ML.Utils.CudaOps.NativeTorchCudaOps;
using static VisionStudioAI.ML.Utils.ImageProcessing.ImageUtils;
using static VisionStudioAI.ML.Utils.TensorProcessing.TensorConversion;

namespace VisionStudioAI.ML.Inference.AnomalyDetection
{
    public static class AnomalyInferenceResults
    {
        public const string HeatmapName = "Anomaly";
    }

    internal sealed class BinaryAnomalyInferencePipeline : IBinaryAnomalyInferencePipeline
    {
        private readonly JsonSerializerOptions jsonOptions;

        public BinaryAnomalyInferencePipeline(JsonSerializerOptions jsonOptions)
        {
            this.jsonOptions = jsonOptions ?? throw new ArgumentNullException(nameof(jsonOptions));
        }

        public async Task RunInference(
            IProjectPresenter project,
            string artifactDirectory,
            IProgress<int> progress,
            CancellationToken cancellationToken)
        {
            if (project == null) throw new ArgumentNullException(nameof(project));
            if (string.IsNullOrWhiteSpace(artifactDirectory))
                throw new ArgumentException("Artifact directory is required.", nameof(artifactDirectory));

            var device = new Device(
                project.Project.Settings.TrainModelSettings.Device == ComputeDevice.Cpu
                    ? DeviceType.CPU
                    : DeviceType.CUDA);

            try
            {
                using (var session = BinaryAnomalyInferenceSession.Load(
                    artifactDirectory,
                    device,
                    jsonOptions))
                {
                    var preprocessing = session.Package.Settings.PreprocessingSettings;
                    var precision = session.Package.ModelConfig.TrainPrecision;
                    var heatmapDirectory = Path.Combine(
                        project.Paths.MasksHeatmaps,
                        AnomalyInferenceResults.HeatmapName);
                    Directory.CreateDirectory(heatmapDirectory);

                    var imageIndex = 0;
                    var total = project.Project.Images.Count;

                    foreach (var item in project.Project.Images)
                    {
                        cancellationToken.ThrowIfCancellationRequested();

                        var imagePath = Path.Combine(
                            project.Paths.Images,
                            item.Guid + project.Paths.ImagesExt);

                        using (var image = Cv2.ImRead(
                            imagePath,
                            preprocessing.TrainAsGreyscale
                                ? ImreadModes.Grayscale
                                : ImreadModes.Color))
                        {
                            if (image.Empty())
                                throw new InvalidOperationException(
                                    "Could not load anomaly inference image: " + imagePath);

                            var imageSpace = new SegmentationImageSpace(
                                new OpenCvSharp.Size(item.ImageSize.Width, item.ImageSize.Height),
                                new OpenCvSharp.Rect(item.Roi.X, item.Roi.Y, item.Roi.Width, item.Roi.Height),
                                preprocessing.SliceSize,
                                preprocessing.DownSample,
                                preprocessing.BorderPadding);
                            var preprocessor = new SegmentationPreprocessor(imageSpace);
                            var postprocessor = new SegmentationPostprocessor(imageSpace);
                            var tiles = preprocessor.ProcessImage(image);

                            try
                            {
                                var stopwatch = Stopwatch.StartNew();

                                using (var input = SlicedImageToTensor(
                                    tiles,
                                    preprocessing.TrainAsGreyscale,
                                    device,
                                    preprocessing.Normalization,
                                    precision))
                                using (var prediction = session.Run(input))
                                using (var heatmap = AnomalyHeatmapStitcher.Stitch(
                                    prediction.TileHeatmaps,
                                    preprocessing.SliceSize,
                                    preprocessing.SliceSize,
                                    postprocessor))
                                {
                                    stopwatch.Stop();

                                    var heatmapPath = Path.Combine(
                                        heatmapDirectory,
                                        item.Guid + project.Paths.ImagesExt);
                                    if (!Cv2.ImWrite(heatmapPath, heatmap))
                                        throw new IOException("Could not save anomaly heatmap: " + heatmapPath);

                                    item.AnomalyScore = prediction.Score;
                                    item.PredictedAnomalyLabel = prediction.IsNotOk
                                        ? AnomalyLabel.NotOk
                                        : AnomalyLabel.Ok;
                                    item.InferenceMs = stopwatch.Elapsed.TotalMilliseconds;
                                }
                            }
                            finally
                            {
                                DisposeTiles(tiles);
                            }
                        }

                        imageIndex++;
                        progress?.Report(total == 0 ? 100 : (int)(100.0 * imageIndex / total));

                        if (device.type == DeviceType.CUDA)
                            EmptyCudaCache();
                    }
                }

                if (project.Project.Images.Count == 0)
                    progress?.Report(100);

                await Task.Yield();
            }
            finally
            {
                if (device.type == DeviceType.CUDA)
                    EmptyCudaCache();
            }
        }
    }
}
