using VisionStudioAI.Core.Models;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Threading.Tasks;
using VisionStudioAI.Core.Services;
using Point = System.Drawing.Point;

namespace VisionStudioAI.ML.Inference
{
    public static class ObjectDetectionHeatmapPostprocessor
    {
        public const byte ObjectCandidateExtractionThreshold = 1;
        public const double DefaultMinimumObjectAreaRatio = 0.65;

        public static List<ObjectLocation> ExtractCenters(
            Mat responseMap8u,
            int classId,
            Rectangle roi,
            int downSample,
            byte threshold,
            int minBlobArea = 9,
            int closingKernelSize = 5,
            int closingIterations = 1,
            int openingKernelSize = 3,
            int openingIterations = 1,
            int objectDiameter = 0,
            double minimumObjectAreaRatio = DefaultMinimumObjectAreaRatio)
        {
            if (responseMap8u == null)
                throw new ArgumentNullException(nameof(responseMap8u));

            if (responseMap8u.Empty())
                return new List<ObjectLocation>();

            if (responseMap8u.Type() != MatType.CV_8UC1)
                throw new ArgumentException("Expected a single-channel 8-bit response map.", nameof(responseMap8u));

            using (var binary = new Mat())
            using (var processedBinary = new Mat())
            {
                var labels = new Mat();
                var stats = new Mat();
                var centroids = new Mat();

                Cv2.Threshold(responseMap8u, binary, threshold, 255, ThresholdTypes.Binary);
                ApplyMorphology(binary, processedBinary, closingKernelSize, closingIterations, openingKernelSize, openingIterations);
                var labelCount = Cv2.ConnectedComponentsWithStats(processedBinary, labels, stats, centroids, PixelConnectivity.Connectivity8, MatType.CV_32S);
                var minimumArea = CalculateMinimumBlobArea(minBlobArea, objectDiameter, downSample, minimumObjectAreaRatio);

                try
                {
                    var detections = new List<ObjectLocation>();

                    for (var label = 1; label < labelCount; label++)
                    {
                        var area = stats.At<int>(label, (int)ConnectedComponentsTypes.Area);
                        if (area < minimumArea)
                            continue;

                        var cxWorking = centroids.At<double>(label, 0);
                        var cyWorking = centroids.At<double>(label, 1);

                        var probabilityPercent = ComputeAverageProbabilityPercent(responseMap8u, labels, label);

                        detections.Add(new ObjectLocation
                        {
                            ClassId = classId,
                            Center = new Point((int)cxWorking, (int)cyWorking),
                            ProbabilityPercent = probabilityPercent
                        });
                    }

                    return detections;
                }
                finally
                {
                    labels.Dispose();
                    stats.Dispose();
                    centroids.Dispose();
                }
            }
        }

        private static int CalculateMinimumBlobArea(int fallbackMinimumArea, int objectDiameter, int downSample, double minimumObjectAreaRatio)
        {
            var minimumArea = Math.Max(1, fallbackMinimumArea);
            if (objectDiameter <= 0 || minimumObjectAreaRatio <= 0.0)
                return minimumArea;

            if (downSample < 0)
                downSample = 0;

            var effectiveDiameter = objectDiameter * (1 << downSample);
            var radius = effectiveDiameter / 2.0;
            var expectedArea = Math.PI * radius * radius;
            var diameterBasedMinimumArea = (int)Math.Ceiling(expectedArea * minimumObjectAreaRatio);

            return Math.Max(minimumArea, diameterBasedMinimumArea);
        }

        private static void ApplyMorphology(
            Mat binary,
            Mat destination,
            int closingKernelSize,
            int closingIterations,
            int openingKernelSize,
            int openingIterations)
        {
            binary.CopyTo(destination);

            if (closingKernelSize > 1 && closingIterations > 0)
            {
                using (var kernel = CreateMorphologyKernel(closingKernelSize))
                {
                    Cv2.MorphologyEx(destination, destination, MorphTypes.Close, kernel, iterations: closingIterations);
                }
            }

            if (openingKernelSize > 1 && openingIterations > 0)
            {
                using (var kernel = CreateMorphologyKernel(openingKernelSize))
                {
                    Cv2.MorphologyEx(destination, destination, MorphTypes.Open, kernel, iterations: openingIterations);
                }
            }
        }

        private static Mat CreateMorphologyKernel(int kernelSize)
        {
            var normalizedSize = Math.Max(1, kernelSize);
            if (normalizedSize % 2 == 0)
                normalizedSize++;

            return Cv2.GetStructuringElement(
                MorphShapes.Ellipse,
                new OpenCvSharp.Size(normalizedSize, normalizedSize));
        }

        private static double ComputeAverageProbabilityPercent(Mat responseMap8u, Mat labels, int targetLabel)
        {
            long sum = 0;
            var count = 0;

            for (var y = 0; y < labels.Rows; y++)
            {
                for (var x = 0; x < labels.Cols; x++)
                {
                    var label = labels.At<int>(y, x);
                    if (label != targetLabel)
                        continue;

                    var value = responseMap8u.At<byte>(y, x);
                    sum += value;
                    count++;
                }
            }

            if (count == 0)
                return 0.0;

            var average = (double)sum / count;
            return average * 100.0 / 255.0;
        }

        public static Task<List<ObjectLocation>> LoadObjectDetectionResultsAsync(ImageItem item, IProjectPresenter projectPresenter, int thresholdPercent)
        {
            if (item == null) throw new ArgumentNullException(nameof(item));
            if (projectPresenter == null) throw new ArgumentNullException(nameof(projectPresenter));

            return Task.Run(() =>
            {
                var newLocations = new List<ObjectLocation>();
                var threshold = ConvertThresholdPercentToByte(thresholdPercent);

                if (string.IsNullOrWhiteSpace(projectPresenter.ProjectPath))
                    return newLocations;

                var paths = projectPresenter.Paths;

                foreach (var feature in projectPresenter.Project.Features)
                {
                    var classId = feature.ClassId;
                    if (classId == 0)
                        continue;

                    var featureSubfolder = Path.Combine(paths.MasksHeatmaps, feature.Name + "_" + classId);

                    if (!Directory.Exists(featureSubfolder))
                        continue;

                    var rawHeatPng = Path.Combine(featureSubfolder, item.Guid + paths.ImagesExt);
                    if (!File.Exists(rawHeatPng))
                        continue;

                    using (var responseMap = Cv2.ImRead(rawHeatPng, ImreadModes.Grayscale))
                    {
                        if (responseMap.Empty())
                            continue;

                        var detections = ExtractCenters(
                            responseMap,
                            classId,
                            item.Roi,
                            projectPresenter.Project.Settings.PreprocessingSettings.DownSample,
                            threshold,
                            objectDiameter: projectPresenter.Project.Settings.ObjectsDiameter);

                        newLocations.AddRange(detections);
                    }
                }

                return newLocations;
            });
        }

        private static byte ConvertThresholdPercentToByte(int thresholdPercent)
        {
            if (thresholdPercent <= 0)
                return 0;

            if (thresholdPercent >= 100)
                return 254;

            var minimumIncludedValue = (int)Math.Ceiling(thresholdPercent * 255.0 / 100.0);
            return (byte)(minimumIncludedValue - 1);
        }
    }
}
