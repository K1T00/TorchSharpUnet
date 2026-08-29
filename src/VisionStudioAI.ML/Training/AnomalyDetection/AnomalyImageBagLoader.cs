using System;
using System.Collections.Generic;
using VisionStudioAI.Core.Models;
using OpenCvSharp;
using static TorchSharp.torch;
using static VisionStudioAI.ML.Utils.TensorProcessing.TensorConversion;

namespace VisionStudioAI.ML.Training.AnomalyDetection
{
    internal static class AnomalyImageBagLoader
    {
        public static Tensor Load(
            AnomalyImageSample sample,
            bool greyscale,
            NormalizationSettings normalization,
            Device device,
            ScalarType precision)
        {
            if (sample == null) throw new ArgumentNullException(nameof(sample));
            return Load(new[] { sample }, greyscale, normalization, device, precision);
        }

        public static Tensor Load(
            IReadOnlyList<AnomalyImageSample> samples,
            bool greyscale,
            NormalizationSettings normalization,
            Device device,
            ScalarType precision)
        {
            if (samples == null) throw new ArgumentNullException(nameof(samples));
            if (samples.Count == 0)
                throw new InvalidOperationException("An anomaly image batch must contain at least one image.");

            using (var scope = NewDisposeScope())
            {
                var tileCount = 0;
                foreach (var sample in samples)
                {
                    if (sample?.TilePaths == null || sample.TilePaths.Count == 0)
                        throw new InvalidOperationException("An anomaly image sample must contain at least one tile.");
                    tileCount += sample.TilePaths.Count;
                }

                var tiles = new List<Tensor>(tileCount);
                foreach (var sample in samples)
                {
                    foreach (var path in sample.TilePaths)
                    {
                        using (var image = Cv2.ImRead(path, greyscale ? ImreadModes.Grayscale : ImreadModes.Color))
                        {
                            if (image.Empty())
                                throw new InvalidOperationException("Could not load anomaly training tile: " + path);

                            var tensor = greyscale
                                ? GreyMatToTensor(image, device, precision)
                                : RgbMatToTensor(image, device, precision);
                            tiles.Add(NormalizeImageTensor(tensor, greyscale, normalization).unsqueeze(0));
                        }
                    }
                }

                return cat(tiles.ToArray(), 0).MoveToOuterDisposeScope();
            }
        }

        public static Tensor LoadTilePaths(
            IReadOnlyList<string> tilePaths,
            bool greyscale,
            NormalizationSettings normalization,
            Device device,
            ScalarType precision)
        {
            if (tilePaths == null) throw new ArgumentNullException(nameof(tilePaths));
            if (tilePaths.Count == 0)
                throw new InvalidOperationException("An anomaly tile batch must not be empty.");

            using (var scope = NewDisposeScope())
            {
                var tiles = new List<Tensor>(tilePaths.Count);
                foreach (var path in tilePaths)
                {
                    using (var image = Cv2.ImRead(
                        path,
                        greyscale ? ImreadModes.Grayscale : ImreadModes.Color))
                    {
                        if (image.Empty())
                            throw new InvalidOperationException("Could not load anomaly training tile: " + path);

                        var imageTensor = greyscale
                            ? GreyMatToTensor(image, device, precision)
                            : RgbMatToTensor(image, device, precision);
                        tiles.Add(NormalizeImageTensor(
                            imageTensor,
                            greyscale,
                            normalization).unsqueeze(0));
                    }
                }

                return cat(tiles.ToArray(), 0).MoveToOuterDisposeScope();
            }
        }
    }
}
