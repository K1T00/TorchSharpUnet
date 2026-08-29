using System;
using OpenCvSharp;
using VisionStudioAI.ML.Processing;
using static TorchSharp.torch;
using static TorchSharp.torch.nn.functional;
using static VisionStudioAI.ML.Utils.ImageProcessing.ImageConversion;

namespace VisionStudioAI.ML.Inference.AnomalyDetection
{
    public static class AnomalyHeatmapStitcher
    {
        /// <summary>
        /// Upsamples calibrated tile maps and reconstructs them into the original image space.
        /// </summary>
        public static Mat Stitch(
            Tensor tileHeatmaps,
            int tileHeight,
            int tileWidth,
            SegmentationPostprocessor postprocessor)
        {
            if (ReferenceEquals(tileHeatmaps, null)) throw new ArgumentNullException(nameof(tileHeatmaps));
            if (postprocessor == null) throw new ArgumentNullException(nameof(postprocessor));
            if (tileHeatmaps.Dimensions != 4 || tileHeatmaps.shape[1] != 1)
                throw new ArgumentException("Expected tile heatmaps [tiles,1,h,w].", nameof(tileHeatmaps));

            using (var scope = NewDisposeScope())
            {
                var upsampled = interpolate(
                    tileHeatmaps,
                    new long[] { tileHeight, tileWidth },
                    mode: InterpolationMode.Bilinear,
                    align_corners: false).cpu();
                var mats = new Mat[upsampled.shape[0]];

                try
                {
                    for (var i = 0; i < mats.Length; i++)
                    {
                        mats[i] = TensorToGreyImage(upsampled[i, 0]);
                    }
                    return postprocessor.ProcessImageTiles(mats);
                }
                finally
                {
                    foreach (var mat in mats) mat?.Dispose();
                }
            }
        }
    }
}
