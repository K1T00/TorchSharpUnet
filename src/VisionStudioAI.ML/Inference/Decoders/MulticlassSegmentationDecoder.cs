using VisionStudioAI.Core.Models;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using static VisionStudioAI.ML.Utils.DatasetStatistics;
using static VisionStudioAI.ML.Utils.TensorProcessing.TensorConversion;
using static TorchSharp.torch;

namespace VisionStudioAI.ML.Inference.Decoders
{
    /// <summary>
    /// Multiclass segmentation decoder.
    ///
    /// Responsibilities:
    /// - Decode logits into per-class probability tiles (softmax)
    /// - Skip background channel
    /// - Compute multiclass metrics from reconstructed probability maps
    ///
    /// Visualization and saving are handled elsewhere.
    /// </summary>
    public sealed class MulticlassSegmentationDecoder : ISegmentationDecoder
    {
        private readonly int numClasses; // includes background (class 0)

        public MulticlassSegmentationDecoder(int numClasses)
        {
            if (numClasses < 2) // Binary case should use BinarySegmentationDecoder
                throw new ArgumentOutOfRangeException(nameof(numClasses));

            this.numClasses = numClasses;
        }

        /// <summary>
        /// Decodes logits into probability tiles (working space).
        /// Returns one Mat[] per feature class (background skipped).
        /// </summary>
        public Dictionary<int, Mat[]> Decode(Tensor logits)
        {
            using (var scope = NewDisposeScope())
            {
                // logits: [N, C, H, W]
                using (var probs = softmax(logits, dim: 1).cpu()) 
                {
                    var result = new Dictionary<int, Mat[]>();

                    // Skip background class 0
                    for (var classId = 1; classId < numClasses; classId++)
                    {
                        // Select probability map for classId ? [N,H,W] with unsqueeze for [N,1,H,W]
                        using (var classProb = probs.select(1, classId).unsqueeze(1))
                        {
                            var predSlices = TensorTo2DArray(classProb);
                            result.Add(classId, SlicedImageTensorToImage(predSlices));
                        }
                    }

                    return result;
                }
            }
        }

        /// <summary>
        /// Computes multiclass segmentation metrics from full-size probability maps.
        /// </summary>
        public Dictionary<int, SegmentationStats> ComputeMetrics(Dictionary<int, Mat> fullMaskPredictions, Mat groundTruth)
        {
            var result = new Dictionary<int, SegmentationStats>();
            var foregroundProbabilities = new List<Mat>();

            for (var classId = 1; classId < numClasses; classId++)
            {
                Mat probabilityMap;
                if (!fullMaskPredictions.TryGetValue(classId, out probabilityMap))
                    throw new InvalidOperationException("Missing probability map for class " + classId + ".");

                foregroundProbabilities.Add(probabilityMap);
            }

            using (var predictedClassMap = BuildClassMapUnsafe(foregroundProbabilities))
            {
                foreach (var classId in fullMaskPredictions.Keys.OrderBy(id => id))
                {
                    result[classId] = ComputeClassMetrics(predictedClassMap, groundTruth, classId);
                }
            }

            return result;
        }

        public void Dispose() { }
    }
}
