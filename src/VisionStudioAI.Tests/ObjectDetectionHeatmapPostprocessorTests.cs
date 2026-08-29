using VisionStudioAI.ML.Inference;
using OpenCvSharp;
using System.Drawing;
using Xunit;

namespace VisionStudioAI.Tests
{
    public class ObjectDetectionHeatmapPostprocessorTests
    {
        [Fact]
        public void ExtractCenters_MergesFragmentedBlobWithMorphology()
        {
            using var responseMap = Mat.Zeros(32, 32, MatType.CV_8UC1).ToMat();
            responseMap[new Rect(6, 10, 6, 8)].SetTo(255);
            responseMap[new Rect(14, 10, 6, 8)].SetTo(255);

            var detections = ObjectDetectionHeatmapPostprocessor.ExtractCenters(
                responseMap,
                classId: 3,
                roi: Rectangle.Empty,
                downSample: 0,
                threshold: 128);

            Assert.Single(detections);
            Assert.Equal(3, detections[0].ClassId);
            Assert.InRange(detections[0].Center.X, 11, 14);
            Assert.InRange(detections[0].Center.Y, 12, 15);
        }

        [Fact]
        public void ExtractCenters_KeepsSeparateBlobsSeparateAfterMorphology()
        {
            using var responseMap = Mat.Zeros(40, 40, MatType.CV_8UC1).ToMat();
            responseMap[new Rect(5, 10, 6, 8)].SetTo(255);
            responseMap[new Rect(28, 10, 6, 8)].SetTo(255);

            var detections = ObjectDetectionHeatmapPostprocessor.ExtractCenters(
                responseMap,
                classId: 3,
                roi: Rectangle.Empty,
                downSample: 0,
                threshold: 128);

            Assert.Equal(2, detections.Count);
        }

        [Fact]
        public void ExtractCenters_RemovesBlobsTooSmallForConfiguredObjectDiameter()
        {
            using var responseMap = Mat.Zeros(32, 32, MatType.CV_8UC1).ToMat();
            responseMap[new Rect(12, 12, 3, 3)].SetTo(255);

            var detections = ObjectDetectionHeatmapPostprocessor.ExtractCenters(
                responseMap,
                classId: 3,
                roi: Rectangle.Empty,
                downSample: 0,
                threshold: 128,
                objectDiameter: 10);

            Assert.Empty(detections);
        }

        [Fact]
        public void ExtractCenters_KeepsBlobWithinConfiguredObjectDiameterMargin()
        {
            using var responseMap = Mat.Zeros(32, 32, MatType.CV_8UC1).ToMat();
            responseMap[new Rect(10, 10, 6, 6)].SetTo(255);

            var detections = ObjectDetectionHeatmapPostprocessor.ExtractCenters(
                responseMap,
                classId: 3,
                roi: Rectangle.Empty,
                downSample: 0,
                threshold: 128,
                objectDiameter: 10);

            Assert.Single(detections);
        }
    }
}
