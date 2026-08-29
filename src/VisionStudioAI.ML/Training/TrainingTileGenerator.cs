using VisionStudioAI.ML.Geometry;
using VisionStudioAI.ML.Processing;
using VisionStudioAI.Core.Services;
using OpenCvSharp;
using System;
using System.IO;
using System.Threading;
using static VisionStudioAI.ML.Utils.ImageProcessing.ImageUtils;

namespace VisionStudioAI.ML.Training
{
    public static class TrainingTileGenerator
    {
        public static void GenerateTrainingTiles(IProjectPresenter project, IProgress<int> progress, CancellationToken ct)
        {
            Mat[] imgTiles = null;
            Mat[] maskTiles = null;
            var paths = project.Paths;

            try
            {
                var processed = 0;
                var total = project.Project.Images.Count;

                foreach (var img in project.Project.Images)
                {
                    ct.ThrowIfCancellationRequested();

                    var imagePath = Path.Combine(paths.Images, img.Guid + paths.ImagesExt);
                    var maskPath = Path.Combine(paths.Masks, img.Guid + paths.ImagesExt);

                    var space = new SegmentationImageSpace(
                        new Size(img.ImageSize.Width, img.ImageSize.Height),
                        new Rect(img.Roi.X, img.Roi.Y, img.Roi.Width, img.Roi.Height),
                        project.Project.Settings.PreprocessingSettings.SliceSize,
                        project.Project.Settings.PreprocessingSettings.DownSample,
                        project.Project.Settings.PreprocessingSettings.BorderPadding);

                    var preProc = new SegmentationPreprocessor(space);
                    
                    using (var image = LoadInputImage(imagePath, project.Project.Settings.PreprocessingSettings.TrainAsGreyscale))
                    using (var maskGt = Cv2.ImRead(maskPath, ImreadModes.Grayscale))
                    {
                        imgTiles = preProc.ProcessImage(image);
                        maskTiles = preProc.ProcessImage(maskGt);

                        for (var i = 0; i < imgTiles.Length; i++)
                        {
                            if (project.Project.Settings.PreprocessingSettings.TrainOnlyFeatures)
                            {
                                if (!TileContainsForeground(maskTiles[i]))
                                    continue;
                            }
                            var baseName = $"{img.Guid}_{i:D4}";
                            Cv2.ImWrite(Path.Combine(paths.SlicedImages, baseName + paths.ImagesExt), imgTiles[i]);
                            Cv2.ImWrite(Path.Combine(paths.SlicedMasks, baseName + paths.ImagesExt), maskTiles[i]);
                        }
                    }
                    DisposeTiles(imgTiles);
                    DisposeTiles(maskTiles);

                    processed++;
                    progress?.Report((int)(100.0 * processed / total));
                }
            }
            finally
            {
                DisposeTiles(imgTiles);
                DisposeTiles(maskTiles);
            }
        }

        /// <summary>
        /// Generates image tiles for image-level anomaly training. Unlike
        /// segmentation preprocessing, this path does not load, filter, or
        /// write mask tiles because anomaly labels belong to the source image.
        /// </summary>
        public static void GenerateAnomalyTrainingTiles(
            IProjectPresenter project,
            IProgress<int> progress,
            CancellationToken ct)
        {
            var paths = project.Paths;
            var processed = 0;
            var total = project.Project.Images.Count;

            foreach (var imageItem in project.Project.Images)
            {
                ct.ThrowIfCancellationRequested();
                Mat[] imageTiles = null;

                try
                {
                    var imagePath = Path.Combine(
                        paths.Images,
                        imageItem.Guid + paths.ImagesExt);
                    var preprocessing = project.Project.Settings.PreprocessingSettings;
                    var space = new SegmentationImageSpace(
                        new Size(imageItem.ImageSize.Width, imageItem.ImageSize.Height),
                        new Rect(
                            imageItem.Roi.X,
                            imageItem.Roi.Y,
                            imageItem.Roi.Width,
                            imageItem.Roi.Height),
                        preprocessing.SliceSize,
                        preprocessing.DownSample,
                        preprocessing.BorderPadding);
                    var preprocessor = new SegmentationPreprocessor(space);

                    using (var image = LoadInputImage(imagePath, preprocessing.TrainAsGreyscale))
                    {
                        imageTiles = preprocessor.ProcessImage(image);
                        for (var index = 0; index < imageTiles.Length; index++)
                        {
                            ct.ThrowIfCancellationRequested();
                            var baseName = $"{imageItem.Guid}_{index:D4}";
                            Cv2.ImWrite(
                                Path.Combine(paths.SlicedImages, baseName + paths.ImagesExt),
                                imageTiles[index]);
                        }
                    }
                }
                finally
                {
                    DisposeTiles(imageTiles);
                }

                processed++;
                progress?.Report((int)(100.0 * processed / total));
            }
        }

        private static Mat LoadInputImage(string path, bool loadAsGreyscale)
        {
            return loadAsGreyscale
                ? Cv2.ImRead(path, ImreadModes.Grayscale)
                : Cv2.ImRead(path, ImreadModes.Color);
        }

        private static bool TileContainsForeground(Mat maskTile)
        {
            // Check for any non-zero pixel
            return Cv2.CountNonZero(maskTile) > 0;
        }
    }
}
