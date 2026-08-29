using OpenCvSharp;
using System;

namespace VisionStudioAI.ML.Utils.ImageProcessing
{
    internal static class ImageComposition
    {

        // Images need to be sorted !!!
        public static Mat MergeImages(Mat[] images, int nRowImages, int nColumnImages)
        {
            var expected = nRowImages * nColumnImages;
            if (images.Length != expected)
                throw new ArgumentException("Amount of images does not match grid size.");

            // Validate all images have the same size & type
            var w = images[0].Cols;
            var h = images[0].Rows;
            var type = images[0].Type();

            for (var k = 1; k < images.Length; k++)
            {
                if (images[k].Cols != w || images[k].Rows != h || images[k].Type() != type)
                    throw new ArgumentException("All images must have the same dimensions and type.");
            }

            // Build rows and concatenate horizontally
            var rows = new Mat[nRowImages];
            Mat result = null;

            try
            {
                var index = 0;

                for (var r = 0; r < nRowImages; r++)
                {
                    var rowImgs = new Mat[nColumnImages];

                    for (var c = 0; c < nColumnImages; c++)
                    {
                        rowImgs[c] = images[index];
                        index++;
                    }

                    rows[r] = new Mat();
                    Cv2.HConcat(rowImgs, rows[r]);
                }

                // Concatenate vertically to form final grid
                result = new Mat();
                Cv2.VConcat(rows, result);

                return result;
            }
            catch
            {
                result?.Dispose();
                throw;
            }
            finally
            {
                foreach (var row in rows)
                {
                    row?.Dispose();
                }
            }
        }

    }
}
