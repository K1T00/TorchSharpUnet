using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;

namespace Utils
{
	public class HelperFunctions
	{

        /// <summary>
        /// Convert array to a C# matrix
        /// </summary>
        /// <param name="flat">Array</param>
        /// <param name="m">Rows</param>
        /// <param name="n">Columns</param>
        /// <returns>The C# 2dim Matrix</returns>
        /// <exception cref="ArgumentException"></exception>
        public static double[,] ConvertToMatrix(double[] flat, int m, int n)
		{
			if (flat.Length != m * n)
			{
				throw new ArgumentException("Invalid length");
			}
			var ret = new double[m, n];

			Buffer.BlockCopy(flat, 0, ret, 0, flat.Length * sizeof(double));
			return ret;
		}

        /// <summary>
        /// Split a list
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="locations"></param>
        /// <param name="nSize"></param>
        /// <returns></returns>
		public static IEnumerable<List<T>> SplitList<T>(List<T> locations, int nSize)
		{
			for (var i = 0; i < locations.Count; i += nSize)
			{
				yield return locations.GetRange(i, Math.Min(nSize, locations.Count - i));
			}
		}

		/// <summary>
		/// Sort images files depending on their file names
		/// </summary>
		/// <param name="files">Files in the image folder</param>
		/// <returns>Tuple of filepath, grabNr </returns>
		public static List<Tuple<string, int>> SortImageFiles(string[] files)
        {
            var fileNames = new List<Tuple<string, int>>();

            foreach (var datasetImage in files)
            {
	            var sStart = datasetImage.LastIndexOf("\\", StringComparison.CurrentCulture) + 1;
	            var sTo = datasetImage.LastIndexOf(".", StringComparison.CurrentCulture);

	            fileNames.Add(new Tuple<string, int>(datasetImage!, Convert.ToInt16(datasetImage.Substring(sStart, sTo - sStart))));
            }
            return fileNames.OrderBy(t => t.Item2).ToList();
        }

		/// <summary>
		/// Sort masks files depending on their file names
		/// </summary>
		/// <param name="files"></param>
		/// <param name="moreThanOneFeature"></param>
		/// <returns></returns>
		public static List<Tuple<string, int, int>> SortMaskFiles(string[] files, bool moreThanOneFeature)
		{
			var fileNames = new List<Tuple<string, int, int>>();

            foreach (var datasetImage in files)
			{
				var sStart = datasetImage.LastIndexOf("\\", StringComparison.CurrentCulture) + 1;
				var sTo = datasetImage.LastIndexOf(".", StringComparison.CurrentCulture);

				if (moreThanOneFeature)
				{
					var sSplit = datasetImage.Substring(sStart, sTo - sStart).Split("_");
					fileNames.Add(new Tuple<string, int, int>(datasetImage, Convert.ToInt16(sSplit[0]) , Convert.ToInt16(sSplit[1])));
				}
				else
				{
					fileNames.Add(new Tuple<string, int, int>(datasetImage, Convert.ToInt16(datasetImage.Substring(sStart, sTo - sStart)), 1));
				}
			}
			return fileNames.OrderBy(t => t.Item2).ToList();
		}

		public static void EmptyFolder(DirectoryInfo directory)
		{
			foreach (var file in directory.GetFiles()) file.Delete();
			//foreach (System.IO.DirectoryInfo subDirectory in directory.GetDirectories()) subDirectory.Delete(true);
		}

		/// <summary>
		/// Hopefully images are sorted ...
		/// </summary>
		/// <param name="images">All the images, that should be merged</param>
		/// <param name="nRowImages">Amount of images in the columns direction</param>
		/// <param name="nColumnImages">Amount of images in the rows direction</param>
		/// <returns>Merged image</returns>
		/// <exception cref="ArgumentException"></exception>
		public static Mat MergeImages(Mat[] images, int nRowImages, int nColumnImages)
        {
            if (images.Length % (nRowImages * nColumnImages) != 0)
            {
                throw new ArgumentException("Amount of images do not fit");
            }

            var concatImages = new Mat();
            var concatImagesC = new Mat[nColumnImages];
            
			var i = 0;
            for (var c = 0; c < nColumnImages; c++)
            {
                var imagesR = new Mat[nRowImages];

                for (var r = 0; r < nRowImages; r++)
                {
                    imagesR[r] = images[i];
                    i++;
                }
                concatImagesC[c] = new Mat();
                Cv2.HConcat(imagesR, concatImagesC[c]);
            }
            Cv2.VConcat(concatImagesC, concatImages);

            return concatImages;
        }
      
        /// <summary>
        /// Check whether there is a feature in the image (mask)
        /// </summary>
        /// <param name="img"></param>
        /// <returns></returns>
        public static bool IsBlobInImage(Mat image)
        {
	        var imgWithBoarder = new Mat();
			//Cv2.BitwiseNot(img, img);

			// Touching boarder blobs are not detected, therefore draw frame
			Cv2.CopyMakeBorder(image, imgWithBoarder, 10, 10, 10, 10, BorderTypes.Constant, Scalar.Black);

            var blobParams = new SimpleBlobDetector.Params
            {
				FilterByColor = false,
				FilterByArea = false,
				FilterByCircularity = false,
				FilterByConvexity = false,
				FilterByInertia = false
			};

            var blobDetector = SimpleBlobDetector.Create(blobParams);

            return blobDetector.Detect(imgWithBoarder).Length > 0;
        }

        /// <summary>
        /// Get roi from image and apply padding if it falls outside the image
        /// </summary>
        /// <param name="input"></param>
        /// <param name="roiTopLeftX"></param>
        /// <param name="roiTopLeftY"></param>
        /// <param name="roiWidth"></param>
        /// <param name="roiHeight"></param>
        /// <param name="paddingColor"></param>
        /// <returns></returns>
        public static Mat GetPaddedRoi(Mat input, int roiTopLeftX, int roiTopLeftY, int roiWidth, int roiHeight, Scalar paddingColor)
        {
            Mat output;
            var bottomRightX = roiTopLeftX + roiWidth;
            var bottomRightY = roiTopLeftY + roiHeight;

            // Border padding will be required
            if (roiTopLeftX < 0 || roiTopLeftY < 0 || bottomRightX > input.Cols || bottomRightY > input.Rows)
            {
                var paddingBorderLeft = 0;
                var paddingBorderRight = 0;
                var paddingBoarderTop = 0;
                var paddingBoarderBottom = 0;
                var newImgTopLeftX = 0;
                var newImgTopLeftY = 0;

                // Check where padding is needed and adjust new image roi values
                if (roiTopLeftX < 0)
                {
                    paddingBorderLeft = roiTopLeftX * -1;
                }
                if (roiTopLeftX > 0)
                {
                    newImgTopLeftX += roiTopLeftX;
                }
                if (roiTopLeftY < 0)
                {
                    paddingBoarderTop = -1 * roiTopLeftY;
                }
                if (roiTopLeftY > 0)
                {
                    newImgTopLeftY += roiTopLeftY;
                }
                if (bottomRightX > input.Cols)
                {
                    paddingBorderRight = bottomRightX - input.Cols;
                }
                if (bottomRightY > input.Rows)
                {
                    paddingBoarderBottom = bottomRightY - input.Rows;
                }
                var padded = new Mat();

                Cv2.CopyMakeBorder(input, padded, paddingBoarderTop, paddingBoarderBottom, paddingBorderLeft, paddingBorderRight, BorderTypes.Constant, paddingColor);

                output = padded.SubMat(newImgTopLeftY, roiHeight + newImgTopLeftY, newImgTopLeftX, newImgTopLeftX + roiWidth);

                padded.Dispose();
            }
			else // No border padding required
			{
                output = input.SubMat(roiTopLeftY, roiHeight + roiTopLeftY,roiTopLeftX,roiWidth + roiTopLeftX);
            }
            return output;
        }

        /// <summary>
        /// Split images into smaller ones (optionally with boarder padding) and save the results
        /// </summary>
        /// <param name="imagesPaths"></param>
        /// <param name="masksPaths"></param>
        /// <param name="saveImagesPath"></param>
        /// <param name="saveMasksPath"></param>
        /// <param name="roiSize"></param>
        /// <param name="downSampling"></param>
        /// <param name="filterByBlobs"></param>
        /// <param name="convertToGreyscale"></param>
        /// <param name="normalizeMasks"></param>
        /// <param name="withBoarderPadding"></param>
        /// <returns></returns>
        public static int SplitImages(
            List<Tuple<string, int>> imagesPaths,
            List<Tuple<string, int, int>> masksPaths,
            string saveImagesPath,
            string saveMasksPath,
            int roiSize,
            int downSampling,
            bool filterByBlobs,
            bool convertToGreyscale,
            bool withBoarderPadding)
        {
            var withMasks = masksPaths.Count > 0;
            var imRoi = 1;
            var img = new Mat();
            var msk = new Mat();
            var imgSub = new Mat();
            var mskSub = new Mat();
            var normMsk = new Mat();

			for (var i = 0; i < imagesPaths.Count; i++)
            {
                // Load image
                if (convertToGreyscale)
                {
                    Cv2.ImRead(imagesPaths[i].Item1, ImreadModes.Grayscale).CopyTo(img);
                }
                else
                {
                    Cv2.ImRead(imagesPaths[i].Item1, ImreadModes.Unchanged).CopyTo(img);
                }
                // Load mask
                if (withMasks)
                {
	                Cv2.ImRead(masksPaths[i].Item1, ImreadModes.Unchanged).CopyTo(msk);
                    msk.ConvertTo(msk, MatType.CV_8UC1);
					normMsk = new Mat(msk.Rows, msk.Cols, MatType.CV_8UC1);
                    Cv2.ExtractChannel(msk, normMsk, 0);
					Cv2.Normalize(normMsk, normMsk, 0, 255, NormTypes.MinMax);
                }
                var imageDs = DownSampleImage(img, downSampling);
                var maskDs = DownSampleImage(normMsk, downSampling);

				var amtImages = GetAmtImages(imageDs.Height, imageDs.Width, roiSize, withBoarderPadding);

                for (var pY = 0; pY < amtImages.Item1; pY++)
                {
                    for (var pX = 0; pX < amtImages.Item2; pX++)
                    {
                        imgSub = GetPaddedRoi(imageDs, pX * roiSize, pY * roiSize, roiSize, roiSize, Scalar.Black);
                        if(withMasks) mskSub = GetPaddedRoi(maskDs, pX * roiSize, pY * roiSize, roiSize, roiSize, Scalar.Black);

                        if (filterByBlobs)
                        {
                            if (IsBlobInImage(mskSub))
                            {
                                Cv2.ImWrite(saveImagesPath + imRoi + ".bmp", imgSub);
                                if (withMasks) Cv2.ImWrite(saveMasksPath + imRoi + ".bmp", mskSub);
                                imRoi++;
                            }
                        }
                        else
                        {
                            Cv2.ImWrite(saveImagesPath + imRoi + ".bmp", imgSub);
                            if (withMasks) Cv2.ImWrite(saveMasksPath + imRoi + ".bmp", mskSub);
                            imRoi++;
                        }
                    }
                }
            }
            img.Dispose();
            msk.Dispose();
            imgSub.Dispose();
            normMsk.Dispose();
            mskSub.Dispose();
            GC.Collect();

            return imRoi;
        }

        /// <summary>
        /// Convert the msk image into a heatmap and superimpose it on to the img
        /// </summary>
        /// <param name="img"></param>
        /// <param name="msk"></param>
        /// <param name="threshold"></param>
        /// <returns></returns>
		public static Mat ImageToHeatmap(Mat img, Mat msk, int threshold)
        {
            var image = new Mat();
            var heatmap = new Mat();
            var superImposedImage = new Mat();

            if (img.Type() == MatType.CV_8UC1)
            {
				Cv2.CvtColor(img, image, ColorConversionCodes.GRAY2RGB);
			}
            else
            {
	            Cv2.CopyTo(img, image);
            }
            Cv2.Threshold(msk, msk, threshold, 255, ThresholdTypes.Tozero);
	        Cv2.ApplyColorMap(msk, heatmap, ColormapTypes.Turbo);
	        Cv2.AddWeighted(heatmap, 1, image, 0.5, 0, superImposedImage);
            Cv2.ApplyColorMap(msk, heatmap, ColormapTypes.Turbo);

            return superImposedImage;
        }

		/// <summary>
		/// Count the amount of images per row and column that fit into the given size
		/// </summary>
		/// <param name="imageHeight"></param>
		/// <param name="imageWidth"></param>
		/// <param name="roiSize"></param>
		/// <param name="withBoarderPadding"></param>
		/// <returns> Item1 := images per row; Item2 := images per column</returns>
		public static (int, int) GetAmtImages(int imageHeight, int imageWidth, int roiSize, bool withBoarderPadding)
        {
	        decimal rowImages = 1;
	        decimal columnImages = 1;

	        if (withBoarderPadding)
	        {
		        rowImages = decimal.Round(Convert.ToDecimal((double)imageHeight / roiSize), 0, MidpointRounding.ToPositiveInfinity);
		        columnImages = decimal.Round(Convert.ToDecimal((double)imageWidth / roiSize), 0, MidpointRounding.ToPositiveInfinity);
	        }
	        else
	        {
		        rowImages = decimal.Round(Convert.ToDecimal((double)imageHeight / roiSize), 0, MidpointRounding.ToZero);
		        columnImages = decimal.Round(Convert.ToDecimal((double)imageWidth / roiSize), 0, MidpointRounding.ToZero);
	        }
	        return ((int)rowImages, (int)columnImages);
        }

        /// <summary>
        /// Split image into smaller ones
        /// </summary>
        /// <param name="image"></param>
        /// <param name="roiSize"></param>
        /// <param name="withBoarderPadding"></param>
        /// <param name="nImagesRow"></param>
        /// <param name="nImagesColumn"></param>
        /// <returns></returns>
		public static Mat[] SplitImage(Mat image, int roiSize, bool withBoarderPadding, int nImagesRow, int nImagesColumn)
		{
			var imagesSplit = new Mat[nImagesRow * nImagesColumn];

			var im = 0;
			for (var pY = 0; pY < nImagesRow; pY++)
			{
				for (var pX = 0; pX < nImagesColumn; pX++)
				{
					var imgSub = GetPaddedRoi(image, pX * roiSize, pY * roiSize, roiSize, roiSize, OpenCvSharp.Scalar.Black);

					imagesSplit[im] = imgSub;
					im++;
				}
			}
			return imagesSplit;
		}
        /// <summary>
        /// Apply down sampling to an image
        /// </summary>
        /// <param name="image"></param>
        /// <param name="nDownSampling"></param>
        /// <returns></returns>
		public static Mat DownSampleImage(Mat image, int nDownSampling)
		{
			var imageDs = image.Clone();

			for (var ds = 0; ds < nDownSampling; ds++)
			{
				imageDs.Resize(new OpenCvSharp.Size(0, 0), 0.5, 0.5, InterpolationFlags.Nearest).CopyTo(imageDs);
			}
			
			return imageDs;
		}

	}
}
