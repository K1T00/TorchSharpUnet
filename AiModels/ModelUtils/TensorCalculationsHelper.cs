using System;
using System.IO;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;


namespace AiModels.ModelUtils
{

	public class TensorCalculationsHelper
	{
		/// <summary>
		/// Converts an OpenCV.Mat of type 8UC1 to tensor (and also normalizing the values)
		/// </summary>
		/// <param name="image"></param>
		/// <param name="device"></param>
		/// <param name="usePrecision"></param>
		/// <returns></returns>
		/// <exception cref="ArgumentException"></exception>
		public static Tensor GreyImageToTensor(Mat image, Device device, ScalarType usePrecision)
		{
			if (image.Type() != MatType.CV_8UC1)
			{
				throw new ArgumentException("Invalid image type");
			}
			Tensor? tens = null;

			var matVec = new Mat<Vec3b>(image);
			var indexer = matVec.GetIndexer();
			var imgCvChan = new float[image.Height * image.Width];
			
			if (usePrecision == bfloat16)
			{
				var imgCvChanNorm = new Half[image.Height * image.Width];

				for (var y = 0; y < image.Height; y++)
				{
					for (var x = 0; x < image.Width; x++)
					{
						imgCvChan[y * image.Width + x] = indexer[y, x].Item0;
					}
				}

				var mean = imgCvChan.Average();
				var stdDevR = Math.Sqrt(imgCvChan.Average(r => Math.Pow(r - mean, 2)));

				for (var i = 0; i < imgCvChan.Length; i++)
				{
					imgCvChanNorm[i] = (Half)Convert.ToSingle((imgCvChan[i] - mean) / stdDevR);
				}

				tens = cat(new[]
				{
					tensor(imgCvChanNorm, new long[] { 1, image.Width, image.Height }).to_type(bfloat16)
				}, 0);
			}
			if (usePrecision == float32)
			{
				var imgCvChanNorm = new float[image.Height * image.Width];

				for (var y = 0; y < image.Height; y++)
				{
					for (var x = 0; x < image.Width; x++)
					{
						imgCvChan[y * image.Width + x] = indexer[y, x].Item0;
					}
				}

				var mean = imgCvChan.Average();
				var stdDevR = Math.Sqrt(imgCvChan.Average(r => Math.Pow(r - mean, 2)));

				for (var i = 0; i < imgCvChan.Length; i++)
				{
					imgCvChanNorm[i] = Convert.ToSingle((imgCvChan[i] - mean) / stdDevR);
				}

				tens = cat(new[]
				{
					tensor(imgCvChanNorm, new long[] { 1, image.Width, image.Height }).to_type(float32)
				}, 0);
			}
			return tens!.to(device);
		}

		/// <summary>
		/// Converts an OpenCV.Mat of type 8UC3 to tensor (and also normalizing the values)
		/// </summary>
		/// <param name="image"></param>
		/// <param name="device"></param>
		/// <param name="usePrecision"></param>
		/// <returns></returns>
		/// <exception cref="ArgumentException"></exception>
		public static Tensor RgbImageToTensor(Mat image, Device device, ScalarType usePrecision)
		{
			if (image.Type() != MatType.CV_8UC3)
			{
				throw new ArgumentException("Invalid image type");
			}

			Tensor? tens = null;

			var matVec = new Mat<Vec3b>(image);
			var indexer = matVec.GetIndexer();
			var imgCvChanR = new float[image.Height * image.Width];
			var imgCvChanG = new float[image.Height * image.Width];
			var imgCvChanB = new float[image.Height * image.Width];

			if (usePrecision == bfloat16)
			{
				var imgCvChanRNorm = new Half[image.Height * image.Width];
				var imgCvChanGNorm = new Half[image.Height * image.Width];
				var imgCvChanBNorm = new Half[image.Height * image.Width];

				for (var y = 0; y < image.Height; y++)
				{
					for (var x = 0; x < image.Width; x++)
					{
						imgCvChanR[y * image.Width + x] = indexer[y, x].Item2;
						imgCvChanG[y * image.Width + x] = indexer[y, x].Item1;
						imgCvChanB[y * image.Width + x] = indexer[y, x].Item0;
					}
				}

				var meanR = imgCvChanR.Average();
				var meanG = imgCvChanG.Average();
				var meanB = imgCvChanB.Average();
				var stdDevR = Math.Sqrt(imgCvChanR.Average(r => Math.Pow(r - meanR, 2)));
				var stdDevG = Math.Sqrt(imgCvChanG.Average(g => Math.Pow(g - meanG, 2)));
				var stdDevB = Math.Sqrt(imgCvChanB.Average(b => Math.Pow(b - meanB, 2)));

				for (var i = 0; i < imgCvChanR.Length; i++)
				{
					imgCvChanRNorm[i] = (Half)Convert.ToSingle((imgCvChanR[i] - meanR) / stdDevR);
					imgCvChanGNorm[i] = (Half)Convert.ToSingle((imgCvChanG[i] - meanG) / stdDevG);
					imgCvChanBNorm[i] = (Half)Convert.ToSingle((imgCvChanB[i] - meanB) / stdDevB);
				}

				tens = cat(new[]
				{
					tensor(imgCvChanRNorm, new long[] { 1, image.Width, image.Height }).to_type(bfloat16),
					tensor(imgCvChanGNorm, new long[] { 1, image.Width, image.Height }).to_type(bfloat16),
					tensor(imgCvChanBNorm, new long[] { 1, image.Width, image.Height }).to_type(bfloat16)
				}, 0);
			}
			if (usePrecision == float32)
			{
				var imgCvChanRNorm = new float[image.Height * image.Width];
				var imgCvChanGNorm = new float[image.Height * image.Width];
				var imgCvChanBNorm = new float[image.Height * image.Width];

				for (var y = 0; y < image.Height; y++)
				{
					for (var x = 0; x < image.Width; x++)
					{
						imgCvChanR[y * image.Width + x] = indexer[y, x].Item2;
						imgCvChanG[y * image.Width + x] = indexer[y, x].Item1;
						imgCvChanB[y * image.Width + x] = indexer[y, x].Item0;
					}
				}

				var meanR = imgCvChanR.Average();
				var meanG = imgCvChanG.Average();
				var meanB = imgCvChanB.Average();
				var stdDevR = Math.Sqrt(imgCvChanR.Average(r => Math.Pow(r - meanR, 2)));
				var stdDevG = Math.Sqrt(imgCvChanG.Average(g => Math.Pow(g - meanG, 2)));
				var stdDevB = Math.Sqrt(imgCvChanB.Average(b => Math.Pow(b - meanB, 2)));

				for (var i = 0; i < imgCvChanR.Length; i++)
				{
					imgCvChanRNorm[i] = Convert.ToSingle((imgCvChanR[i] - meanR) / stdDevR);
					imgCvChanGNorm[i] = Convert.ToSingle((imgCvChanG[i] - meanG) / stdDevG);
					imgCvChanBNorm[i] = Convert.ToSingle((imgCvChanB[i] - meanB) / stdDevB);
				}

				tens = cat(new[]
				{
					tensor(imgCvChanRNorm, new long[] { 1, image.Width, image.Height }).to_type(float32),
					tensor(imgCvChanGNorm, new long[] { 1, image.Width, image.Height }).to_type(float32),
					tensor(imgCvChanBNorm, new long[] { 1, image.Width, image.Height }).to_type(float32)
				}, 0);
			}
			return tens!.to(device);
		}

		/// <summary>
		/// Converts an image mask of type 8UC1 to tensor (not normalizing the values)
		/// </summary>
		/// <param name="img"></param>
		/// <param name="device"></param>
		/// <param name="usePrecision"></param>
		/// <returns></returns>
		/// <exception cref="ArgumentException"></exception>
		public static Tensor MaskToTensor(Mat img, Device device, ScalarType usePrecision)
		{
			if (img.Type() != MatType.CV_8UC1)
			{
				throw new ArgumentException("Invalid image type");
			}

			Tensor? tens = null;
			var matVec = new Mat<Vec3b>(img);
			var indexer = matVec.GetIndexer();

			if (usePrecision == bfloat16)
			{
				var imgCvChan = new Half[img.Height * img.Width];

				for (var y = 0; y < img.Height; y++)
				{
					for (var x = 0; x < img.Width; x++)
					{
						imgCvChan[y * img.Width + x] = (Half)(indexer[y, x].Item0 / 255.0f);
					}
				}

				tens = cat(new[]
				{
					tensor(imgCvChan, new long[] { 1, img.Width, img.Height }).to_type(bfloat16)
				}, 0);
			}
			if (usePrecision == float32)
			{
				var imgCvChan = new float[img.Height * img.Width];

				for (var y = 0; y < img.Height; y++)
				{
					for (var x = 0; x < img.Width; x++)
					{
						imgCvChan[y * img.Width + x] = indexer[y, x].Item0 / 255.0f;
					}
				}

				tens = cat(new[]
				{
					tensor(imgCvChan, new long[] { 1, img.Width, img.Height }).to_type(float32)
				}, 0);
			}
			return tens!.to(device);
		}

		/// <summary>
		/// Converts a tensor to a Mat image of type 8UC1 (with standard normalization of *255)
		/// Tensor of [H x W] expected
		/// </summary>
		/// <param name="tens"></param>
		/// <returns></returns>
		/// <exception cref="ArgumentException"></exception>
		public static Mat TensorToGreyImage(Tensor tens)
		{
			if (tens.Dimensions != 2)
			{
				throw new ArgumentException("Invalid tensor dimensions");
			}

			var imgGrey = new Mat((int)tens.shape[0], (int)tens.shape[1], MatType.CV_8UC1);
			var tensFloatArray = tens.data<float>().ToArray();

			var a = 0;
			for (var x = 0; x < (int)tens.shape[0]; x++)
			{
				for (var y = 0; y < (int)tens.shape[1]; y++)
				{
					imgGrey.At<Vec3b>(x, y) =
						new Vec3b((byte)(tensFloatArray[a] * 255.0), (byte)(tensFloatArray[a] * 255.0), (byte)(tensFloatArray[a] * 255.0));

					a++;
				}
			}
			return imgGrey;
		}

		/// <summary>
		/// Converts a tensor to a Mat image of type 8UC1 (with standard normalization of *255)
		/// Tensor of [3 x H x W] expected
		/// </summary>
		/// <param name="tens"></param>
		/// <returns></returns>
		public static Mat TensorToRgbImage(Tensor tens)
		{
		
			var imgRgb = new Mat((int)tens[0].shape[0], (int)tens[0].shape[1], MatType.CV_8UC3);

			var tensSigR = functional.sigmoid(tens[0]);
			var tensSigG = functional.sigmoid(tens[1]);
			var tensSigB = functional.sigmoid(tens[2]);

			var tensFloatArrayR = tensSigR.data<float>().ToArray();
			var tensFloatArrayG = tensSigG.data<float>().ToArray();
			var tensFloatArrayB = tensSigB.data<float>().ToArray();

			var a = 0;
			for (var x = 0; x < (int)tens[0].shape[0]; x++)
			{
				for (int y = 0; y < (int)tens[0].shape[1]; y++)
				{
					imgRgb.At<Vec3b>(x, y) =
						new Vec3b((byte)(tensFloatArrayR[a] * 255.0), (byte)(tensFloatArrayG[a] * 255.0), (byte)(tensFloatArrayB[a] * 255.0));

					a++;
				}
			}
			return imgRgb;
		}

		/// <summary>
		/// Converts the first two dimensions of a tensor into arrays:
		/// Tensor of [A x B x H x W] to tensor of [A][B][H x W]
		/// </summary>
		/// <param name="tens"></param>
		/// <returns></returns>
		/// <exception cref="ArgumentException"></exception>
		public static Tensor[][] TensorTo2DArray(Tensor tens)
		{
			if (tens.Dimensions != 4)
			{
				throw new ArgumentException("Invalid tensor dimensions");
			}

			var dim0 = tens.shape[0];
			var dim1 = tens.shape[1];
			var tensArr = new Tensor[dim0][];

			// Split first dim to array
			var tens1Dim = tens.clone().split(1);

			for (var a = 0; a < dim0; a++)
			{
				// Split second dim to array
				var tens2Dim = tens1Dim[a].clone().squeeze(0).split(1);

				tensArr[a] = new Tensor[dim1];

				for (var b = 0; b < dim1; b++)
				{
					// Should always be [H x W]
					tensArr[a][b] = tens2Dim[b].clone().squeeze(0);
				}
			}
			return tensArr;
		}

		/// <summary>
		/// Converts an array of Mat images to a tensor which has a first dimension size of the amount of images:
		/// Images[] -> Tensor:[Images.Length x 1 x Images.Width x Images.Height]
		/// This makes calculations later on much more efficient
		/// </summary>
		/// <param name="images"></param>
		/// <param name="model"></param>
		/// <param name="uNetPara"></param>
		/// <returns></returns>
		public static async Task<Tensor> ImageToRoiSizedTensor(Mat[] images, Module<Tensor, Tensor> model, IModelParameter uNetPara)
		{
			var tasks = new Task<Tensor>[images.Length];

			var t = 0;
			foreach (var image in images)
			{
				if (uNetPara.TrainImagesAsGreyscale)
				{
					tasks[t] = Task.Run(() => GreyImageToTensor(image, new Device(uNetPara.TrainOnDevice), uNetPara.TrainPrecision).unsqueeze(1));
				}
				else
				{
					tasks[t] = Task.Run(() => RgbImageToTensor(image, new Device(uNetPara.TrainOnDevice), uNetPara.TrainPrecision).unsqueeze(0));
				}
				t++;
			}
			return cat(await Task.WhenAll(tasks), 0);
		}

	}
}
