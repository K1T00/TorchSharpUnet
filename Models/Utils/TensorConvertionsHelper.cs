using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using ScottPlot.Plottables;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static Models.Utils.HelperFunctions;
using System.IO;
using ScottPlot;
using static TorchSharp.torch.utils;
using System.ComponentModel.DataAnnotations;
using System.Runtime.InteropServices;
using OpenTK.Graphics.ES20;
using Buffer = System.Buffer;
using static TorchSharp.torch.nn;


namespace Models.Utils
{
	public class TensorConvertionsHelper
	{
		
		public static Tensor RgbImageToTensor(Mat img, Device device)
		{
			if (img.Type() != MatType.CV_8UC3)
			{
				throw new ArgumentException("Invalid image type");
			}

            var mat3 = new Mat<Vec3b>(img);
            var indexer = mat3.GetIndexer();

            var imgCvChanR = new List<float>();
			var imgCvChanG = new List<float>();
			var imgCvChanB = new List<float>();

			for (int y = 0; y < img.Height; y++)
            {
                for (int x = 0; x < img.Width; x++)
                {
					imgCvChanR.Add(indexer[y, x].Item2);
					imgCvChanG.Add(indexer[y, x].Item1);
					imgCvChanB.Add(indexer[y, x].Item0);
				}
            }

			var imgCvChanRNorm = new List<float>();
			var imgCvChanGNorm = new List<float>();
			var imgCvChanBNorm = new List<float>();

			var meanR = imgCvChanR.Average();
			var meanG = imgCvChanG.Average();
			var meanB = imgCvChanB.Average();

			var stdDevR = Math.Sqrt(imgCvChanR.Average(r => Math.Pow(r - meanR, 2)));
			var stdDevG = Math.Sqrt(imgCvChanG.Average(g => Math.Pow(g - meanG, 2)));
			var stdDevB = Math.Sqrt(imgCvChanB.Average(b => Math.Pow(b - meanB, 2)));

			for (int i = 0; i < imgCvChanR.Count; i++)
			{
				var subMR = imgCvChanR[i] - meanR;
				var subMG = imgCvChanG[i] - meanG;
				var subMB = imgCvChanB[i] - meanB;

				imgCvChanRNorm.Add(Convert.ToSingle(subMR / stdDevR));
				imgCvChanGNorm.Add(Convert.ToSingle(subMG / stdDevG));
				imgCvChanBNorm.Add(Convert.ToSingle(subMB / stdDevB));
			}

			var testMax = imgCvChanRNorm.Max();
			var testMin = imgCvChanRNorm.Min();

            var tens = cat(new[]
            {
                tensor(imgCvChanRNorm, new long[] { 1, img.Width, img.Height }),
				tensor(imgCvChanGNorm, new long[] { 1, img.Width, img.Height }),
				tensor(imgCvChanBNorm, new long[] { 1, img.Width, img.Height })
			}, 0);

            return device == CUDA ? tens.cuda(device) : tens;
		}

		public static Tensor GreyImageToTensor(Mat img, Device device)
		{
			if (img.Type() != MatType.CV_8UC1)
			{
				throw new ArgumentException("Invalid image type");
			}

			var mat3 = new Mat<Vec3b>(img);
			var indexer = mat3.GetIndexer();

			var imgCvChan = new List<float>();

			for (int y = 0; y < img.Height; y++)
			{
				for (int x = 0; x < img.Width; x++)
				{
					imgCvChan.Add(indexer[y, x].Item0);
				}
			}

			var imgCvChanNorm = new List<float>();

			var mean = imgCvChan.Average();

			var stdDev = Math.Sqrt(imgCvChan.Average(v => Math.Pow(v - mean, 2)));

			foreach (var idx in imgCvChan)
			{
				var subM = idx - mean;

				imgCvChanNorm.Add(Convert.ToSingle(subM / stdDev));
			}

			var testMax = imgCvChanNorm.Max();
			var testMin = imgCvChanNorm.Min();

			var tens = cat(new[]
			{
				tensor(imgCvChanNorm, new long[] { 1, img.Width, img.Height })
			}, 0);

			return device == CUDA ? tens.cuda(device) : tens;
		}

		public static Tensor GreyMaskToTensor(Mat img, Device device)
		{
			if (img.Type() != MatType.CV_8UC1)
			{
				throw new ArgumentException("Invalid image type");
			}

			var mat3 = new Mat<Vec3b>(img);
            var indexer = mat3.GetIndexer();

            var imgCvChan = new List<float>();

            for (int y = 0; y < img.Height; y++)
            {
                for (int x = 0; x < img.Width; x++)
                {
                    imgCvChan.Add(indexer[y, x].Item2 / 255.0f);
                }
            }

            var tens = cat(new[]
            {
                tensor(imgCvChan, new long[] { 1, img.Width, img.Height })
            }, 0);

            return device == CUDA ? tens.cuda(device) : tens;
		}

		public static Plot TensorToHeatmap(Tensor tens, bool withColorBar)
		{
			if (tens.Dimensions != 2)
			{
				throw new ArgumentException("Invalid tensor dimensions");
			}

			Plot heatmap = new Plot();

			float[] tensFloatArray = tens.data<float>().ToArray();

			var tensDoubleArray = ConvertToMatrix(tensFloatArray.Select(f => (double)f).ToArray(), (int)tens.shape[0], (int)tens.shape[1]);

			var myHeatmap = heatmap.Add.Heatmap(tensDoubleArray);

			myHeatmap.Colormap = new ScottPlot.Colormaps.Turbo();

			if (withColorBar)
			{
				heatmap.Add.ColorBar(myHeatmap);
			}

			//heatmap.SavePng("D:\\_DEV\\_GITHUB\\DATA\\TestData1Feature\\validate\\heatmaps\\" + "TEST.png", 500, 400);

			return heatmap;
		}

		/// <summary>
		/// Tensor of [RGB][H x W] expected
		/// </summary>
		/// <param name="tens"></param>
		public static Mat TensorToRgbImage(Tensor[] tens)
		{
			if (tens[0].Dimensions != 2 | tens[1].Dimensions != 2 | tens[2].Dimensions != 2 | tens.Length != 3)
			{
				throw new ArgumentException("Invalid tensor dimensions");
			}

			Mat imgRgb = new Mat((int)tens[0].shape[0], (int)tens[0].shape[1], MatType.CV_8UC3);

			// TODO: Calc/Serialize Mean/StdDev...
			var tensSigR = functional.sigmoid(tens[0]);
			var tensSigG = functional.sigmoid(tens[1]);
			var tensSigB = functional.sigmoid(tens[2]);

			float[] tensFloatArrayR = tensSigR.data<float>().ToArray();
			float[] tensFloatArrayG = tensSigG.data<float>().ToArray();
			float[] tensFloatArrayB = tensSigB.data<float>().ToArray();

			int a = 0;
			for (int x = 0; x < (int)tens[0].shape[0]; x++)
			{
				for (int y = 0; y < (int)tens[0].shape[1]; y++)
				{
					imgRgb.At<Vec3b>(x, y) =
						new Vec3b((byte)(tensFloatArrayR[a] * 255.0), (byte)(tensFloatArrayG[a] * 255.0), (byte)(tensFloatArrayB[a] * 255.0));

					a++;
				}
			}

			//imgRgb.SaveImage("D:\\_DEV\\_GITHUB\\DATA\\TestData1Feature\\validate\\heatmaps\\" + "TEST.png");

			return imgRgb;
		}

		/// <summary>
		/// Tensor of [H x W] expected
		/// </summary>
		/// <param name="tens"></param>
		public static Mat TensorToGreyImage(Tensor tens)
		{
			if (tens.Dimensions != 2)
			{
				throw new ArgumentException("Invalid tensor dimensions");
			}

			Mat imgGrey = new Mat((int)tens.shape[0], (int)tens.shape[1], MatType.CV_8UC1);

			float[] tensFloatArray = tens.data<float>().ToArray();

			int a = 0;
			for (int x = 0; x < (int)tens.shape[0]; x++)
			{
				for (int y = 0; y < (int)tens.shape[1]; y++)
				{
					imgGrey.At<Vec3b>(x, y) = 
						new Vec3b((byte)(tensFloatArray[a] * 255.0), (byte)(tensFloatArray[a] * 255.0), (byte)(tensFloatArray[a] * 255.0));

					a++;
				}
			}

			//imgGrey.SaveImage("D:\\_DEV\\_GITHUB\\DATA\\TestData1Feature\\validate\\heatmaps\\" + "TEST.png");

			return imgGrey;
		}

		/// <summary>
		/// Tensor of [a x b x H x W] to tensor of [a][b][H x W]
		/// </summary>
		/// <param name="tens"></param>
		public static Tensor[][] TensorTo2DArray(Tensor tens)
		{
			if (tens.Dimensions != 4)
			{
				throw new ArgumentException("Invalid tensor dimensions");
			}

			long dim0 = tens.shape[0];
			long dim1 = tens.shape[1];

			Tensor[][] tensArr = new Tensor[dim0][];

			// Split first dim to array
			var tens1Dim = tens.clone().split(1);

			for (int a = 0; a < dim0; a++)
			{
				// Split second dim to array
				var tens2Dim = tens1Dim[a].clone().squeeze(0).split(1);

				tensArr[a] = new Tensor[dim1];

				for (int b = 0; b < dim1; b++)
				{
					// Should always be [H x W]
					tensArr[a][b] = tens2Dim[b].clone().squeeze(0); ;
				}
			}
			return tensArr;
		}

    }
}
