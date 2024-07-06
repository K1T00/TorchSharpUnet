using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Models.UNet.UNetDataset;

namespace Models.Utils
{
	public class HelperFunctions
	{

		/// <summary>
		/// Convert an array to a C# matrix
		/// </summary>
		/// <param name="flat">An array</param>
		/// <param name="m">Rows</param>
		/// <param name="n">Columns</param>
		/// <returns>The C# Matrix</returns>
		/// <exception cref="ArgumentException"></exception>
		public static double[,] ConvertToMatrix(double[] flat, int m, int n)
		{
			if (flat.Length != m * n)
			{
				throw new ArgumentException("Invalid length");
			}
			double[,] ret = new double[m, n];

			Buffer.BlockCopy(flat, 0, ret, 0, flat.Length * sizeof(double));
			return ret;
		}

		/// <summary>
		/// Assume the following file structure:
		///
		/// train
		///      ...grabs
		///         ..."imgNr".bmp
		///      ...masks
		///         ..."imgNr"_"maskNr".bmp
		/// 
		/// validate
		///         ...grabs
		///            ..."imgNr".bmp
		///         ...masks
		///            ..."imgNr"_"maskNr".bmp
		/// </summary>
		/// <param name="path">Path of the train/validate folders</param>
		/// <returns>Tuple of filepath, grabNr, maskNr </returns>
		public static List<Tuple<string, int, int>> LoadAndSortImageFiles(string path)
        {
			List<Tuple<string, int, int>> fileNames = new List<Tuple<string, int, int>>();

			foreach (var datasetImage in Directory.GetFiles(path, "*", SearchOption.TopDirectoryOnly))
			{
                int sStart = datasetImage.LastIndexOf("\\") + 1;
                int sTo = datasetImage.LastIndexOf(".");

				fileNames.Add(
					new Tuple<string, int, int>(
						datasetImage,
						Convert.ToInt16(datasetImage.Substring(sStart, sTo - sStart)),
						0));

			}
			return fileNames.OrderBy(t => t.Item2).ToList();
        }

		/// <summary>
		/// Assume the following file structure:
		///
		/// train
		///      ...grabs
		///         ..."imgNr".bmp
		///      ...masks
		///         ..."imgNr"_"maskNr".bmp
		/// 
		/// validate
		///         ...grabs
		///            ..."imgNr".bmp
		///         ...masks
		///            ..."imgNr"_"maskNr".bmp
		/// </summary>
		/// <param name="path">Path of the train/validate folders</param>
		/// <returns>Tuple of filepath, grabNr, maskNr </returns>
		public static List<Tuple<string, int, int>> LoadAndSortMaskFiles(string path)
        {
			List<Tuple<string, int, int>> fileNames = new List<Tuple<string, int, int>>();

			foreach (var datasetImage in Directory.GetFiles(path, "*", SearchOption.TopDirectoryOnly))
			{

                int sStart = datasetImage.LastIndexOf("\\") + 1;
                int sTo = datasetImage.LastIndexOf(".");
				int inBetween = datasetImage.LastIndexOf("_");

				fileNames.Add(
                    new Tuple<string, int, int>(
					datasetImage,
					Convert.ToInt16(datasetImage.Substring(sStart, sTo - sStart).Split("_")[0]),
					Convert.ToInt16(datasetImage.Substring(sStart, sTo - sStart).Split("_")[1])));

			}
			return fileNames.OrderBy(t => t.Item2).ToList();
        }

    }
}
