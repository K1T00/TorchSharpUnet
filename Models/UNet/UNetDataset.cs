using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Models.Utils;
using TorchSharp;
using static TorchSharp.torch;
using static Models.Utils.HelperFunctions;
using static Models.Utils.TensorConvertionsHelper;
using OpenCvSharp;


namespace Models.UNet
{
	public class UNetDataset : torch.utils.data.Dataset
    {
        private readonly List<Tensor> data = [];
        private readonly List<Tensor> masks = [];
        private readonly torchvision.ITransform transform;
		private readonly int count;
		private readonly Device device;


		/// <summary>
		/// Load images/masks and convert them to tensors
		/// </summary>
		/// <param name="grabsPath">Path to the folder containing the image files.</param>
		/// <param name="masksPath">Path to the folder containing the masks files.</param>
		/// <param name="nFeatures">Number of features.</param>
		/// <param name="device">The device, i.e. CPU or GPU to place the output tensors on.</param>
		/// <param name="transform">A list of image transformations, helpful for data augmentation.</param>
		/// <returns>
		///	data = images as Tensors
		/// masks = feature masks as Tensors
		/// </returns>
		public UNetDataset(string grabsPath, string masksPath, int nFeatures, Device device, torchvision.ITransform transform)
		{
			this.device = device;
			this.transform = transform;

			count = nFeatures == 1 ? ReadFiles1Feature(grabsPath, masksPath) : ReadFilesNFeatures(grabsPath, masksPath, nFeatures);
		}

		private int ReadFilesNFeatures(string grabsPath, string masksPath, int nFeatures)
		{
            var datasetImages = LoadAndSortImageFiles(grabsPath);
            var datasetLabels = LoadAndSortMaskFiles(masksPath);

			for (int imIdx = 0; imIdx < datasetImages.Count; imIdx++)
			{
				// Image
				using var imgGrey = Cv2.ImRead(datasetImages[imIdx].Item1, ImreadModes.Grayscale);
				//using var imgRgb = Cv2.ImRead(datasetImages[imIdx].Item1, ImreadModes.Color);

				data.Add(GreyImageToTensor(imgGrey, device));

				// Masks
				Tensor[] tensorLabelsAr = new Tensor[nFeatures];

				for (int ma = 0; ma < nFeatures; ma++)
				{
					using var mskGrey = Cv2.ImRead(datasetLabels[imIdx * nFeatures + ma].Item1, ImreadModes.Grayscale);

					tensorLabelsAr[ma] = GreyMaskToTensor(mskGrey, device);
				}
				masks.Add(cat(tensorLabelsAr, 0));

				// DEBUG //
				if (imIdx == 119)
				{
					break;
				}
			}
			return data.Count;
		}

		private int ReadFiles1Feature(string grabsPath, string masksPath)
		{
			var datasetImages = LoadAndSortImageFiles(grabsPath);
			var datasetLabels = LoadAndSortMaskFiles(masksPath);

			for (int imIdx = 0; imIdx < datasetImages.Count; imIdx++)
			{
				// Image
				using var imgGrey = Cv2.ImRead(datasetImages[imIdx].Item1, ImreadModes.Grayscale);
				//using var imgRgb = Cv2.ImRead(datasetImages[imIdx].Item1, ImreadModes.Color);

				data.Add(GreyImageToTensor(imgGrey, device));


				// Masks

				// DEBUG Test //
				int mskIdx = imIdx * 6 + 3;
				var test = datasetLabels[imIdx * 6 + 3];

				using var mskGrey = Cv2.ImRead(datasetLabels[mskIdx].Item1, ImreadModes.Grayscale);

				masks.Add(GreyMaskToTensor(mskGrey, device));

				// DEBUG //
				if (imIdx == 119)
				{
					break;
				}
			}

			return data.Count;
		}

		protected override void Dispose(bool disposing)
		{
			if (disposing)
			{
				data.ForEach(d => d.Dispose());
				masks.ForEach(d => d.Dispose());
			}
		}

		/// <summary>
		/// Get tensor according to index
		/// </summary>
		/// <param name="index">Index for tensor</param>
		/// <returns>Tensors of index. DataLoader will catenate these tensors into batches.</returns>
		public override Dictionary<string, Tensor> GetTensor(long index)
		{
			var rdic = new Dictionary<string, Tensor>();

			if (transform is not null)
			{
				rdic.Add("data", transform.call(data[(int)index]));
			}
			else
			{
				rdic.Add("data", data[(int)index]);
			}

            if (transform is not null)
            {
				// TODO: Only certain transforms shall be performed on masks
                rdic.Add("masks", transform.call(masks[(int)index]));
            }
            else
            {
                rdic.Add("masks", masks[(int)index]);
            }

			// return new() {{"data", torch.tensor(1)}, {"label", torch.tensor(13)}, {"index", torch.tensor(index)}};

			return rdic;
		}

		public override long Count => count;

	}
}
