using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using OpenCvSharp;
using static TorchSharp.torch;
using static TorchSharp.torchvision;
using static AiModels.ModelUtils.TensorCalculationsHelper;
using AiModels.ModelUtils;


namespace AiModels.UNet
{

    public class UNetDataset : torch.utils.data.Dataset
    {
        private readonly List<Tensor> data = [];
        private readonly List<Tensor> masks = [];
        private readonly ITransform transform;
		private readonly int count;
		private readonly Device device;
		private readonly List<Tuple<string, int>> imagesPaths;
		private readonly List<Tuple<string, int, int>> masksPaths;
		private readonly IModelParameter uNetPara;


		/// <summary>
		/// Load images/masks into a Dataset that can be consumed by DataLoader class
		/// </summary>
		/// <param name="imagesPaths">Path to the folder containing the image files.</param>
		/// <param name="masksPaths">Path to the folder containing the masks files.</param>
		/// <param name="uNetPara">Model hyper-parameter.</param>
		/// <param name="device">The device, i.e. CPU or GPU to place the output tensors on.</param>
		/// <param name="transform">A list of image transformations, helpful for data augmentation.</param>
		/// <returns>
		///	data = images as Tensors
		/// masks = feature masks as Tensors
		/// </returns>
		public UNetDataset(List<Tuple<string, int>> imagesPaths, List<Tuple<string, int, int>> masksPaths, IModelParameter uNetPara, Device device, ITransform transform)
		{
			this.imagesPaths = imagesPaths;
			this.masksPaths = masksPaths;
			this.device = device;
			this.uNetPara = uNetPara;
			this.transform = transform;
			this.count = ReadFiles();
		}

        private int ReadFiles()
        {
            // For every image there needs to be one mask per feature
            for (var imIdx = 0; imIdx < imagesPaths.Count; imIdx++)
            {
                // Images
                if (this.uNetPara.TrainImagesAsGreyscale)
                {
                    using var img = Cv2.ImRead(imagesPaths[imIdx].Item1, ImreadModes.Grayscale);
                    data.Add( GreyImageToTensor(img, this.device, this.uNetPara.TrainPrecision));
                }
                else
                {
                    using var img = Cv2.ImRead(imagesPaths[imIdx].Item1, ImreadModes.Color);
					data.Add( RgbImageToTensor(img, this.device, this.uNetPara.TrainPrecision));
                }
                // Masks
                if (this.uNetPara.Features == 1)
                {
                    using var mskGrey = Cv2.ImRead(masksPaths[imIdx].Item1, ImreadModes.Grayscale);
                    masks.Add( MaskToTensor(mskGrey, this.device, this.uNetPara.TrainPrecision));
                }
                else
                {
                    var tensorLabelsAr = new Tensor[this.uNetPara.Features];

                    for (var ma = 0; ma < this.uNetPara.Features; ma++)
                    {
                        using var mskGrey = Cv2.ImRead(masksPaths[imIdx * this.uNetPara.Features + ma].Item1, ImreadModes.Unchanged);
                        tensorLabelsAr[ma] = MaskToTensor(mskGrey, this.device, this.uNetPara.TrainPrecision);
                    }
                    masks.Add(cat(tensorLabelsAr, 0));
                }
            }
            return data.Count;
        }

		protected override void Dispose(bool disposing)
        {
            if (!disposing) return;
			data.ForEach(d => d.Dispose());
			masks.ForEach(d => d.Dispose());
		}

		/// <summary>
		/// Get tensor according to index
		/// </summary>
		/// <param name="index">Index for tensor</param>
		/// <returns>Tensors of index.</returns>
		public override Dictionary<string, Tensor> GetTensor(long index)
		{
            return
                new Dictionary<string, Tensor>()
                {
                    { "data", transform.call(data[(int)index]) },
                    // TODO: On masks only certain transforms shall be performed
                    { "masks", transform.call(masks[(int)index]) }
                };
        }

		public override long Count => count;

	}
}
