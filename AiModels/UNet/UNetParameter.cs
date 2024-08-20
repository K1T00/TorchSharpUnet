using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AiModels.ModelUtils;
using TorchSharp;
using static TorchSharp.torch;

namespace AiModels.UNet
{
	public class UNetParameter : ICloneable, IModelParameter
	{
		/// <summary>
		/// At what epoch the training should stop
		/// </summary>
		public int MaxEpochs { get; set; } = 0;

		/// <summary>
		/// Learning rate of the optimizer
		/// </summary>
		public double LearningRate { get; set; } = 0.00;

		/// <summary>
		/// Size of one batch (Tensor)
		/// </summary>
		public int BatchSize { get; set; } = 17;

		/// <summary>
		/// Amount of classes the model has
		/// </summary>
		public int Features { get; set; } = 1;

		/// <summary>
		/// At which loss the model should stop training
		/// </summary>
		public float StopAtLoss { get; set; } = 0.15f;

		/// <summary>
		/// Amount of classes the model has to be trained on
		/// </summary>
		public bool TrainImagesAsGreyscale { get; set; } = true;

		/// <summary>
		/// The percentage split between train and validation data
		/// </summary>
		public float SplitTrainValidationSet { get; set; } = 0.8f;

		/// <summary>
		/// Precision used for image tensors
		/// </summary>
		public ScalarType TrainPrecision { get; set; } = ScalarType.Float32;

		/// <summary>
		/// Train device: CPU or CUDA
		/// </summary>
		public DeviceType TrainOnDevice { get; set; } = DeviceType.CPU;

		/// <summary>
		/// Filter size of first Conv_2d layer
		/// </summary>
		public int FirstFilterSize { get; set; } = 64;

		public object Clone()
		{
			return new UNetParameter
			{
				MaxEpochs = this.MaxEpochs,
				LearningRate = this.LearningRate,
				BatchSize = this.BatchSize,
				Features = this.Features,
				StopAtLoss = this.StopAtLoss,
				TrainImagesAsGreyscale = this.TrainImagesAsGreyscale,
				SplitTrainValidationSet = this.SplitTrainValidationSet,
				TrainPrecision = this.TrainPrecision,
				TrainOnDevice = this.TrainOnDevice,
				FirstFilterSize = this.FirstFilterSize
			};
		}
	}
}
