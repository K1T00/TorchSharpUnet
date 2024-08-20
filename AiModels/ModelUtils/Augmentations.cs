using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using TorchSharp;
using AiModels.UNet;
using static TorchSharp.torchvision;

namespace AiModels.ModelUtils
{
	public class Augmentations : ICloneable
	{
		// Item1 : Apply on masks

		public Tuple<bool, double> RandomRotation { get; set; } = new Tuple<bool, double>(true, 0);

		public Tuple<bool, double> RandomSharpness { get; set; } = new Tuple<bool, double>(false, 0);

		public Tuple<bool, double> RandomPerspective { get; set; } = new Tuple<bool, double>(true, 0);

		public Tuple<bool, double> RandomContrast { get; set; } = new Tuple<bool, double>(false, 0);

		public ITransform GetAugmentations()
		{
			var transformList = new List<ITransform>();

			if (this.RandomRotation.Item2 > 0)
			{
				transformList.Add(transforms.RandomRotation(this.RandomRotation.Item2));
			}

			if (this.RandomSharpness.Item2 > 0)
			{
				transformList.Add(transforms.RandomAdjustSharpness(this.RandomSharpness.Item2));
			}

			if (this.RandomPerspective.Item2 > 0)
			{
				transformList.Add(transforms.RandomPerspective(this.RandomPerspective.Item2));
			}

			if (this.RandomContrast.Item2 > 0)
			{
				transformList.Add(transforms.RandomAutoContrast(this.RandomContrast.Item2));
			}

			//var hflip = torchvision.transforms.
			//var solarize = transforms.RandomSolarize(0.5, 0.5);
			//var gray = transforms.Grayscale(3);
			//var rotate = transforms.Rotate(90);
			//var contrast = transforms.AdjustContrast(1.25);
			//var transform = torchvision.transforms.RandAugment(generator: g);
			//var transform = torchvision.transforms.AugMix(generator: g);
			//var normalize = transforms.Normalize( new double[] { 0.485, 0.456, 0.406 }, new double[] { 0.229, 0.224, 0.225 },
			//	ScalarType.Float32,
			//	device);

			//return transforms.Compose(hflip, rotate);

			ITransform[] tranArr = new ITransform[transformList.Count];

			var t = 0;
			foreach (var tra in transformList)
			{
				tranArr[t] = tra;
				t++;
			}

			var test = transforms.Compose(tranArr);


			return transforms.Compose();


		}

		public object Clone()
		{
			return new Augmentations
			{
				RandomRotation = this.RandomRotation,
				RandomSharpness = this.RandomSharpness,
				RandomPerspective = this.RandomPerspective,
				RandomContrast = this.RandomContrast
			};
		}




	}
}
