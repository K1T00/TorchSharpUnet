using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace AiModels.ModelUtils
{

    public class LossFunctions
	{

		public static Tensor CalculateUNetLoss(Tensor prediction, Tensor target)
		{
			return CalculateLossNonBinary(prediction, target);
		}

        private static Tensor CalculateLossNonBinary(Tensor prediction, Tensor target)
        {
            const float bceWeight = 0.5f;
            var bceLoss = functional.binary_cross_entropy_with_logits(prediction, target);
            var sigLoss = functional.sigmoid(prediction);
            var diceLoss = DiceLoss(sigLoss, target, 1.0f);

            return bceLoss * bceWeight + diceLoss * (1 - bceWeight);
        }
        private static Tensor DiceLoss(Tensor prediction, Tensor target, float smooth)
		{
			var intersection = (prediction * target).sum(2).sum(2);

			return (1 - ((2.0f * intersection + smooth) / (prediction.sum(2).sum(2) + target.sum(2).sum(2) + smooth))).mean();
		}

		private static Tensor CalculateLossNonBinarySigmoid(Tensor prediction, Tensor target)
		{
			const float smooth = 1.0f;

			var sigLoss = functional.sigmoid(prediction);
			var inputsFlat = prediction.view(-1);
			var targetsFlat = target.view(-1);

			var intersection = (inputsFlat * targetsFlat).sum();
			var diceLoss = 1 - (2.0f * intersection + smooth) / (inputsFlat.sum() + target.sum() + smooth);
			var bceLoss = functional.binary_cross_entropy_with_logits(targetsFlat, inputsFlat, reduction: Reduction.Mean);

			return bceLoss + diceLoss;
		}

		private static Tensor CalculateLossNonBinaryCase(Tensor prediction, Tensor target)
		{
			// https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
			// https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
			// https://github.com/Mr-TalhaIlyas/Loss-Functions-Package-Tensorflow-Keras-PyTorch
			// https://github.com/Mr-TalhaIlyas/Loss-Functions-Package-Tensorflow-Keras-PyTorch
			// https://discuss.pytorch.org/t/multiclass-dice-loss-for-scene-labeling-problems/191505

			//if (n_classes > 1)
			//{
			//	torch.nn.CrossEntropyLoss();
			//}
			//else
			//{
			//	torch.nn.BCEWithLogitsLoss();
			//}

			// https://github.com/milesial/Pytorch-UNet/blob/master/train.py
			// -> criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()

			//var bceLoss = functional.binary_cross_entropy_with_logits(prediction, target);

			var bceLoss = functional.binary_cross_entropy_with_logits(prediction, target);
			var predLoss = functional.sigmoid(prediction);


			// MSE Loss
			var diceLoss = DiceLoss(prediction, target, 1.0f);

			var bceWeight = 0.5f;

			//var test = bceLoss * bceWeight + diceLoss * (1 - bceWeight);
			//var bceLossTest = bceLoss.item<float>();
			//var predLossTest = predLoss.mean().item<float>();
			//var diceLossTest = diceLoss.item<float>();
			//var testTest = test.item<float>();

			return bceLoss * bceWeight + diceLoss * (1 - bceWeight);
		}

	}
}
