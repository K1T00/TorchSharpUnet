using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using AiModels.ModelUtils;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.utils.data;
using static TorchSharp.torchvision;
using static AiModels.ModelUtils.LossFunctions;
using Utils;
using static Utils.HelperFunctions;


namespace AiModels.UNet
{
    public class UNetRun
	{
	
		private Device? device;
		

		public void Run(IModelParameter uNetPara, Db cfgPara, IProgress<string> logProgress, IProgress<Tuple<int, float, float>> lossProgress, CancellationToken token)
		{
			this.device = new Device(uNetPara.TrainOnDevice);

			logProgress.Report(DateTime.Now.ToString("HH:mm:ss tt") + "\r\n" + "Started training UNet model on device: " + this.device);

            // To make RNG reproducible between training runs
            random.manual_seed(42);
            var epoch = 1;
            var valLoss = 1.0f;

            // Split: Train -> [0] / Validation -> [1]
            var trainAndValidateImages = SplitList(cfgPara.DatasetImages, Convert.ToInt16(cfgPara.DatasetImages.Count * uNetPara.SplitTrainValidationSet)).ToList();
            var trainAndValidateMasks = SplitList(cfgPara.DatasetMasks, Convert.ToInt16(cfgPara.DatasetMasks.Count * uNetPara.SplitTrainValidationSet)).ToList();
            // At least one training image/mask
            if (trainAndValidateImages.Count == 1)
            {
                trainAndValidateImages.Add(new List<Tuple<string, int>> { cfgPara.DatasetImages[0] });
				trainAndValidateMasks.Add(new List<Tuple<string, int, int>> { cfgPara.DatasetMasks[0] });
			}

            if (trainAndValidateImages.Count == 0 | trainAndValidateMasks.Count == 0)
            {
	            throw new TrainImagesException("Train images or masks could not be loaded.");
            }

            using var model = new UNetModelBn(uNetPara);
            using var trainingDataSet = new UNetDataset(trainAndValidateImages[0], trainAndValidateMasks[0], uNetPara, this.device, AddAugmentations());
            using var validationDataSet = new UNetDataset(trainAndValidateImages[1], trainAndValidateMasks[1], uNetPara, this.device, AddAugmentations());
            using var trainData = DataLoader(trainingDataSet, uNetPara.BatchSize, false, this.device, num_worker: 3);
            using var validationData = DataLoader(validationDataSet, uNetPara.BatchSize, false, this.device, num_worker: 3);
            using var optimizer = optim.Adam(model.parameters(), uNetPara.LearningRate);
            //using var optimizer = optim.SGD(model.parameters(), uNetPara.LearningRate, 0.9, 0, 0.0005);

            // TODO: Will be merged into TorchSharp soon: https://github.com/dotnet/TorchSharp/issues/1353
            //var dataset = new ConcatDataset<List<Tuple<string, int>>>([trainingDataSet, validationDataSed]);

            var lrScheduler = optim.lr_scheduler.StepLR(optimizer, Convert.ToInt32(0.8 * uNetPara.MaxEpochs), 0.6);
            //var scheduleOptimizer = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", 0.1, 5);

            while (epoch <= uNetPara.MaxEpochs & valLoss > uNetPara.StopAtLoss)
			{
				var trainLoss = Train(model, optimizer!, trainData);

				lrScheduler.step();

				valLoss = Validate(model, validationData);

				lossProgress.Report(new Tuple<int, float, float>(epoch, (float)Math.Round(trainLoss, 4), (float)Math.Round(valLoss, 4)));
				logProgress.Report("Epoch: " + epoch + "\r\n" + "Train loss: " + trainLoss + "\r\n" + "Val loss: " + valLoss + "\r\n" +
				                   "Learning rate: " + optimizer.ParamGroups.ToList().FirstOrDefault()?.LearningRate);

				if (epoch % 5 == 0)
				{
					model.save(cfgPara.ModelPath + "model.bin");
					logProgress.Report("Current model saved.");
				}
				epoch++;

				if (token.IsCancellationRequested)
				{
					break;
				}
			}
			model.save(cfgPara.ModelPath + "model.bin");
			logProgress.Report(DateTime.Now.ToString("HH:mm:ss tt") + "\r\n" + "Training done, model saved.");
		}

		private static float Train(Module<Tensor, Tensor> model, optim.Optimizer optimizer, DataLoader trainData)
		{
			var batchLoss = 0.0f;
			model.train();
			using var d = NewDisposeScope();
			using var useGrad = enable_grad();
			//using var inferenceMode = inference_mode(false);

			// Run each batch
			foreach (var data in trainData)
			{
				// Run model with Tensor
				var prediction = model.call(data["data"]);

				// Compute the loss: Comparing result tensor with ground truth tensor
				var computedLoss = CalculateUNetLoss(prediction, data["masks"]);

				batchLoss = computedLoss.ToSingle();

				// Clear the gradients before doing the back-propagation
				optimizer.zero_grad();

				// Do back-propagation, which computes all the gradients
				computedLoss.backward();

				// Adjust the weights using the (newly calculated) gradients
				optimizer.step();

				d.DisposeEverything();
			}
			return batchLoss;
		}

		private static float Validate(Module<Tensor, Tensor> model, DataLoader validationData)
		{
			var loss = 0.0f;
			model.eval();
			using var d = NewDisposeScope();
			using var noGrad = no_grad();
			//using var inferenceMode = inference_mode(true);

			foreach (var data in validationData)
			{
				var prediction = model.call(data["data"]);

				var computedLoss = CalculateUNetLoss(prediction, data["masks"]);

				loss = computedLoss.ToSingle();

				d.DisposeEverything();

				break; // TODO
			}
			return loss;
		}

		private static ITransform AddAugmentations()
		{
			// TODO
			//var hflip = torchvision.transforms.HorizontalFlip();
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
            return transforms.Compose();
		}


    }

}
