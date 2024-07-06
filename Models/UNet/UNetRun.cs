using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torchvision;
using Models.Utils;
using static Models.Utils.TensorConvertionsHelper;
using static Models.Utils.LossFunctions;

namespace Models.UNet
{

	public class UNetRun
	{
		public event EventHandler<LossDataAvailableEventArgs>? LossDataAvailable;
		public event EventHandler<LogDataAvailableEventArgs>? LogDataAvailable;
		private static Device? device;
		

		public async void Run(UNetParameter uNetPara, ConfigParameter cfgPara)
		{

            device = cuda.is_available() ? device = CUDA : device = CPU;

            OnLogDataAvailable(new LogDataAvailableEventArgs(
	            DateTime.Now.ToString("HH:mm:ss tt") + Environment.NewLine + 
	            "Started training UNet model on device: " + device.ToString()));

            // To make RNG reproducible between training runs
            torch.random.manual_seed(42);

			var model = new UNetModel(device);

			if (uNetPara.UseSavedModel)
			{
				model.load(cfgPara.ModelPath);

				OnLogDataAvailable(new LogDataAvailableEventArgs("Loaded model: " + cfgPara.ModelPath));
			}


			using (var trainingData = new UNetDataset(cfgPara.TrainGrabsPath, cfgPara.TrainMasksPath, uNetPara.AmtClasses, device,null))
			using (var valData = new UNetDataset(cfgPara.ValidateGrabsPath, cfgPara.ValidateMasksPath, uNetPara.AmtClasses, device, null))
			using (var train = torch.utils.data.DataLoader(trainingData, uNetPara.BatchSize, false, device, num_worker: 3))
			using (var val = torch.utils.data.DataLoader(valData, uNetPara.BatchSize, false, device, num_worker: 3))
			using (var optimizer = optim.Adam(model.parameters(), uNetPara.LearningRate))
			//using (var optimizer = optim.SGD(model.parameters(), uNetPara.LearningRate, 0.9, 0, 0.0005))
			{

				int epoch = 1;
				float valLoss = 1;
				float trainLoss = 1;

				// Multiply learning rate by gamma after X% of epochs
				var scheduleOptimizer = optim.lr_scheduler.StepLR(optimizer, Convert.ToInt32(0.3 * uNetPara.MaxEpochs), 0.9);
				//var scheduleOptimizer = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", 0.1, 5);

				while (epoch <= uNetPara.MaxEpochs & valLoss > uNetPara.StopAtLoss & !uNetPara.UseSavedModel)
				{

					OnLossDataAvailable(new LossDataAvailableEventArgs(epoch, trainLoss, valLoss));

					await Task.Run(() =>
					{
						using var d = NewDisposeScope();

						trainLoss = Train(model, optimizer, train);

						scheduleOptimizer.step();

						valLoss = Validate(model, val);

					}).ConfigureAwait(true);


					OnLogDataAvailable(
						new LogDataAvailableEventArgs(
							Environment.NewLine +
							"Epoch: " + epoch.ToString() + Environment.NewLine +
							"Train loss: " + trainLoss.ToString() + Environment.NewLine +
							"Val loss: " + valLoss.ToString() + Environment.NewLine +
							"Learning rate: " + optimizer?.ParamGroups?.ToList()?.FirstOrDefault()?.LearningRate.ToString()
							));

					epoch++;
				}

				
				if (!uNetPara.UseSavedModel)
				{
					OnLogDataAvailable(new LogDataAvailableEventArgs(DateTime.Now.ToString("HH:mm:ss tt") + Environment.NewLine + "Done training, saving model to " + cfgPara.NewModelPath));

					if (File.Exists(cfgPara.NewModelPath)) File.Delete(cfgPara.NewModelPath);

					model.save(cfgPara.NewModelPath);
				}

				foreach (var data in val)
				{
					var prediction = model.forward(data["data"]);


					var predSplitted = TensorTo2DArray(functional.sigmoid(prediction));


					for (int batch = 0; batch < (int)prediction.shape[0]; batch++)
					{
						for (int chan = 0; chan < (int)prediction.shape[1]; chan++)
						{
							var heatmap = TensorToHeatmap(predSplitted[batch][chan], true);

							heatmap.SavePng(cfgPara.HeatmapsPath + batch.ToString() + "_" + chan.ToString() + ".png",
								(int)prediction.shape[2] * 3,
								(int)prediction.shape[3] * 3);

						}
					}
				}
			}
			model.Dispose();
		}

		private static float Train(Module<Tensor, Tensor> model, optim.Optimizer optimizer, DataLoader train)
		{
			model.train();

			float loss = 0.0f;

			// Run each batch
			foreach (var data in train)
			{
				// Run model with Tensor
				var prediction = model.forward(data["data"]);

				// Compute the loss
				var computedLoss = CalculateUnetLoss(prediction, data["masks"]);

				loss = computedLoss.ToSingle();

				// Clear the gradients before doing the back-propagation
				optimizer.zero_grad();

				// Do back-propagation, which computes all the gradients
				computedLoss.backward();

				// Adjust the weights using the (newly calculated) gradients
				optimizer.step();
			}
			return loss;
		}

		private static float Validate(Module<Tensor, Tensor> model, DataLoader val)
		{
			model.eval();

			float loss = 0.0f;

			foreach (var data in val)
			{
				// Run model with Tensor
				var prediction = model.forward(data["data"]);

				// Compute the loss
				var computedLoss = CalculateUnetLoss(prediction, data["masks"]);

				loss = computedLoss.ToSingle();

				break;
			}
			return loss;
		}

		private static ITransform AddAugmentations()
		{

			//var hflip = torchvision.transforms.HorizontalFlip();
			var solarize = transforms.RandomSolarize(0.5, 0.5);
			//var gray = transforms.Grayscale(3);
			//var rotate = transforms.Rotate(90);
			//var contrast = transforms.AdjustContrast(1.25);
			//var transform = torchvision.transforms.RandAugment(generator: g);
			//var transform = torchvision.transforms.AugMix(generator: g);
			//var normalize = transforms.Normalize( new double[] { 0.485, 0.456, 0.406 }, new double[] { 0.229, 0.224, 0.225 },
			//	ScalarType.Float32,
			//	device);


			return transforms.Compose(solarize);
		}

		public void OnLossDataAvailable(LossDataAvailableEventArgs e)
		{
			LossDataAvailable?.Invoke(this, e);
		}

		public void OnLogDataAvailable(LogDataAvailableEventArgs e)
		{
			LogDataAvailable?.Invoke(this, e);
		}

	}
}
