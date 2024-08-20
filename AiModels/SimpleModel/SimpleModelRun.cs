using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Utils;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace AiModels.SimpleModel
{
	public class SimpleModelRun
	{
		private Device? device;

		public void Run(DeviceType deviceType, IProgress<string> logProgress, IProgress<Tuple<int, float, float>> lossProgress, CancellationToken token)
		{
			this.device = new Device(deviceType);

			logProgress.Report(DateTime.Now.ToString("HH:mm:ss tt") + "\r\n" + "Started training simple model on device: " + this.device);

			// Create random Data
            var dataBatch = rand(32, 1000, device: this.device); // Input data
	        var targetBatch = rand(32, 1, device: this.device); // Ground truth

	        // Init model
	        var model = new SimpleModel(this.device, 1000);

	        // Loss function compares the output of the model with the ground truth of the labels
	        var mseLoss = MSELoss();

	        var optimizer = optim.SGD(model.parameters(), 0.01f);

	        //  Add variable learning rate (SGD optimizer has fixed learning rate)
	        var scheduler = optim.lr_scheduler.StepLR(optimizer, 25, 0.95);

            const int epochs = 100;
            var epoch = 1;

            while (epoch < epochs)
            {
	            var loss = 1.0f;

				// Run one batch and compute the loss by comparing the result to the ground truth
				var outputLoss = mseLoss.call(model.call(dataBatch), targetBatch);

                loss = outputLoss.ToSingle();

                // Clear the gradients before doing the back-propagation
                model.zero_grad();

                // Do back-propagation, which computes all the gradients
                outputLoss.backward();

                // Adjust the weights using the gradients
                optimizer.step();
                scheduler.step();

				lossProgress.Report(new Tuple<int, float, float>(epoch, loss, loss));

                epoch++;

		        Thread.Sleep(25);

		        if (token.IsCancellationRequested)
		        {
			        break;
		        }
	        }
            logProgress.Report(DateTime.Now.ToString("HH:mm:ss tt") + "\r\n" + "Training simple model done.");
		}

	}
}
