using Models.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Models.SimpleModel
{
	public class SimpleModelRun
	{

		public event EventHandler<LossDataAvailableEventArgs>? LossDataAvailable;


		public async void Run()
		{

			Device device = cuda.is_available() ? device = CUDA : device = CPU;

	
			// Create data
			var dataBatch = rand(32, 1000, device: device); // Input data
			var targetBatch = rand(32, 1, device: device); // Ground truth

			// Model
			var model = new SimpleModel();

			// Loss function compares the output from the model with the ground truth of labels
			var MSEloss = MSELoss();


			var learning_rate = 0.01f;
			int epochs = 100;
			int epoch = 1;


			var optimizer = optim.SGD(model.parameters(), learning_rate);
			//  Add variable learning rate (SGD optimizer has fixed learning rate)
			var scheduler = optim.lr_scheduler.StepLR(optimizer, 25, 0.95);

			float loss = 0;

			do
			{
				await Task.Run(() =>
				{

					// Run one batch and compute the loss by comparing the result to the ground truth
					var outputLoss = MSEloss.forward(model.forward(dataBatch), targetBatch);

					loss = outputLoss.ToSingle();

					// Clear the gradients before doing the back-propagation
					model.zero_grad();

					// Do back-propagation, which computes all the gradients
					outputLoss.backward();

					// Adjust the weights using the gradients
					optimizer.step();
					scheduler.step();

					Thread.Sleep(100);

				}).ConfigureAwait(true);

				OnLossDataAvailable(new LossDataAvailableEventArgs(epoch, loss, loss));

				epoch++;

			} while (epoch < epochs);

		}

		public void OnLossDataAvailable(LossDataAvailableEventArgs e)
		{
			LossDataAvailable?.Invoke(this, e);
		}
	}
}
