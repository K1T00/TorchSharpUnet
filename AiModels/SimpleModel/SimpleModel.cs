using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace AiModels.SimpleModel
{
	public sealed class SimpleModel : Module<Tensor, Tensor>
	{

        #region Fields
        
        // Layers kept in fields as they rely on (trainable) weights
        private readonly Module<Tensor, Tensor> lin1;
        private readonly Module<Tensor, Tensor> lin2;
		private readonly Module<Tensor, Tensor> lin3;

        #endregion

        public SimpleModel(Device? device, int inputSize) : base(nameof(SimpleModel))
		{
            
            lin1 = Linear(inputSize, 100, true, device: device);
            lin2 = Linear(100, 10, true, device: device);
            lin3 = Linear(10, 1, true, device: device);

            RegisterComponents();

            this.to(device: device);
		}

		// Every model must have a forward function, where the logic of the model is placed
		// Input is usually in batches
		public override Tensor forward(torch.Tensor input)
		{
			using var layer1 = lin1.forward(input);
			using var layer2 = functional.relu(layer1);
			using var layer3 = lin2.forward(layer2);
			using var layer4 = functional.relu(layer3);

			return lin3.forward(layer4); // Return last layer result
		}

		protected override void Dispose(bool disposing)
		{
			if (disposing)
			{
				lin1.Dispose();
				lin2.Dispose();
				lin3.Dispose();
				ClearModules();
			}
			base.Dispose(disposing);
		}

	}
}
