using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace Models.SimpleModel
{
	public class SimpleModel : nn.Module<Tensor, Tensor>
	{
		// Constructor
		public SimpleModel() : base(nameof(SimpleModel))
		{
			RegisterComponents();
		}

		// Constructor to load the model from a file with ":this()" it is guaranteed that it is called after RegisterComponents()
		public SimpleModel(Device device) : this()
		{
			this.to(device);
		}


		// Every model must have a forward function, where the logic of the model is placed
		// Input is usually in batches
		public override Tensor forward(Tensor input)
		{
			using var layer1 = lin1.forward(input);
			using var layer2 = nn.functional.relu(layer1);
			using var layer3 = lin2.forward(layer2);
			using var layer4 = nn.functional.relu(layer3);

			return lin3.forward(layer4); // Return last layer result
		}

		// Layers kept in fields as they rely on (trainable) weights

		// Linear layer that expects a tensor with x input elements and will yield a tensor with y elements
		private nn.Module<Tensor, Tensor> lin1 = nn.Linear(1000, 100, true, CUDA); // y = xA^T+b
		private nn.Module<Tensor, Tensor> lin2 = nn.Linear(100, 10, true, CUDA); // y = xA^T+b
		private nn.Module<Tensor, Tensor> lin3 = nn.Linear(10, 1, true, CUDA); // y = xA^T+b



	}
}
