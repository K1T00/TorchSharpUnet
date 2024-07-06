using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace Models.UNet
{
	/// <summary>
	/// https://en.wikipedia.org/wiki/U-Net
	/// 
	/// Input images tensor: [batch size x channels x width x height]
	/// Output masks tensor: [batch size x features x width x height]
	///
	/// Conv2d Output = (W-F+2P)/S+1
	/// W := input volume
	/// F := neuron size
	/// S := stride
	/// P := zero-padding
	/// 
	/// </summary>
	public class UNetModel : nn.Module<Tensor, Tensor>
	{

        // TODO: Test BatchNorm2d
        // Conv2d -> ReLU -> Conv2d -> ReLU -> MaxPool  ===>>  Conv2d -> BN2d -> ReLU -> Conv2d -> BN2d -> ReLU (maybe -> MaxPool)

        private const int InChannels = 1;
        private const int OutClasses = 6;



		// Encoder
		private nn.Module<Tensor, Tensor> e11 = nn.Conv2d(InChannels, 64, 3, 1, 1);
		private nn.Module<Tensor, Tensor> e12 = nn.Conv2d(64, 64, 3, 1, 1);
		private nn.Module<Tensor, Tensor> pool1 = nn.MaxPool2d(2, 2);

		private nn.Module<Tensor, Tensor> e21 = nn.Conv2d(64, 128, 3, 1, 1);
		private nn.Module<Tensor, Tensor> e22 = nn.Conv2d(128, 128, 3, 1, 1);
		private nn.Module<Tensor, Tensor> pool2 = nn.MaxPool2d(2, 2);

		private nn.Module<Tensor, Tensor> e31 = nn.Conv2d(128, 256, 3, 1, 1);
		private nn.Module<Tensor, Tensor> e32 = nn.Conv2d(256, 256, 3, 1, 1);
		private nn.Module<Tensor, Tensor> pool3 = nn.MaxPool2d(2, 2);

		private nn.Module<Tensor, Tensor> e41 = nn.Conv2d(256, 512, 3, 1, 1);
		private nn.Module<Tensor, Tensor> e42 = nn.Conv2d(512, 512, 3, 1, 1);
		private nn.Module<Tensor, Tensor> pool4 = nn.MaxPool2d(2, 2);

		private nn.Module<Tensor, Tensor> e51 = nn.Conv2d(512, 1024, 3, 1, 1);
		private nn.Module<Tensor, Tensor> e52 = nn.Conv2d(1024, 1024, 3, 1, 1);

		// Decoder
		private nn.Module<Tensor, Tensor> upconv1 = nn.ConvTranspose2d(1024, 512, 2, 2);
		private nn.Module<Tensor, Tensor> d11 = nn.Conv2d(1024, 512, 3, 1, 1);
		private nn.Module<Tensor, Tensor> d12 = nn.Conv2d(512, 512, 3, 1, 1);

		private nn.Module<Tensor, Tensor> upconv2 = nn.ConvTranspose2d(512, 256, 2, 2);
		private nn.Module<Tensor, Tensor> d21 = nn.Conv2d(512, 256, 3, 1, 1);
		private nn.Module<Tensor, Tensor> d22 = nn.Conv2d(256, 256, 3, 1, 1);

		private nn.Module<Tensor, Tensor> upconv3 = nn.ConvTranspose2d(256, 128, 2, 2);
		private nn.Module<Tensor, Tensor> d31 = nn.Conv2d(256, 128, 3, 1, 1);
		private nn.Module<Tensor, Tensor> d32 = nn.Conv2d(128, 128, 3, 1, 1);

		private nn.Module<Tensor, Tensor> upconv4 = nn.ConvTranspose2d(128, 64, 2, 2);
		private nn.Module<Tensor, Tensor> d41 = nn.Conv2d(128, 64, 3, 1, 1);
		private nn.Module<Tensor, Tensor> d42 = nn.Conv2d(64, 64, 3, 1, 1);

		private nn.Module<Tensor, Tensor> outconv = nn.Conv2d(64, OutClasses, 1);

		public UNetModel(Device device) : base(nameof(UNetModel))
		{
			RegisterComponents();

			this.to(device);
		}

		public override Tensor forward(Tensor input)
		{
			// Encoder
			var xe11 = nn.functional.relu(e11.forward(input));
			var xe12 = nn.functional.relu(e12.forward(xe11));
			var xp1 = pool1.forward(xe12);

			var xe21 = nn.functional.relu(e21.forward(xp1));
			var xe22 = nn.functional.relu(e22.forward(xe21));
			var xp2 = pool2.forward(xe22);

			var xe31 = nn.functional.relu(e31.forward(xp2));
			var xe32 = nn.functional.relu(e32.forward(xe31));
			var xp3 = pool3.forward(xe32);

			var xe41 = nn.functional.relu(e41.forward(xp3));
			var xe42 = nn.functional.relu(e42.forward(xe41));
			var xp4 = pool4.forward(xe42);

			var xe51 = nn.functional.relu(e51.forward(xp4));
			var xe52 = nn.functional.relu(e52.forward(xe51));

			// Decoder
			var xu1 = upconv1.forward(xe52);
			var xu11 = cat(new List<Tensor>() { xu1, xe42 }, 1);
			var xd11 = nn.functional.relu(d11.forward(xu11));
			var xd12 = nn.functional.relu(d12.forward(xd11));

			var xu2 = upconv2.forward(xd12);
			var xu22 = cat(new List<Tensor>() { xu2, xe32 }, 1);
			var xd21 = nn.functional.relu(d21.forward(xu22));
			var xd22 = nn.functional.relu(d22.forward(xd21));

			var xu3 = upconv3.forward(xd22);
			var xu33 = cat(new List<Tensor>() { xu3, xe22 }, 1);
			var xd31 = nn.functional.relu(d31.forward(xu33));
			var xd32 = nn.functional.relu(d32.forward(xd31));

			var xu4 = upconv4.forward(xd32);
			var xu44 = cat(new List<Tensor>() { xu4, xe12 }, 1);
			var xd41 = nn.functional.relu(d41.forward(xu44));
			var xd42 = nn.functional.relu(d42.forward(xd41));

			return outconv.forward(xd42); // Return last layer result
		}
	}
}
