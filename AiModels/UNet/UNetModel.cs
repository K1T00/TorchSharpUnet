using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace AiModels.UNet
{
    [Obsolete("Works better with BatchNorm, use UNetModelBn instead.", false)]
    public sealed class UNetModel : Module<Tensor, Tensor>
	{
        #region Fields

        // Encoder
        private readonly Module<Tensor, Tensor> e11;
        private readonly Module<Tensor, Tensor> e12;
        private readonly Module<Tensor, Tensor> pool1;

        private readonly Module<Tensor, Tensor> e21;
        private readonly Module<Tensor, Tensor> e22;
        private readonly Module<Tensor, Tensor> pool2;

        private readonly Module<Tensor, Tensor> e31;
        private readonly Module<Tensor, Tensor> e32;
        private readonly Module<Tensor, Tensor> pool3;

        private readonly Module<Tensor, Tensor> e41;
        private readonly Module<Tensor, Tensor> e42;
        private readonly Module<Tensor, Tensor> pool4;

        // Mid
        private readonly Module<Tensor, Tensor> e51;
        private readonly Module<Tensor, Tensor> e52;

        // Decoder
        private readonly Module<Tensor, Tensor> upConv1;
        private readonly Module<Tensor, Tensor> d11;
        private readonly Module<Tensor, Tensor> d12;

        private readonly Module<Tensor, Tensor> upConv2;
        private readonly Module<Tensor, Tensor> d21;
        private readonly Module<Tensor, Tensor> d22;

        private readonly Module<Tensor, Tensor> upConv3;
        private readonly Module<Tensor, Tensor> d31;
        private readonly Module<Tensor, Tensor> d32;

        private readonly Module<Tensor, Tensor> upConv4;
        private readonly Module<Tensor, Tensor> d41;
        private readonly Module<Tensor, Tensor> d42;

        // Output
        private readonly Module<Tensor, Tensor> outConv;

        #endregion

        public UNetModel(Device? device, int inChannels, int outClasses) : base(nameof(UNetModel))
		{
            #region Model

            // Encoder
            e11 = Conv2d(inChannels, 64, 3, 1, 1, device: device);
            e12 = Conv2d(64, 64, 3, 1, 1, device: device);
            pool1 = MaxPool2d(2, 2);

            e21 = Conv2d(64, 128, 3, 1, 1, device: device);
            e22 = Conv2d(128, 128, 3, 1, 1, device: device);
            pool2 = MaxPool2d(2, 2);

            e31 = Conv2d(128, 256, 3, 1, 1, device: device);
            e32 = Conv2d(256, 256, 3, 1, 1, device: device);
            pool3 = MaxPool2d(2, 2);

            e41 = Conv2d(256, 512, 3, 1, 1, device: device);
            e42 = Conv2d(512, 512, 3, 1, 1, device: device);
            pool4 = MaxPool2d(2, 2);

			// Mid
            e51 = Conv2d(512, 1024, 3, 1, 1, device: device);
            e52 = Conv2d(1024, 1024, 3, 1, 1, device: device);

            // Decoder
            upConv1 = ConvTranspose2d(1024, 512, 2, 2, device: device);
            d11 = Conv2d(1024, 512, 3, 1, 1, device: device);
            d12 = Conv2d(512, 512, 3, 1, 1, device: device);

            upConv2 = ConvTranspose2d(512, 256, 2, 2, device: device);
            d21 = Conv2d(512, 256, 3, 1, 1, device: device);
            d22 = Conv2d(256, 256, 3, 1, 1, device: device);

            upConv3 = ConvTranspose2d(256, 128, 2, 2, device: device);
            d31 = Conv2d(256, 128, 3, 1, 1, device: device);
            d32 = Conv2d(128, 128, 3, 1, 1, device: device);

            upConv4 = ConvTranspose2d(128, 64, 2, 2, device: device);
            d41 = Conv2d(128, 64, 3, 1, 1, device: device);
            d42 = Conv2d(64, 64, 3, 1, 1, device: device);

			// Output
            outConv = Conv2d(64, outClasses, 1, device: device);

            #endregion

            RegisterComponents();
		}

		public override Tensor forward(Tensor input)
		{
			// Encoder
			var xe11 = relu(e11.call(input));
			var xe12 = relu(e12.call(xe11));
			var xp1 = pool1.call(xe12);

			var xe21 = relu(e21.call(xp1));
			var xe22 = relu(e22.call(xe21));
			var xp2 = pool2.call(xe22);

			var xe31 = relu(e31.call(xp2));
			var xe32 = relu(e32.call(xe31));
			var xp3 = pool3.call(xe32);

			var xe41 = relu(e41.call(xp3));
			var xe42 = relu(e42.call(xe41));
			var xp4 = pool4.call(xe42);

			// Mid
			var xe51 = relu(e51.call(xp4));
			var xe52 = relu(e52.call(xe51));

			// Decoder
			var xu1 = upConv1.call(xe52);
			var xu11 = cat(new List<Tensor>() { xu1, xe42 }, 1);
			var xd11 = relu(d11.call(xu11));
			var xd12 = relu(d12.call(xd11));

			var xu2 = upConv2.call(xd12);
			var xu22 = cat(new List<Tensor>() { xu2, xe32 }, 1);
			var xd21 = relu(d21.call(xu22));
			var xd22 = relu(d22.call(xd21));

			var xu3 = upConv3.call(xd22);
			var xu33 = cat(new List<Tensor>() { xu3, xe22 }, 1);
			var xd31 = relu(d31.call(xu33));
			var xd32 = relu(d32.call(xd31));

			var xu4 = upConv4.call(xd32);
			var xu44 = cat(new List<Tensor>() { xu4, xe12 }, 1);
			var xd41 = relu(d41.call(xu44));
			var xd42 = relu(d42.call(xd41));

			// Output
			return outConv.call(xd42);
		}

	}
}
