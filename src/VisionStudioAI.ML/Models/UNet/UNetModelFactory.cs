using VisionStudioAI.Core.Models;
using System;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace VisionStudioAI.ML.Models.UNet
{
    public class UNetModelFactory : ISegmentationModelFactory
    {
        public string Name
        {
            get { return "SegmentationModel"; }
        }

        public Module<Tensor, Tensor> Create(int inChannels, int featureCount, SegmentationModelConfig cfg, Device device)
        {
            switch (cfg.Architecture)
            {
                case SegmentationArchitecture.UNet:
                    return new UNetFamilyModel(inChannels, featureCount, cfg, device);

                case SegmentationArchitecture.UNetPlusPlus:
                    return new UNetPlusPlusFamilyModel(inChannels, featureCount, cfg, device);

                default:
                    throw new ArgumentOutOfRangeException(nameof(cfg.Architecture), cfg.Architecture, "Unknown segmentation architecture.");
            }
        }
    }

    public static class UNetModelFactorySdk
    {
        public static string Name
        {
            get { return "SegmentationModel"; }
        }

        public static Module<Tensor, Tensor> Create(int inChannels, int featureCount, SegmentationModelConfig cfg, Device device)
        {
            switch (cfg.Architecture)
            {
                case SegmentationArchitecture.UNet:
                    return new UNetFamilyModel(inChannels, featureCount, cfg, device);

                case SegmentationArchitecture.UNetPlusPlus:
                    return new UNetPlusPlusFamilyModel(inChannels, featureCount, cfg, device);

                default:
                    throw new ArgumentOutOfRangeException(nameof(cfg.Architecture), cfg.Architecture, "Unknown segmentation architecture.");
            }
        }
    }
}
