using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace VisionStudioAI.ML.Models
{
    /// <summary>
    /// Factory interface to create configured TorchSharp segmentation modules.
    /// </summary>
    public interface ISegmentationModelFactory
    {
        /// <summary>
        /// Model name.
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Creates a segmentation model for the given settings and device.
        /// </summary>
        Module<Tensor, Tensor> Create(int inChannels, int featureCount, SegmentationModelConfig cfg, Device device);
    }
}
