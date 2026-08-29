using VisionStudioAI.Core.Models;

namespace VisionStudioAI.ML.Models
{
    public interface IModelComplexityConfigProvider
    {
        SegmentationModelConfig GetConfig(ModelComplexity complexity);
    }
}