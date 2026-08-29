using VisionStudioAI.Core.Models;

namespace VisionStudioAI.ML.Models.AnomalyDetection
{
    /// <summary>
    /// Creates anomaly-model presets for the user-facing complexity levels.
    /// Weight initialization is intentionally not part of a complexity preset.
    /// </summary>
    public interface IBinaryAnomalyComplexityConfigProvider
    {
        BinaryAnomalyModelConfig GetConfig(ModelComplexity complexity);
    }
}
