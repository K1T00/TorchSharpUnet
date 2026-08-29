using VisionStudioAI.Core.Services;
using System;
using System.Threading;
using System.Threading.Tasks;

namespace VisionStudioAI.ML.Inference
{
    public interface ISegmentationInferencePipeline
    {
        /// <summary>
        /// Runs the full inference pipeline asynchronously:
        ///   - loads model
        ///   - preprocesses image
        ///   - performs forward pass
        ///   - postprocesses output
        /// </summary>
        Task RunInference(IProjectPresenter project, string modelPath, IProgress<int> progress, CancellationToken ct);
    }
}
