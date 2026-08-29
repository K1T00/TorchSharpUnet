using System;
using System.Threading;
using System.Threading.Tasks;
using VisionStudioAI.Core.Services;

namespace VisionStudioAI.ML.Inference.AnomalyDetection
{
    public interface IBinaryAnomalyInferencePipeline
    {
        Task RunInference(
            IProjectPresenter project,
            string artifactDirectory,
            IProgress<int> progress,
            CancellationToken cancellationToken);
    }
}
