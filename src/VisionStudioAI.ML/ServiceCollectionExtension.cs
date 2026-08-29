using VisionStudioAI.ML.Inference;
using VisionStudioAI.ML.Models;
using VisionStudioAI.ML.Models.UNet;
using VisionStudioAI.ML.Training;
using VisionStudioAI.ML.Inference.AnomalyDetection;
using VisionStudioAI.ML.Models.AnomalyDetection;
using VisionStudioAI.ML.Training.AnomalyDetection;
using Microsoft.Extensions.DependencyInjection;

namespace VisionStudioAI.ML
{
    public static class ServiceCollectionExtensions
    {
        public static IServiceCollection AddAiServices(this IServiceCollection services)
        {
            // Stateless configuration/services can stay singleton.
            services.AddSingleton<IModelComplexityConfigProvider, ModelComplexityConfigProvider>();
            services.AddSingleton<IBinaryAnomalyComplexityConfigProvider, BinaryAnomalyComplexityConfigProvider>();
            services.AddSingleton<ISegmentationModelFactory, UNetModelFactory>();
            services.AddSingleton<IBinaryAnomalyModelFactory, BinaryAnomalyModelFactory>();

            // Runtime orchestration classes should not be singletons because they coordinate per-run state.
            services.AddTransient<SegmentationTrainingPipeline>();
            services.AddTransient<ISegmentationInferencePipeline, SegmentationInferencePipeline>();
            services.AddTransient<IBinaryAnomalyInferencePipeline, BinaryAnomalyInferencePipeline>();
            services.AddTransient<BinaryAnomalyTrainer>();
            services.AddTransient<SimilarityMemoryBankBuilder>();
            services.AddTransient<AnomalyCalibrationBuilder>();
            services.AddTransient<BinaryAnomalyTrainingPipeline>();

            return services;
        }
    }
}
