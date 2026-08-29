using VisionStudioAI.ML.Training;
using VisionStudioAI.Core.IO;
using VisionStudioAI.Core.Models;
using VisionStudioAI.Core.Services;
using Microsoft.Extensions.Logging;
using System;
using System.IO;
using System.Text.Json;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace VisionStudioAI.ML.IO
{
    internal static class ModelMetaData
    {
        internal static void SaveModelAndMetadata(
            Module<Tensor, Tensor> model,
            IProjectPresenter project,
            IProjectOptionsService projectOptionsService,
            JsonSerializerOptions jsonOptions,
            ILogger<SegmentationTrainingPipeline> logger)
        {
            try
            {
                var (modelPath, jsonSettingsPath) =
                    projectOptionsService.GetModelMetadataFilePaths(projectOptionsService.GetFolderPath(project.ProjectPath, ProjectFolderType.Models));

                model.save(modelPath);

                // Wrapper to save relevant metadata (that may be used for sdk)
                var modelPackage = new SavedModelPackage
                {
                    Settings = project.Project.Settings,
                    NumClasses = project.Project.Features.Count
                };

                var projectJson = JsonSerializer.Serialize(modelPackage, jsonOptions);
                File.WriteAllText(jsonSettingsPath, projectJson);

                logger.LogInformation("Model saved.");
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Error saving model and metadata.");
            }
        }
    }
}
