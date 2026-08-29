using VisionStudioAI.ML;
using VisionStudioAI.App.Forms;
using VisionStudioAI.Core;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;


namespace VisionStudioAI.App
{
    public static class ServiceRegistration
    {
        public static IServiceProvider ConfigureServices()
        {
            var services = new ServiceCollection();

            // Logging
            services.AddLogging(builder =>
            {
                builder.SetMinimumLevel(LogLevel.Information);
            });

            // Add modules
            services.AddCoreServices();
            services.AddAiServices();

            // Forms
            services.AddTransient<MainForm>();
            services.AddTransient<TrainingForm>();
            services.AddTransient<InferenceForm>();
            services.AddTransient<TrainedModelsForm>();
            services.AddTransient<StartupModeForm>();

            // DI can resolve Func<Form> -> () => sp.GetRequiredService<Form>()
            services.AddTransient<Func<TrainingForm>>(sp => () => sp.GetRequiredService<TrainingForm>());
            services.AddTransient<Func<InferenceForm>>(sp => () => sp.GetRequiredService<InferenceForm>());
            services.AddTransient<Func<TrainedModelsForm>>(sp => () => sp.GetRequiredService<TrainedModelsForm>());
            services.AddTransient<Func<StartupModeForm>>(sp => () => sp.GetRequiredService<StartupModeForm>());

            return services.BuildServiceProvider();
        }
    }
}
