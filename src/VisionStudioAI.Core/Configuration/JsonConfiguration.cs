using Microsoft.Extensions.DependencyInjection;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace VisionStudioAI.Core.Configuration
{
    /// <summary>
    /// Copy/pasted into VisionStudioAI.SDK.InferenceJsonHelper
    /// ToDo: Refactor to avoid code duplication
    /// </summary>
    public static class JsonConfiguration
    {
        public static IServiceCollection AddJsonConfiguration(this IServiceCollection services)
        {
            var jsonOptions = new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
                AllowTrailingCommas = true,
                ReadCommentHandling = JsonCommentHandling.Skip,
                NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals
            };

            // Serialize enums as strings
            jsonOptions.Converters.Add(new JsonStringEnumConverter(JsonNamingPolicy.CamelCase));

            // Custom converters
            // jsonOptions.Converters.Add(new MyCustomConverter());

            services.AddSingleton(jsonOptions);

            return services;
        }
    }
}
