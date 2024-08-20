using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace AiModels.ModelUtils
{
    /// <summary>
    /// Hyper-parameter
    /// </summary>
    public interface IModelParameter
    {
        int MaxEpochs { get; set; }
        double LearningRate { get; set; }
        int BatchSize { get; set; }
        int Features { get; set; }
        float StopAtLoss { get; set; }
        bool TrainImagesAsGreyscale { get; set; }
        float SplitTrainValidationSet { get; set; }
        ScalarType TrainPrecision { get; set; }
        DeviceType TrainOnDevice { get; set; }
        int FirstFilterSize { get; set; }

    }

}
