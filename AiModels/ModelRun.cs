using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static AiModels.ModelUtils.TensorCalculationsHelper;
using static Utils.HelperFunctions;
using Utils;
using AiModels.ModelUtils;

namespace AiModels
{

    public class ModelRun
    {
        private Device? device;

        public async Task Run(
            Module<Tensor, Tensor> model,
            Db cfgPara,
            IModelParameter uNetPara,
            IProgress<string> logProgress,
            IProgress<int> barUpdateProgress,
            CancellationToken token)
        {
            using var image = new Mat();
            device = new Device(DeviceType.CUDA);
            model.load(cfgPara.ModelPath + "model.bin").to(device, false);
            model.eval();
            using var d = NewDisposeScope();
            using var noGrad = no_grad();
            //using var inferenceMode = inference_mode(true);

            logProgress.Report("Running model on " + cfgPara.DatasetImages.Count + " images");

            var i = 0;
            foreach (var dImage in cfgPara.DatasetImages)
            {
                if (uNetPara.TrainImagesAsGreyscale)
                {
                    Cv2.ImRead(dImage.Item1, ImreadModes.Grayscale).CopyTo(image);
                }
                else
                {
                    Cv2.ImRead(dImage.Item1, ImreadModes.Color).CopyTo(image);
                }

                var imageDs = DownSampleImage(image, cfgPara.DownSampling);
                var amtImages = GetAmtImages(imageDs.Height, imageDs.Width, cfgPara.Roi, cfgPara.WithBoarderPadding);
                var splitImages = SplitImage(imageDs, cfgPara.Roi, cfgPara.WithBoarderPadding, amtImages.Item1, amtImages.Item2);
                var roiImagesTensor = await ImageToRoiSizedTensor(splitImages, model, uNetPara);
                var prediction = model.call(roiImagesTensor);
                var predictionSplit = TensorTo2DArray(functional.sigmoid(prediction));

                var tasks = new Task<Mat>[predictionSplit.Length];
                var t = 0;
                foreach (var pred in predictionSplit)
                {
                    tasks[t] = Task.Run(() => TensorToGreyImage(pred[0]));
                    t++;
                }
                var resGreyImages = await Task.WhenAll(tasks);
                var mergedGreyImage = MergeImages(resGreyImages, amtImages.Item2, amtImages.Item1);

                var imageWithBoarder = new Mat();

                Cv2.CopyMakeBorder(imageDs, imageWithBoarder, 0, mergedGreyImage.Height - imageDs.Height, 0,
                    mergedGreyImage.Width - imageDs.Width, BorderTypes.Constant, OpenCvSharp.Scalar.Black);

                Cv2.ImWrite(cfgPara.HeatmapsTestPath + i + ".png",
	                ImageToHeatmap(imageWithBoarder, mergedGreyImage, cfgPara.ThresholdHeatmaps)[0, imageDs.Height, 0, imageDs.Width]);

                i++;

                barUpdateProgress.Report(i == cfgPara.DatasetImages.Count ? 100 : (int)(100.0 / cfgPara.DatasetImages.Count * i));

                d.DisposeEverything();

                if (token.IsCancellationRequested)
                {
                    break;
                }
            }
        }

    }
}
