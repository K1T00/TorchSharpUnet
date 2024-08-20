using System.Diagnostics;
using AiModels.UNet;
using AiModels;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Utils;
using static Utils.HelperFunctions;
using static AiModels.ModelUtils.TensorCalculationsHelper;
using System;
using ScottPlot;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Xml.Serialization;
using Newtonsoft.Json;
using System.Text.Json.Serialization;
using System.Text.Json;
using System.Collections;
using System.Drawing;
using System.Text;
using Newtonsoft.Json.Converters;
using static TorchSharp.torchvision;
using NativeOps;


namespace RunAiModel
{

	public partial class MainForm : Form
	{


		public MainForm()
		{
			InitializeComponent();

			const string projPath = @"ProjectPath.txt";

			if (!File.Exists(projPath))
			{
				var fileCreated = File.CreateText(projPath);
				fileCreated.Close();
			}
			var serializer = new Newtonsoft.Json.JsonSerializer();
			serializer.Converters.Add(new JavaScriptDateTimeConverter());
			serializer.NullValueHandling = NullValueHandling.Ignore;
			using var sr = new StreamReader(projPath);
			using var reader = new JsonTextReader(sr);
			tbProjectPath.Text = (string)serializer.Deserialize(reader)!;
		}

		private async void btnRunModel_Click(object sender, EventArgs e)
		{
			btnRunModel.Enabled = false;
			progressBar1.Value = 0;
			pBImageToAnalyze.Image = null;
			pbImageAnalyzeResult.Image = null;

			var runOnDevice = DeviceType.CPU;
			if (rBTrainOnDeviceCuda.Checked) runOnDevice = DeviceType.CUDA;

			// Load parameter from files
			var uNetPara = LoadModelSettings(tbProjectPath.Text);
			var cfgPara = LoadDbSettings(tbProjectPath.Text);

			// Load image, apply down-sampling, and count amount of images in row and column direction
			using var image = Cv2.ImRead(
				Directory.GetFiles(tbProjectPath.Text + "test\\grabs", "*", SearchOption.TopDirectoryOnly)[0], 
				ImreadModes.Unchanged);
			var imageDs = DownSampleImage(image, cfgPara.DownSampling);
			var amtImages = GetAmtImages(imageDs.Height, imageDs.Width, cfgPara.Roi, true);

			pBImageToAnalyze.Image = image.ToBitmap();
			progressBar1.Value = 30;

			// Load model
			var device = new Device(runOnDevice);
			var model = new UNetModelBn(uNetPara);
			using var d = NewDisposeScope();
			model.load(tbProjectPath.Text + @"models\model.bin").to(device: device);
			model.eval();
			no_grad();
			//using var inferenceMode = inference_mode(true);

			progressBar1.Value = 60;

			var sw = new Stopwatch();
			sw.Restart();

			// Get ROI sized batch images as tensor
			Task<Tensor> roiImagesTensor = null;
			await Task.Run(() =>
			{
				try
				{
					var splitImages = SplitImage(imageDs, cfgPara.Roi, true, amtImages.Item1, amtImages.Item2);
					roiImagesTensor = ImageToRoiSizedTensor(splitImages, model, uNetPara);
				}
				catch (Exception exception)
				{
					MessageBox.Show(exception.ToString());
				}
			}).ConfigureAwait(true);
			
			// Get prediction result by using model on tensor
			var prediction = model.call(roiImagesTensor!.Result);
			var predictionNorm = functional.sigmoid(prediction);
			var predictionSplit = TensorTo2DArray(predictionNorm);

			progressBar1.Value = 80;


			// Tensor result to --> choose by user

			// Result to byte array
			if (cbCreateByteArray.Checked)
			{
				// TODO
				//Tensor image = ... result
				// image = (image * 255.0).to(torch.ScalarType.Byte).cpu();

				//var tensFloatArray = predictionNorm.data<float>().ToArray();

				//var byteArray = Encoding.UTF8.GetBytes(System.Text.Json.JsonSerializer.Serialize(tensFloatArray, new JsonSerializerOptions()
				//{
				//	PropertyNamingPolicy = null,
				//	WriteIndented = true,
				//	AllowTrailingCommas = true,
				//	DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
				//}));

				var tensFloatArray = new float[predictionNorm.shape[0] * predictionNorm.shape[1] * predictionNorm.shape[2] * predictionNorm.shape[3]];

				var l = 0;
				foreach (var pred in predictionSplit)
				{
					var predArray = pred[0].data<float>().ToArray();
					predArray.CopyTo(tensFloatArray, l * predArray.Length);
					l++;
				}
				var byteArray = Encoding.UTF8.GetBytes(System.Text.Json.JsonSerializer.Serialize(tensFloatArray, new JsonSerializerOptions()
				{
					PropertyNamingPolicy = null,
					WriteIndented = true,
					AllowTrailingCommas = true,
					DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
				}));

				// Test
				// TODO
			}

			// Result to image
			if (cbCreateMask.Checked | cbCreateHeatmap.Checked)
			{
				var tasks = new Task<Mat>[predictionSplit.Length];
				var t = 0;
				foreach (var pred in predictionSplit)
				{
					tasks[t] = Task.Run(() => TensorToGreyImage(pred[0]));
					t++;
				}
				var resGreyImages = await Task.WhenAll(tasks);

				var mergedGreyImage = MergeImages(resGreyImages, amtImages.Item2, amtImages.Item1);

				if (cbCreateHeatmap.Checked)
				{
					var imageWithBoarder = new Mat();

					Cv2.CopyMakeBorder(imageDs, imageWithBoarder, 0, mergedGreyImage.Height - imageDs.Height, 0,
						mergedGreyImage.Width - imageDs.Width, BorderTypes.Constant, OpenCvSharp.Scalar.Black);

					pbImageAnalyzeResult.Image = 
						ImageToHeatmap(imageWithBoarder, mergedGreyImage, cfgPara.ThresholdHeatmaps)[0, imageDs.Height, 0, imageDs.Width].ToBitmap();
				}
				else
				{
					pbImageAnalyzeResult.Image = mergedGreyImage[0, imageDs.Height, 0, imageDs.Width].ToBitmap();
				}
			}

			progressBar1.Value = 100;
			tBCalcTime.Text = sw.ElapsedMilliseconds.ToString();
			tbAmtBatchImages.Text = (amtImages.Item1 * amtImages.Item2).ToString();


			if (runOnDevice == DeviceType.CUDA)
			{
				LibTorchLoader.EnsureLoaded();

				if (LibTorchLoader.Loaded & cuda_is_available())
				{
					Ops.cuda_empty_cache();
				}
			}

			btnRunModel.Enabled = true;
		}


		private Db LoadDbSettings(string projDir)
		{
			var configParameter = new Db();
			var filePath = projDir + @"models\DbParameter.xml";

			if (File.Exists(filePath))
			{
				var dbConfigSerializer = new XmlSerializer(typeof(Db));
				using var cfgReader = new StreamReader(filePath);
				configParameter = (Db)dbConfigSerializer.Deserialize(cfgReader)!;
			}
			else
			{
				MessageBox.Show("No data base parameter file found.");
			}
			return configParameter;
		}

		private UNetParameter LoadModelSettings(string projDir)
		{
			var modelParameter = new UNetParameter();
			var filePath = projDir + @"models\ModelParameter.xml";

			if (File.Exists(filePath))
			{
				var uNetParameterSerializer = new XmlSerializer(typeof(UNetParameter));
				using var uNetReader = new StreamReader(filePath);
				modelParameter = (UNetParameter)uNetParameterSerializer.Deserialize(uNetReader)!;
			}
			else
			{
				MessageBox.Show("No model parameter file found.");
			}
			return modelParameter;
		}

		private void btnExit_Click(object sender, EventArgs e)
		{
			Application.Exit();
		}

		private void tbProjectPath_KeyDown(object sender, KeyEventArgs e)
		{
			if (e.KeyData != Keys.Enter)
				return;

			if (!tbProjectPath.Text.EndsWith('\\'))
			{
				tbProjectPath.Text += "\\";
			}

			tbProjectPath.BackColor = System.Drawing.Color.Orange;
			Application.DoEvents();
			Thread.Sleep(500);

			var serializer = new Newtonsoft.Json.JsonSerializer();
			serializer.Converters.Add(new JavaScriptDateTimeConverter());
			serializer.NullValueHandling = NullValueHandling.Ignore;
			using var sw = new StreamWriter(@"ProjectPath.txt");
			using var writer = new JsonTextWriter(sw);
			serializer.Serialize(writer, tbProjectPath.Text);

			tbProjectPath.BackColor = System.Drawing.Color.White;
		}
	}
}
