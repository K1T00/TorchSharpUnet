using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using ScottPlot.Plottables;
using Models.SimpleModel;
using Models.UNet;
using Models.Utils;
using Color = ScottPlot.Color;


namespace DLSharp
{
	public partial class MainForm : Form
	{

		private readonly DataLogger plotTrainLoss;
		private readonly DataLogger plotValLoss;


		private static readonly string ProjDir = 
			Directory.GetParent(Directory.GetCurrentDirectory()).Parent?.Parent?.Parent?.FullName + "\\Data";

		private readonly string modelPath = ProjDir +         "\\models\\model.bin";
		private readonly string newModelPath = ProjDir +      "\\models\\model_new.bin";
		private readonly string trainGrabsPath = ProjDir +    "\\train\\grabs\\";
		private readonly string trainMasksPath = ProjDir +    "\\train\\masks\\";
		private readonly string validateGrabsPath = ProjDir + "\\validate\\grabs\\";
		private readonly string validateMasksPath = ProjDir + "\\validate\\masks\\";
		private readonly string heatmapsPath = ProjDir +      "\\validate\\heatmaps\\";


		public MainForm()
		{
			InitializeComponent();

			// Forms plotter
			fmpLoss.Interaction.Disable();
			//fmpLoss.Plot.XLabel("Epochs");
			//fmpLoss.Plot.YLabel("Loss");

			plotTrainLoss = fmpLoss.Plot.Add.DataLogger();
			plotValLoss = fmpLoss.Plot.Add.DataLogger();


			plotTrainLoss.Color = Color.FromColor(System.Drawing.Color.Blue);
			plotValLoss.Color = Color.FromColor(System.Drawing.Color.Brown);

			//plotLogger.Axes.XAxis.Label

			plotTrainLoss.LegendText = "Train";
			plotValLoss.LegendText = "Val";
		}

		private void btnSimpleModel_Click(object sender, EventArgs e)
		{
			var mySimpleModel = new SimpleModelRun();

			mySimpleModel.LossDataAvailable += ProcessLossData;

			mySimpleModel.Run();
		}

		private void btnUNet_Click(object sender, EventArgs e)
		{
			
			var uNetPara = new UNetParameter
			{
				AmtClasses = Convert.ToInt16(txtFeatures.Text),
				MaxEpochs = Convert.ToInt16(txtEpochs.Text),
				BatchSize = Convert.ToInt16(txtBatchSize.Text),
				LearningRate = Convert.ToDouble(txtLearningRate.Text),
				UseSavedModel = ckbUseSavedModel.Checked,
				StopAtLoss = float.Parse(txtStopAtLoss.Text, CultureInfo.InvariantCulture.NumberFormat)
			};

            var cfgParameter = new ConfigParameter
            {
				ModelPath = modelPath,
                NewModelPath = newModelPath,
                TrainGrabsPath = trainGrabsPath,
				TrainMasksPath = trainMasksPath,
				ValidateGrabsPath = validateGrabsPath,
				ValidateMasksPath = validateMasksPath,
				HeatmapsPath = heatmapsPath
            };

			var myUNetModel = new UNetRun();

			myUNetModel.LossDataAvailable += ProcessLossData;
			myUNetModel.LogDataAvailable += ProcessLogData;

			myUNetModel.Run(uNetPara, cfgParameter);

		}

		private void PlotLoss2D(int epoch, float trainLoss, float valLoss)
		{
			plotTrainLoss.Add(epoch, trainLoss);
			plotValLoss.Add(epoch, valLoss);

			fmpLoss.Refresh();
		}

		private void ProcessLossData(object sender, LossDataAvailableEventArgs e)
		{
			PlotLoss2D(e.Epoch, e.TrainLoss, e.ValLoss);
		}

		private void ProcessLogData(object sender, LogDataAvailableEventArgs e)
		{
			txtLog.AppendText(e.LogText + "\r\n");
		}

		private void ckbUseSavedModel_CheckedChanged(object sender, EventArgs e)
		{
			if (ckbUseSavedModel.Checked & !File.Exists(modelPath))
			{
				ckbUseSavedModel.Checked = false;

				MessageBox.Show("No model file found in: " + modelPath);
			}
		}
	}
}
