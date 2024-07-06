namespace DLSharp
{
	partial class MainForm
	{
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.IContainer components = null;

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		/// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
		protected override void Dispose(bool disposing)
		{
			if (disposing && (components != null))
			{
				components.Dispose();
			}
			base.Dispose(disposing);
		}

		#region Windows Form Designer generated code

		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			btnSimpleModel = new Button();
			btnUNet = new Button();
			fmpLoss = new ScottPlot.WinForms.FormsPlot();
			ckbUseSavedModel = new CheckBox();
			label1 = new Label();
			label2 = new Label();
			label3 = new Label();
			txtEpochs = new TextBox();
			txtLearningRate = new TextBox();
			txtBatchSize = new TextBox();
			txtLog = new TextBox();
			txtFeatures = new TextBox();
			label4 = new Label();
			txtStopAtLoss = new TextBox();
			label5 = new Label();
			SuspendLayout();
			// 
			// btnSimpleModel
			// 
			btnSimpleModel.Enabled = false;
			btnSimpleModel.Font = new Font("Segoe UI", 12F, FontStyle.Regular, GraphicsUnit.Point, 0);
			btnSimpleModel.Location = new Point(12, 21);
			btnSimpleModel.Name = "btnSimpleModel";
			btnSimpleModel.Size = new Size(86, 84);
			btnSimpleModel.TabIndex = 0;
			btnSimpleModel.Text = "Train simple model";
			btnSimpleModel.UseVisualStyleBackColor = true;
			btnSimpleModel.Click += btnSimpleModel_Click;
			// 
			// btnUNet
			// 
			btnUNet.Font = new Font("Segoe UI", 18F, FontStyle.Bold, GraphicsUnit.Point, 0);
			btnUNet.Location = new Point(127, 21);
			btnUNet.Name = "btnUNet";
			btnUNet.Size = new Size(177, 125);
			btnUNet.TabIndex = 1;
			btnUNet.Text = "Train/Run UNet model";
			btnUNet.UseVisualStyleBackColor = true;
			btnUNet.Click += btnUNet_Click;
			// 
			// fmpLoss
			// 
			fmpLoss.DisplayScale = 1F;
			fmpLoss.Location = new Point(12, 272);
			fmpLoss.Name = "fmpLoss";
			fmpLoss.Size = new Size(568, 460);
			fmpLoss.TabIndex = 2;
			// 
			// ckbUseSavedModel
			// 
			ckbUseSavedModel.AutoSize = true;
			ckbUseSavedModel.Font = new Font("Segoe UI Semibold", 14.25F, FontStyle.Bold);
			ckbUseSavedModel.Location = new Point(337, 12);
			ckbUseSavedModel.Name = "ckbUseSavedModel";
			ckbUseSavedModel.Size = new Size(175, 29);
			ckbUseSavedModel.TabIndex = 3;
			ckbUseSavedModel.Text = "Use saved model";
			ckbUseSavedModel.UseVisualStyleBackColor = true;
			ckbUseSavedModel.CheckedChanged += ckbUseSavedModel_CheckedChanged;
			// 
			// label1
			// 
			label1.AutoSize = true;
			label1.Font = new Font("Segoe UI Semibold", 14.25F, FontStyle.Bold);
			label1.Location = new Point(356, 50);
			label1.Name = "label1";
			label1.Size = new Size(77, 25);
			label1.TabIndex = 4;
			label1.Text = "Epochs:";
			// 
			// label2
			// 
			label2.AutoSize = true;
			label2.Font = new Font("Segoe UI Semibold", 14.25F, FontStyle.Bold);
			label2.Location = new Point(331, 83);
			label2.Name = "label2";
			label2.Size = new Size(102, 25);
			label2.TabIndex = 5;
			label2.Text = "Batch size:";
			// 
			// label3
			// 
			label3.AutoSize = true;
			label3.Font = new Font("Segoe UI Semibold", 14.25F, FontStyle.Bold);
			label3.Location = new Point(307, 117);
			label3.Name = "label3";
			label3.Size = new Size(130, 25);
			label3.TabIndex = 6;
			label3.Text = "Learning rate:";
			// 
			// txtEpochs
			// 
			txtEpochs.Font = new Font("Segoe UI Semibold", 14.25F, FontStyle.Bold, GraphicsUnit.Point, 0);
			txtEpochs.Location = new Point(439, 47);
			txtEpochs.Name = "txtEpochs";
			txtEpochs.Size = new Size(73, 33);
			txtEpochs.TabIndex = 9;
			txtEpochs.Text = "5";
			// 
			// txtLearningRate
			// 
			txtLearningRate.Font = new Font("Segoe UI Semibold", 14.25F, FontStyle.Bold, GraphicsUnit.Point, 0);
			txtLearningRate.Location = new Point(439, 114);
			txtLearningRate.Name = "txtLearningRate";
			txtLearningRate.Size = new Size(73, 33);
			txtLearningRate.TabIndex = 10;
			txtLearningRate.Text = "1e-3";
			// 
			// txtBatchSize
			// 
			txtBatchSize.Font = new Font("Segoe UI Semibold", 14.25F, FontStyle.Bold, GraphicsUnit.Point, 0);
			txtBatchSize.Location = new Point(439, 80);
			txtBatchSize.Name = "txtBatchSize";
			txtBatchSize.Size = new Size(73, 33);
			txtBatchSize.TabIndex = 11;
			txtBatchSize.Text = "30";
			// 
			// txtLog
			// 
			txtLog.Font = new Font("Segoe UI Semibold", 12F, FontStyle.Bold, GraphicsUnit.Point, 0);
			txtLog.Location = new Point(586, 12);
			txtLog.Multiline = true;
			txtLog.Name = "txtLog";
			txtLog.ScrollBars = ScrollBars.Vertical;
			txtLog.Size = new Size(354, 705);
			txtLog.TabIndex = 12;
			// 
			// txtFeatures
			// 
			txtFeatures.Font = new Font("Segoe UI Semibold", 14.25F, FontStyle.Bold, GraphicsUnit.Point, 0);
			txtFeatures.Location = new Point(439, 153);
			txtFeatures.Name = "txtFeatures";
			txtFeatures.Size = new Size(73, 33);
			txtFeatures.TabIndex = 14;
			txtFeatures.Text = "6";
			// 
			// label4
			// 
			label4.AutoSize = true;
			label4.Font = new Font("Segoe UI Semibold", 14.25F, FontStyle.Bold);
			label4.Location = new Point(343, 156);
			label4.Name = "label4";
			label4.Size = new Size(90, 25);
			label4.TabIndex = 13;
			label4.Text = "Features:";
			// 
			// txtStopAtLoss
			// 
			txtStopAtLoss.Font = new Font("Segoe UI Semibold", 14.25F, FontStyle.Bold, GraphicsUnit.Point, 0);
			txtStopAtLoss.Location = new Point(439, 192);
			txtStopAtLoss.Name = "txtStopAtLoss";
			txtStopAtLoss.Size = new Size(73, 33);
			txtStopAtLoss.TabIndex = 16;
			txtStopAtLoss.Text = "0.35";
			// 
			// label5
			// 
			label5.AutoSize = true;
			label5.Font = new Font("Segoe UI Semibold", 14.25F, FontStyle.Bold);
			label5.Location = new Point(306, 195);
			label5.Name = "label5";
			label5.Size = new Size(127, 25);
			label5.TabIndex = 15;
			label5.Text = "Stop at loss <";
			// 
			// MainForm
			// 
			AutoScaleDimensions = new SizeF(7F, 15F);
			AutoScaleMode = AutoScaleMode.Font;
			ClientSize = new Size(952, 729);
			Controls.Add(txtStopAtLoss);
			Controls.Add(label5);
			Controls.Add(txtFeatures);
			Controls.Add(label4);
			Controls.Add(txtLog);
			Controls.Add(txtBatchSize);
			Controls.Add(txtLearningRate);
			Controls.Add(txtEpochs);
			Controls.Add(label3);
			Controls.Add(label2);
			Controls.Add(label1);
			Controls.Add(ckbUseSavedModel);
			Controls.Add(fmpLoss);
			Controls.Add(btnUNet);
			Controls.Add(btnSimpleModel);
			Name = "MainForm";
			Text = "MainForm";
			ResumeLayout(false);
			PerformLayout();
		}

		#endregion

		private Button btnSimpleModel;
		private Button btnUNet;
		private ScottPlot.WinForms.FormsPlot fmpLoss;
		private CheckBox ckbUseSavedModel;
		private Label label1;
		private Label label2;
		private Label label3;
		private TextBox txtEpochs;
		private TextBox txtLearningRate;
		private TextBox txtBatchSize;
		private TextBox txtLog;
		private TextBox txtFeatures;
		private Label label4;
		private TextBox txtStopAtLoss;
		private Label label5;
	}
}