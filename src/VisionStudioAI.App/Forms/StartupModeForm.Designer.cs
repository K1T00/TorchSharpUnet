namespace VisionStudioAI.App.Forms
{
    partial class StartupModeForm
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
                segmentationHoverImage.Dispose();
                objectDetectionHoverImage.Dispose();
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(StartupModeForm));
            btnSegmentation = new Button();
            btnObjectDetection = new Button();
            btnCancel = new Button();
            btnLoadProject = new Button();
            lblSegmentation = new Label();
            lblObjectDetection = new Label();
            lblAnomalyDetection = new Label();
            btnAnomalyDetection = new Button();
            SuspendLayout();
            // 
            // btnSegmentation
            // 
            btnSegmentation.BackColor = Color.White;
            btnSegmentation.BackgroundImage = Properties.Resources.Segmentation;
            btnSegmentation.BackgroundImageLayout = ImageLayout.Stretch;
            btnSegmentation.FlatStyle = FlatStyle.Flat;
            btnSegmentation.Location = new Point(12, 64);
            btnSegmentation.Name = "btnSegmentation";
            btnSegmentation.Size = new Size(189, 289);
            btnSegmentation.TabIndex = 0;
            btnSegmentation.UseVisualStyleBackColor = false;
            btnSegmentation.Click += btnSegmentation_Click;
            // 
            // btnObjectDetection
            // 
            btnObjectDetection.BackColor = Color.White;
            btnObjectDetection.BackgroundImage = Properties.Resources.ObjectDetection;
            btnObjectDetection.BackgroundImageLayout = ImageLayout.Stretch;
            btnObjectDetection.FlatStyle = FlatStyle.Flat;
            btnObjectDetection.Location = new Point(207, 64);
            btnObjectDetection.Name = "btnObjectDetection";
            btnObjectDetection.Size = new Size(189, 289);
            btnObjectDetection.TabIndex = 1;
            btnObjectDetection.UseVisualStyleBackColor = false;
            btnObjectDetection.Click += btnObjectDetection_Click;
            // 
            // btnCancel
            // 
            btnCancel.Anchor = AnchorStyles.Bottom | AnchorStyles.Right;
            btnCancel.Font = new Font("Segoe UI Semibold", 14.25F, FontStyle.Bold);
            btnCancel.Location = new Point(462, 361);
            btnCancel.Name = "btnCancel";
            btnCancel.Size = new Size(130, 48);
            btnCancel.TabIndex = 2;
            btnCancel.Text = "Cancel";
            btnCancel.UseVisualStyleBackColor = true;
            btnCancel.Click += btnCancel_Click;
            // 
            // btnLoadProject
            // 
            btnLoadProject.Anchor = AnchorStyles.Bottom | AnchorStyles.Left;
            btnLoadProject.Font = new Font("Segoe UI Semibold", 14.25F, FontStyle.Bold);
            btnLoadProject.Location = new Point(12, 361);
            btnLoadProject.Name = "btnLoadProject";
            btnLoadProject.Size = new Size(130, 48);
            btnLoadProject.TabIndex = 3;
            btnLoadProject.Text = "Load";
            btnLoadProject.UseVisualStyleBackColor = true;
            btnLoadProject.Click += btnLoadProject_Click;
            // 
            // lblSegmentation
            // 
            lblSegmentation.AutoSize = true;
            lblSegmentation.Font = new Font("Segoe UI Semibold", 14F, FontStyle.Bold);
            lblSegmentation.Location = new Point(12, 36);
            lblSegmentation.Name = "lblSegmentation";
            lblSegmentation.Size = new Size(132, 25);
            lblSegmentation.TabIndex = 4;
            lblSegmentation.Text = "Segmentation";
            // 
            // lblObjectDetection
            // 
            lblObjectDetection.AutoSize = true;
            lblObjectDetection.Font = new Font("Segoe UI Semibold", 14F, FontStyle.Bold);
            lblObjectDetection.Location = new Point(207, 36);
            lblObjectDetection.Name = "lblObjectDetection";
            lblObjectDetection.Size = new Size(157, 25);
            lblObjectDetection.TabIndex = 5;
            lblObjectDetection.Text = "Object Detection";
            // 
            // lblAnomalyDetection
            // 
            lblAnomalyDetection.AutoSize = true;
            lblAnomalyDetection.Font = new Font("Segoe UI Semibold", 14F, FontStyle.Bold);
            lblAnomalyDetection.Location = new Point(402, 36);
            lblAnomalyDetection.Name = "lblAnomalyDetection";
            lblAnomalyDetection.Size = new Size(178, 25);
            lblAnomalyDetection.TabIndex = 6;
            lblAnomalyDetection.Text = "Anomaly Detection";
            // 
            // btnAnomalyDetection
            // 
            btnAnomalyDetection.BackColor = Color.White;
            btnAnomalyDetection.BackgroundImage = (Image)resources.GetObject("btnAnomalyDetection.BackgroundImage");
            btnAnomalyDetection.BackgroundImageLayout = ImageLayout.Stretch;
            btnAnomalyDetection.FlatStyle = FlatStyle.Flat;
            btnAnomalyDetection.Location = new Point(402, 64);
            btnAnomalyDetection.Name = "btnAnomalyDetection";
            btnAnomalyDetection.Size = new Size(189, 289);
            btnAnomalyDetection.TabIndex = 7;
            btnAnomalyDetection.UseVisualStyleBackColor = false;
            btnAnomalyDetection.Click += btnAnomalyDetection_Click;
            // 
            // StartupModeForm
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(604, 421);
            Controls.Add(btnAnomalyDetection);
            Controls.Add(lblAnomalyDetection);
            Controls.Add(lblObjectDetection);
            Controls.Add(lblSegmentation);
            Controls.Add(btnLoadProject);
            Controls.Add(btnCancel);
            Controls.Add(btnObjectDetection);
            Controls.Add(btnSegmentation);
            Name = "StartupModeForm";
            Text = "Choose project use case";
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private Button btnSegmentation;
        private Button btnObjectDetection;
        private Button btnCancel;
        private Button btnLoadProject;
        private Label lblSegmentation;
        private Label lblObjectDetection;
        private Label lblAnomalyDetection;
        private Button btnAnomalyDetection;
    }
}
