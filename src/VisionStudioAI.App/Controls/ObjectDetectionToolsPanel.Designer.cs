namespace VisionStudioAI.App.Controls
{
    partial class ObjectDetectionToolsPanel
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

        #region Component Designer generated code

        /// <summary> 
        /// Required method for Designer support - do not modify 
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            btnPlaceObject = new Button();
            btnEraseObject = new Button();
            lbDiameterSize = new Label();
            tbDiameterSize = new TrackBar();
            ((System.ComponentModel.ISupportInitialize)tbDiameterSize).BeginInit();
            SuspendLayout();
            // 
            // btnPlaceObject
            // 
            btnPlaceObject.BackColor = Color.White;
            btnPlaceObject.BackgroundImage = Properties.Resources.PlaceObjects;
            btnPlaceObject.BackgroundImageLayout = ImageLayout.Stretch;
            btnPlaceObject.FlatStyle = FlatStyle.Flat;
            btnPlaceObject.Location = new Point(0, 0);
            btnPlaceObject.Name = "btnPlaceObject";
            btnPlaceObject.Size = new Size(42, 42);
            btnPlaceObject.TabIndex = 0;
            btnPlaceObject.UseVisualStyleBackColor = false;
            btnPlaceObject.Click += btnPlaceObject_Click;
            // 
            // btnEraseObject
            // 
            btnEraseObject.BackgroundImage = Properties.Resources.Eraser;
            btnEraseObject.BackgroundImageLayout = ImageLayout.Stretch;
            btnEraseObject.FlatStyle = FlatStyle.Flat;
            btnEraseObject.Location = new Point(0, 48);
            btnEraseObject.Name = "btnEraseObject";
            btnEraseObject.Size = new Size(42, 34);
            btnEraseObject.TabIndex = 1;
            btnEraseObject.Text = "-";
            btnEraseObject.UseVisualStyleBackColor = true;
            btnEraseObject.Click += btnEraseObject_Click;
            // 
            // lbDiameterSize
            // 
            lbDiameterSize.AutoSize = true;
            lbDiameterSize.Font = new Font("Segoe UI Semibold", 14.25F, FontStyle.Bold | FontStyle.Italic, GraphicsUnit.Point, 0);
            lbDiameterSize.Location = new Point(6, 85);
            lbDiameterSize.Name = "lbDiameterSize";
            lbDiameterSize.Size = new Size(32, 25);
            lbDiameterSize.TabIndex = 4;
            lbDiameterSize.Text = "99";
            // 
            // tbDiameterSize
            // 
            tbDiameterSize.Location = new Point(-1, 115);
            tbDiameterSize.Maximum = 150;
            tbDiameterSize.Minimum = 5;
            tbDiameterSize.Name = "tbDiameterSize";
            tbDiameterSize.Orientation = Orientation.Vertical;
            tbDiameterSize.Size = new Size(45, 165);
            tbDiameterSize.TabIndex = 5;
            tbDiameterSize.Value = 50;
            tbDiameterSize.ValueChanged += tbDiameterSize_ValueChanged;
            tbDiameterSize.MouseDown += tbDiameterSize_MouseDown;
            tbDiameterSize.MouseUp += tbDiameterSize_MouseUp;
            // 
            // ObjectDetectionToolsPanel
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            Controls.Add(tbDiameterSize);
            Controls.Add(lbDiameterSize);
            Controls.Add(btnEraseObject);
            Controls.Add(btnPlaceObject);
            Name = "ObjectDetectionToolsPanel";
            Size = new Size(44, 280);
            ((System.ComponentModel.ISupportInitialize)tbDiameterSize).EndInit();
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private Button btnPlaceObject;
        private Button btnEraseObject;
        private Label lbDiameterSize;
        private TrackBar tbDiameterSize;
    }
}
