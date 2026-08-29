namespace VisionStudioAI.App.Controls
{
    partial class AnomalyDetectionToolsPanel
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
            btnSetImageOk = new Button();
            btnSetImageNok = new Button();
            SuspendLayout();
            // 
            // btnSetImageOk
            // 
            btnSetImageOk.BackgroundImage = Properties.Resources.SetImageOk;
            btnSetImageOk.BackgroundImageLayout = ImageLayout.Stretch;
            btnSetImageOk.FlatStyle = FlatStyle.Flat;
            btnSetImageOk.Location = new Point(0, 3);
            btnSetImageOk.Name = "btnSetImageOk";
            btnSetImageOk.Size = new Size(44, 44);
            btnSetImageOk.TabIndex = 0;
            btnSetImageOk.UseVisualStyleBackColor = true;
            btnSetImageOk.Click += btnSetImageOk_Click;
            // 
            // btnSetImageNok
            // 
            btnSetImageNok.BackgroundImage = Properties.Resources.SetImageNok;
            btnSetImageNok.BackgroundImageLayout = ImageLayout.Stretch;
            btnSetImageNok.FlatStyle = FlatStyle.Flat;
            btnSetImageNok.Location = new Point(0, 53);
            btnSetImageNok.Name = "btnSetImageNok";
            btnSetImageNok.Size = new Size(44, 44);
            btnSetImageNok.TabIndex = 1;
            btnSetImageNok.UseVisualStyleBackColor = true;
            btnSetImageNok.Click += btnSetImageNok_Click;
            // 
            // AnomalyDetectionToolsPanel
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            Controls.Add(btnSetImageNok);
            Controls.Add(btnSetImageOk);
            Name = "AnomalyDetectionToolsPanel";
            Size = new Size(44, 280);
            ResumeLayout(false);
        }

        #endregion

        private Button btnSetImageOk;
        private Button btnSetImageNok;
    }
}
