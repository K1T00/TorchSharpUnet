using VisionStudioAI.Core.Models;

namespace VisionStudioAI.App.Controls
{
    public partial class AnomalyDetectionToolsPanel : UserControl
    {
        public event EventHandler<AnomalyLabel>? ImageLabelRequested;

        public AnomalyDetectionToolsPanel()
        {
            InitializeComponent();
        }

        private void btnSetImageOk_Click(object? sender, EventArgs e)
        {
            ImageLabelRequested?.Invoke(this, AnomalyLabel.Ok);
        }

        private void btnSetImageNok_Click(object? sender, EventArgs e)
        {
            ImageLabelRequested?.Invoke(this, AnomalyLabel.NotOk);
        }
    }
}
