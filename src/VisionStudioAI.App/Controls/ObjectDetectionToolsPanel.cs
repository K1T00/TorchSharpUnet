using VisionStudioAI.Core.Models;


namespace VisionStudioAI.App.Controls
{
    public partial class ObjectDetectionToolsPanel : UserControl
    {
        public event EventHandler<InteractionMode>? ModeRequested;
        public event EventHandler<int>? ObjectDiameterChanged;

        public BrushMode LastClickedPlaceObjectMode { get; private set; } = BrushMode.None;

        public ObjectDetectionToolsPanel()
        {
            InitializeComponent();
            tbDiameterSize.Value = 50;
            lbDiameterSize.Text = tbDiameterSize.Value.ToString();
        }

        public int DiameterSize
        {
            get { return tbDiameterSize.Value; }
            set
            {
                var clamped = Math.Clamp(value, tbDiameterSize.Minimum, tbDiameterSize.Maximum);
                tbDiameterSize.Value = clamped;
                lbDiameterSize.Text = clamped.ToString();
            }
        }

        public bool DiameterSizeEnabled
        {
            get { return tbDiameterSize.Enabled; }
            set { tbDiameterSize.Enabled = value; }
        }

        private void btnPlaceObject_Click(object sender, EventArgs e)
        {
            ModeRequested?.Invoke(this, InteractionMode.PlaceObject);
        }

        private void btnEraseObject_Click(object sender, EventArgs e)
        {
            ModeRequested?.Invoke(this, InteractionMode.EraseObject);
        }

        public void SetActiveMode(InteractionMode mode)
        {
            btnPlaceObject.BackgroundImage =
                mode == InteractionMode.PlaceObject
                    ? Properties.Resources.PlaceObjectClicked
                    : Properties.Resources.PlaceObjects;

            btnEraseObject.BackgroundImage =
                mode == InteractionMode.EraseObject
                    ? Properties.Resources.EraserClicked
                    : Properties.Resources.Eraser;
        }

        private void tbDiameterSize_ValueChanged(object sender, EventArgs e)
        {
            lbDiameterSize.Text = tbDiameterSize.Value.ToString();
            ObjectDiameterChanged?.Invoke(this, tbDiameterSize.Value);
        }

        private void tbDiameterSize_MouseDown(object sender, MouseEventArgs e)
        {
            LastClickedPlaceObjectMode = BrushMode.MouseDown;
        }

        private void tbDiameterSize_MouseUp(object sender, MouseEventArgs e)
        {
            LastClickedPlaceObjectMode = BrushMode.MouseUp;
            //ObjectDiameterChanged?.Invoke(this, tbDiameterSize.Value);
        }
    }
}
