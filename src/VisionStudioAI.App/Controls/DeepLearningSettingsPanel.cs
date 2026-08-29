using VisionStudioAI.Core.Models;
using VisionStudioAI.Core.Services;

namespace VisionStudioAI.App.Controls
{

    public partial class DeepLearningSettingsPanel : UserControl
    {
        public event EventHandler? SettingsChanged;
        private IProjectPresenter? presenter; // ToDo: "?" not correct here

        public DeepLearningSettingsPanel()
        {
            InitializeComponent();

            // When the panel's handle is created, we can safely tweak layout if needed
            this.HandleCreated += (s, e) =>
            {
                if (pgDeepLearningSettings.SelectedObject != null)
                {
                    ExpandAllGridItems();
                    AdjustPropertyGridSplitter();
                    SetPropertyGridDescriptionHeight(24);
                }
            };
        }

        public bool ForceCpuOnly { get; set; } = false;

        public bool ForceSliceSize { get; set; } = false;

        public int CurrentObjectDiameter { get; set; } = 0;

        public void Initialize(IProjectPresenter presenter)
        {
            this.presenter = presenter;
            pgDeepLearningSettings.PropertyValueChanged += PgDeepLearningSettings_PropertyValueChanged;
            RefreshBindings();
            pgDeepLearningSettings.ExpandAllGridItems();
        }

        private void PgDeepLearningSettings_PropertyValueChanged(object? s, PropertyValueChangedEventArgs e)
        {
            if (presenter == null)
                return;

            var propertyName = e!.ChangedItem!.PropertyDescriptor!.Name!;

            if (ForceCpuOnly && propertyName == "Device")
            {
                presenter.Project.Settings.TrainModelSettings.Device = ComputeDevice.Cpu;

                // Snap UI back
                pgDeepLearningSettings.Refresh();

                MessageBox.Show("GPU acceleration is disabled on this system because CUDA is not available.", "Cuda not available", MessageBoxButtons.OK, MessageBoxIcon.Information);

                return; // do not propagate change to SettingsChanged
            }

            if (ForceSliceSize && (propertyName == "SliceSize" || propertyName == "DownSample"))
            {
                UpdateAutoSliceSizeFromObjectDiameter();

                return; // do not propagate change to SettingsChanged
            }

            SettingsChanged?.Invoke(this, EventArgs.Empty);
        }

        public static int GetAutoSliceSize(int objectDiameter, int downSample, Size imageSize)
        {
            int[] allowedSizes = { 48, 64, 96, 128, 160, 192, 224, 256 }; // Check TypeConverters.cs

            var effectiveDiameter = objectDiameter * (1 << downSample);
            var targetInOriginalSpace = effectiveDiameter * 4;
            
            var workingWidth = Math.Max(1, imageSize.Width >> downSample);
            var workingHeight = Math.Max(1, imageSize.Height >> downSample);

            // Don't let one slice consume almost the whole image.
            var maxReasonableSlice = Math.Max(64, Math.Min(workingWidth, workingHeight) / 2);

            foreach (var size in allowedSizes)
            {
                if (targetInOriginalSpace <= (size << downSample) && size <= maxReasonableSlice)
                    return size;
            }

            return allowedSizes.Where(size => size <= maxReasonableSlice).DefaultIfEmpty(64).Max();
        }

        public void RefreshBindings()
        {
            if (presenter?.Project?.Settings == null)
            {
                pgDeepLearningSettings.SelectedObject = null;
                return;
            }

            pgDeepLearningSettings.SelectedObject = presenter.Project.Settings;

            if (ForceCpuOnly)
            {
                presenter.Project.Settings.TrainModelSettings.Device = ComputeDevice.Cpu;
            }

            pgDeepLearningSettings.Refresh();

            if (this.IsHandleCreated)
            {
                ExpandAllGridItems();
                AdjustPropertyGridSplitter();
                SetPropertyGridDescriptionHeight(240);
            }
            else
            {
                // If handle isn't created yet, defer once
                void handler(object? s, EventArgs e)
                {
                    this.HandleCreated -= handler;

                    if (pgDeepLearningSettings.SelectedObject != null)
                    {
                        ExpandAllGridItems();
                        AdjustPropertyGridSplitter();
                        SetPropertyGridDescriptionHeight(240);
                    }
                }

                this.HandleCreated += handler;
            }
        }

        /// <summary>
        /// Expands all categories/items in the PropertyGrid.
        /// </summary>
        private void ExpandAllGridItems()
        {
            var root = pgDeepLearningSettings.SelectedGridItem;
            if (root == null)
                return;

            foreach (GridItem child in root.GridItems)
            {
                ExpandRecursive(child);
            }
        }

        private static void ExpandRecursive(GridItem item)
        {
            try { item.Expanded = true; } catch { }

            foreach (GridItem child in item.GridItems)
            {
                ExpandRecursive(child);
            }
        }

        private void AdjustPropertyGridSplitter()
        {
            if (!this.IsHandleCreated)
                return;

            this.BeginInvoke(new Action(() =>
            {
                var gridView = pgDeepLearningSettings.Controls
                    .Cast<Control>()
                    .FirstOrDefault(c => c.GetType().Name == "PropertyGridView");

                if (gridView == null)
                    return;

                const int desiredPosition = 220;

                var moveSplitter = gridView.GetType().GetMethod(
                    "MoveSplitterTo",
                    System.Reflection.BindingFlags.Instance |
                    System.Reflection.BindingFlags.NonPublic);

                moveSplitter?.Invoke(gridView, new object[] { desiredPosition });
            }));
        }

        public void UpdateAutoSliceSizeFromObjectDiameter()
        {
            if (presenter == null || !ForceSliceSize)
                return;

            var limitingImageSize = presenter.Project.Images
                .Select(img => img.ImageSize)   // or img.ImageSize
                .OrderBy(s => Math.Min(s.Width, s.Height))
                .FirstOrDefault();

            presenter.Project.Settings.PreprocessingSettings.SliceSize = GetAutoSliceSize(CurrentObjectDiameter, presenter.Project.Settings.PreprocessingSettings.DownSample, limitingImageSize);

            pgDeepLearningSettings.Refresh();
            SettingsChanged?.Invoke(this, EventArgs.Empty);
        }

        /// <summary>
        /// Changes the height of the help/description box at the bottom.
        /// // ToDo: This seems to not work as expected. :(
        /// </summary>
        private void SetPropertyGridDescriptionHeight(int height)
        {
            if (!this.IsHandleCreated)
                return;

            this.BeginInvoke(new Action(() =>
            {
                var doc = pgDeepLearningSettings.Controls
                .Cast<Control>()
                .FirstOrDefault(c => c.GetType().Name == "DocComment");

                if (doc == null)
                    return;

                var method = doc.GetType().GetMethod(
                    "SetCommentHeight",
                    System.Reflection.BindingFlags.Instance |
                    System.Reflection.BindingFlags.NonPublic);

                method?.Invoke(doc, new object[] { height });
            }));
        }




    }

}
