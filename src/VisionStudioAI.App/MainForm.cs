using VisionStudioAI.App.Controls;
using VisionStudioAI.App.Forms;
using VisionStudioAI.App.Interaction;
using VisionStudioAI.App.Rendering;
using VisionStudioAI.Core.Interaction;
using VisionStudioAI.Core.Logging;
using VisionStudioAI.Core.Models;
using VisionStudioAI.Core.Services;
using Microsoft.Extensions.Logging;
using Microsoft.VisualBasic.Devices;
using System.Drawing.Drawing2D;
using VisionStudioAI.ML.Inference.AnomalyDetection;
using static VisionStudioAI.ML.Inference.ObjectDetectionHeatmapPostprocessor;
using static VisionStudioAI.ML.Utils.DatasetStatistics;
using static VisionStudioAI.ML.Utils.CudaOps.NativeTorchCudaOps;
using static VisionStudioAI.Core.Utils.CoreUtils;
using static TorchSharp.torch;


namespace VisionStudioAI.App
{
    public partial class MainForm : Form
    {
        // ===== Dependencies =====

        // Whole project state
        private readonly IProjectPresenter projectPresenter;

        private readonly IImageRuntimeLoader imageRuntimeLoader;
        private readonly ILoggerFactory loggerFactory;
        private readonly Func<TrainingForm> trainingFormFactory;
        private readonly Func<InferenceForm> inferenceFormFactory;
        private readonly Func<TrainedModelsForm> trainedModelsFormFactory;
        private readonly Func<StartupModeForm> startupModeFormFactory;

        // ===== Runtime service/cache =====
        private readonly ImageRepository imagesRepo = new();

        // ===== Rendering + interaction =====
        private Viewport viewport = new Viewport(1f, PointF.Empty);
        private readonly InteractionModeController interactionModeController = new InteractionModeController();
        private readonly BrushController brushController = new BrushController();
        private readonly ViewportController viewportController = new ViewportController();
        private RoiController? roiController;

        // ===== UI state =====
        private Guid currentSelectedImageGuid = Guid.Empty;
        private Feature? currentSelectedFeature;
        private string currentSelectedModelFileName = "";
        private int currentHeatmapThreshold = 50;
        private int currentBrushSize;
        private int currentObjectDiameter;
        private BrushMode lastClickedBrushMode = BrushMode.None;
        private PipelineLoopState currentPipelineLoopState = PipelineLoopState.Annotation;
        private Dictionary<int, Color> currentFeatureColorMap = new Dictionary<int, Color>();
        private List<Feature> currentFeatures = [];
        private readonly string startupProjectPath;
        private readonly ObjectDetectionInteractionController objectDetectionController;

        // ===== Show image based on user setting =====
        private Bitmap? previewBitmap;
        private Guid previewImageGuid = Guid.Empty;
        private int previewDownSamplingLevel = -1;

        private long cpuMemoryBudgetBytes;
        private long gpuMemoryBudgetBytes;
        private const byte overlayAlpha = 128; // 0-255 Annotation overlay alpha

        // TODO: This per-import limit is a simple workaround for the ImageGrid's
        // single-column FlowLayoutPanel layout issue with large live imports.
        // Review or remove it once the grid uses paging or virtualization.
        private const int MaxImagesPerImport = 250;

        private const bool UsePretrainedAnomalyWeights = false;
        private const string PretrainedAnomalyWeightsRelativePath =
            @"Models\Pretrained\resnet18-imagenet1k-v1.dat";

        public MainForm(
            string startupProjectPath,
            AppUseCaseMode currentUseCaseMode,
            IProjectPresenter projectPresenter,
            IImageRuntimeLoader imageRuntimeLoader,
            ILoggerFactory loggerFactory,
            Func<TrainingForm> trainingFormFactory,
            Func<InferenceForm> inferenceFormFactory,
            Func<TrainedModelsForm> trainedModelsFormFactory,
            Func<StartupModeForm> startupModeFormFactory)
        {
            InitializeComponent();

            this.projectPresenter = projectPresenter!;
            this.imageRuntimeLoader = imageRuntimeLoader!;
            this.trainingFormFactory = trainingFormFactory!;
            this.inferenceFormFactory = inferenceFormFactory!;
            this.loggerFactory = loggerFactory!;
            this.trainedModelsFormFactory = trainedModelsFormFactory!;
            this.startupModeFormFactory = startupModeFormFactory!;
            this.startupProjectPath = startupProjectPath;

            // Bind runtime repository so dirty edits are persisted on repository eviction
            if (this.projectPresenter is ProjectPresenter pp)
                pp.BindRepository(this.imagesRepo);

            this.objectDetectionController = new ObjectDetectionInteractionController(this.projectPresenter, this.imagesRepo);

            // Control events
            this.imagesControl.ImageSelected += ImageGridControl_ImageSelected;
            this.imagesControl.ImageAdded += ImageGridControl_ImageAdded;
            this.mainPictureBox.MouseWheel += DisplayPictureBox_MouseWheel;

            this.segmentationToolsControl.ModeRequested += SegmentationAnnotationButton_Clicked;
            this.segmentationToolsControl.BrushSizeChanged += BrushSize_Changed;

            this.objectDetectionToolsControl.ModeRequested += ObjectDetectionAnnotationButton_Clicked;
            this.objectDetectionToolsControl.ObjectDiameterChanged += ObjectDiameter_Changed;

            this.anomalyDetectionToolsControl.ImageLabelRequested += AnomalyImageLabel_Requested;

            this.featuresControl.FeatureSelected += FeaturesControl_FeatureSelected;
            this.cbUseEmptyCudeCache.CheckedChanged += cbUseEmptyCudeCache_CheckedChanged;

            // Presenter events must be wired before any project load can happen.
            this.projectPresenter.ProjectLoaded += Presenter_ProjectLoaded;
            this.projectPresenter.ErrorOccured += Presenter_ErrorOccured;

            // Lightweight control setup.
            this.deepLearningSettingsControl.Initialize(this.projectPresenter);
            this.deepLearningSettingsControl.SettingsChanged += DeepLearningSettingsControl_SettingsChanged;

            // Default to Pan
            this.interactionModeController.SetMode(InteractionMode.Pan);

            // Set use case mode from startup form selection
            this.projectPresenter.Project.UseCaseMode = currentUseCaseMode;
            SyncEmptyCudaCacheSetting();
        }

        private async void MainForm_Shown(object sender, EventArgs e)
        {
            await Task.Yield(); // lets the first paint happen

            deepLearningSettingsControl.RefreshBindings();

            deepLearningSettingsControl.ForceSliceSize =
                projectPresenter.Project.UseCaseMode == AppUseCaseMode.ObjectDetection;

            UpdateAnnotationToolPanelVisibility();
            UpdateButtonsPipeLineLoopState();

            currentObjectDiameter = objectDetectionToolsControl.DiameterSize;
            deepLearningSettingsControl.CurrentObjectDiameter = currentObjectDiameter;

            lblThreshold.Text = "50";

            await InitializeHardwareInfoAsync();

            if (!string.IsNullOrWhiteSpace(startupProjectPath))
                await LoadProjectAsync(startupProjectPath);
        }

        #region Methods

        private int CurrentDownSample =>  projectPresenter.Project.Settings.PreprocessingSettings.DownSample;

        private void RebuildLoadedObjectLocationAnnotations()
        {
            objectDetectionController.RebuildLoadedObjectLocationAnnotations(currentFeatureColorMap, currentObjectDiameter, CurrentDownSample);
            mainPictureBox.Invalidate();
        }

        private async Task InitializeHardwareInfoAsync()
        {
            lblSystemRam.Text = "Checking RAM...";
            lblSystemVram.Text = "Checking GPU...";

            var hardware = await Task.Run(() =>
            {
                var cpuBytes = (long)new ComputerInfo().TotalPhysicalMemory;

                if (cuda.is_available())
                {
                    var gpuBytes = GetCudaVRam();
                    return (cpuBytes, gpuBytes, hasCuda: true);
                }

                return (cpuBytes, gpuBytes: 0L, hasCuda: false);
            });

            cpuMemoryBudgetBytes = hardware.cpuBytes;
            gpuMemoryBudgetBytes = hardware.gpuBytes;

            lblSystemRam.Text =
                $"{Math.Round(hardware.cpuBytes / (1024.0 * 1024.0 * 1024.0), 0)} GB RAM";

            if (hardware.hasCuda)
            {
                lblSystemVram.Text =
                    $"{Math.Round(hardware.gpuBytes / (1024.0 * 1024.0 * 1024.0), 0)} GB VRAM";
            }
            else
            {
                lblSystemVram.Text = "No cuda";
                deepLearningSettingsControl.ForceCpuOnly = true;
            }
        }

        private async Task LoadProjectAsync(string fileName)
        {
            await UnloadCurrentProjectAsync();

            tbProjectPath.Text = fileName;
            projectPresenter.Project.Name = fileName;
            projectPresenter.ProjectPath = Path.GetDirectoryName(fileName);

            ProgressForm? progress = null;

            try
            {
                progress = new ProgressForm
                {
                    Text = "Loading project..."
                };

                progress.Show(this);
                progress.Refresh();

                await Task.Run(() => { projectPresenter.LoadProject(fileName); });
            }
            finally
            {
                progress?.Close();
                progress?.Dispose();
            }

            var allFeatureStats = projectPresenter.Project.Images
                .SelectMany(img => img.SegmentationStats?.Values ?? Enumerable.Empty<SegmentationStats>())
                .ToList();

            inferenceResultsControlAllImages.ClearPlot();
            inferenceResultsControlCurrentImage.ClearPlot();

            deepLearningSettingsControl.ForceSliceSize =
                projectPresenter.Project.UseCaseMode == AppUseCaseMode.ObjectDetection;

            currentPipelineLoopState = PipelineLoopState.Annotation;
            UpdateButtonsPipeLineLoopState();

            var ids = imagesControl.GetImageIds();
            if (ids.Count > 0)
                imagesControl.SelectImage(ids[0]);
        }

        private Bitmap GetMainDisplayImage(ImageRuntime rt)
        {
            if (cbShowDownSampledImage.Checked)
                return rt.FullImage;

            var downSample = projectPresenter.Project.Settings.PreprocessingSettings.DownSample;
            if (downSample <= 0)
                return rt.FullImage;

            if (previewBitmap != null && previewImageGuid == currentSelectedImageGuid && previewDownSamplingLevel == downSample)
            {
                return previewBitmap;
            }

            DisposeDownSamplePreview();

            previewBitmap = CreateDownSamplePreview(rt.FullImage, downSample);
            previewImageGuid = currentSelectedImageGuid;
            previewDownSamplingLevel = downSample;

            return previewBitmap;
        }

        private void cbShowDownSampledImage_CheckedChanged(object sender, EventArgs e)
        {
            DisposeDownSamplePreview();
            mainPictureBox.Invalidate();
        }

        private void cbUseEmptyCudeCache_CheckedChanged(object? sender, EventArgs e)
        {
            SyncEmptyCudaCacheSetting();
        }

        private void SyncEmptyCudaCacheSetting()
        {
            UseEmptyCudaCache = cbUseEmptyCudeCache.Checked;
        }

        private static Bitmap CreateDownSamplePreview(Bitmap source, int downSample)
        {
            var factor = 1 << downSample;

            var reducedWidth = Math.Max(1, source.Width / factor);
            var reducedHeight = Math.Max(1, source.Height / factor);

            var reduced = new Bitmap(reducedWidth, reducedHeight);
            using (var g = Graphics.FromImage(reduced))
            {
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                g.SmoothingMode = SmoothingMode.HighQuality;

                g.DrawImage(source, new Rectangle(0, 0, reducedWidth, reducedHeight));
            }

            var preview = new Bitmap(source.Width, source.Height);
            using (var g = Graphics.FromImage(preview))
            {
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.NearestNeighbor;
                g.PixelOffsetMode = PixelOffsetMode.Half;
                g.SmoothingMode = SmoothingMode.None;

                g.DrawImage(reduced, new Rectangle(0, 0, preview.Width, preview.Height));
            }

            reduced.Dispose();
            return preview;
        }

        private void DisposeDownSamplePreview()
        {
            previewBitmap?.Dispose();
            previewBitmap = null;
            previewImageGuid = Guid.Empty;
            previewDownSamplingLevel = -1;
        }

        private void FeaturesControl_FeatureSelected(object? sender, Feature feature)
        {
            currentSelectedFeature = feature;

            if (currentSelectedImageGuid != Guid.Empty)
                imagesControl.SelectImage(currentSelectedImageGuid);

            mainPictureBox.Invalidate();
        }

        private async void Presenter_ErrorOccured(object? sender, string e)
        {
            if (IsDisposed || Disposing || !IsHandleCreated)
                return;

            try
            {
                await InvokeAsync(() =>
                {
                    if (!IsDisposed && !Disposing)
                        MessageBox.Show(this, e, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                });
            }
            catch (ObjectDisposedException)
            {
            }
            catch (InvalidOperationException)
            {
            }
        }

        private async void Presenter_ProjectLoaded(object? sender, EventArgs e)
        {
            if (IsDisposed || Disposing || !IsHandleCreated)
                return;

            try
            {
                if (InvokeRequired)
                {
                    await InvokeAsync(async cancellationToken =>
                    {
                        await Presenter_ProjectLoadedAsync(cancellationToken);
                    });

                    return;
                }

                await Presenter_ProjectLoadedAsync(CancellationToken.None);
            }
            catch (OperationCanceledException)
            {
            }
            catch (ObjectDisposedException)
            {
            }
            catch (Exception ex)
            {
                MessageBox.Show(this, ex.Message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private async Task Presenter_ProjectLoadedAsync(CancellationToken ct)
        {
            ct.ThrowIfCancellationRequested();

            currentSelectedImageGuid = Guid.Empty;
            mainPictureBox.Invalidate();

            imagesControl.ClearGrid();

            currentFeatures = projectPresenter.Project.Features
                .Select(fd => new Feature { ClassId = fd.ClassId, Name = fd.Name, Argb = fd.Argb })
                .ToList();

            featuresControl.UpdateFeatures(currentFeatures);

            currentFeatureColorMap = currentFeatures
                 .Where(f => f.ClassId != 0)
                 .GroupBy(f => f.ClassId)
                 .ToDictionary(g => g.Key, g => Color.FromArgb(g.First().Argb));

            // Fill image grid
            if (projectPresenter.Project.Images != null)
            {
                foreach (var it in projectPresenter.Project.Images)
                {
                    var imgPath = projectPresenter.ResolveImagePath(it.Guid);
                    var thumb = await imageRuntimeLoader.CreateThumbnailAsync(imgPath, imagesControl.Width);
                    imagesControl.AddImage(it.Guid, thumb);
                    imagesControl.UpdateCategory(it.Guid, it.Split);
                    if (projectPresenter.Project.UseCaseMode == AppUseCaseMode.AnomalyDetection)
                        imagesControl.UpdateAnomalyLabel(it.Guid, it.AnomalyLabel);
                }
            }
            deepLearningSettingsControl.RefreshBindings();

            UpdateAnnotationToolPanelVisibility();

            if (projectPresenter.Project.UseCaseMode == AppUseCaseMode.ObjectDetection)
            {
                currentObjectDiameter = projectPresenter.Project.Settings.ObjectsDiameter;

                objectDetectionToolsControl.DiameterSize = currentObjectDiameter;
                deepLearningSettingsControl.CurrentObjectDiameter = currentObjectDiameter;
                RebuildLoadedObjectLocationAnnotations();
            }
        }

        private void DeepLearningSettingsControl_SettingsChanged(object? sender, EventArgs e)
        {
            // Reset name so user can run same model again
            currentSelectedModelFileName = "";

            if (projectPresenter.Project.UseCaseMode == AppUseCaseMode.ObjectDetection)
            {
                RebuildLoadedObjectLocationAnnotations();
            }
            mainPictureBox.Invalidate();
        }

        private void UpdateTrainResultLabels()
        {
            foreach (var item in projectPresenter.Project.Images)
            {
                imagesControl.UpdateTrainResult(item.Guid, FormatTrainResultText(item));
            }
        }

        private static string FormatTrainResultText(ImageItem item)
        {
            var qualityPercent = GetImageDetectionQualityPercent(item);
            if (!qualityPercent.HasValue)
                return string.Empty;

            return $"{qualityPercent.Value:0}%";
        }

        private static double? GetImageDetectionQualityPercent(ImageItem item)
        {
            var stats = item.SegmentationStats?.Values
                .Where(stat => stat != null && !double.IsNaN(stat.Dice) && !double.IsInfinity(stat.Dice))
                .ToList();

            if (stats == null || stats.Count == 0)
                return null;

            var dicePercent = stats.Average(stat => stat.Dice) * 100.0;
            return Math.Clamp(dicePercent, 0.0, 100.0);
        }

        private void ImageGridControl_ImageAdded(object? sender, Guid e)
        {
            if (e == Guid.Empty)
                return;

            // Auto-select the first added image if nothing is selected yet
            if (currentSelectedImageGuid == Guid.Empty)
                ImageGridControl_ImageSelected(this, e);
        }

        private void mainDisplayPictureBox_Paint(object sender, PaintEventArgs e)
        {
            if (!imagesRepo.TryGetRuntime(currentSelectedImageGuid, out var rt))
                return;

            if (!rt.HasFullImage)
                return;

            var g = e.Graphics;
            g.Clear(Color.Black);
            g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.NearestNeighbor;
            g.PixelOffsetMode = PixelOffsetMode.Half;
            g.SmoothingMode = SmoothingMode.HighSpeed;

            // Original full size image
            var baseImage = GetMainDisplayImage(rt);

            OverlayRenderer.DrawImage(g, baseImage, viewport);

            // Draw the semitransparent annotation or heatmap overlay based on current pipeline state
            switch (currentPipelineLoopState)
            {
                case PipelineLoopState.Annotation:
                    if (projectPresenter.Project.UseCaseMode != AppUseCaseMode.AnomalyDetection)
                        rt.MutateAnnotation(bmp => OverlayRenderer.DrawAnnotation(g, bmp, viewport));
                    break;

                case PipelineLoopState.InferenceResults:
                    if (projectPresenter.Project.UseCaseMode == AppUseCaseMode.ObjectDetection)
                    {
                        rt.MutateAnnotation(bmp => OverlayRenderer.DrawAnnotation(g, bmp, viewport));
                    }
                    else
                    {
                        rt.MutateHeatmap(bmp => OverlayRenderer.DrawHeatmap(g, bmp, viewport));
                    }
                    break;
            }

            if (roiController == null)
                return;

            // Draw the ROI if enabled
            if (interactionModeController.ActiveMode == InteractionMode.Roi)
            {
                OverlayRenderer.DrawRoi(g, roiController, viewport);
            }

            // Draw brush size indicator
            if (lastClickedBrushMode == BrushMode.MouseDown)
            {
                OverlayRenderer.DrawBrushIndicator(g, roiController, viewport, mainPictureBox, currentBrushSize);
            }

            // Draw slice size rectangle only in segmentation mode
            if (projectPresenter.Project.UseCaseMode == AppUseCaseMode.Segmentation ||
                projectPresenter.Project.UseCaseMode == AppUseCaseMode.AnomalyDetection)
            {
                OverlayRenderer.DrawSliceSizeRectangle(
                    g,
                    roiController,
                    viewport,
                    projectPresenter.Project.Settings.PreprocessingSettings.SliceSize,
                    projectPresenter.Project.Settings.PreprocessingSettings.DownSample);
            }
        }

        private void BrushSize_Changed(object? sender, int e)
        {
            currentBrushSize = e;
            lastClickedBrushMode = segmentationToolsControl.LastClickedBrushMode;
            mainPictureBox.Invalidate();
        }

        private void ObjectDiameter_Changed(object? sender, int e)
        {
            currentObjectDiameter = e;
            lastClickedBrushMode = objectDetectionToolsControl.LastClickedPlaceObjectMode;

            deepLearningSettingsControl.CurrentObjectDiameter = currentObjectDiameter;
            deepLearningSettingsControl.UpdateAutoSliceSizeFromObjectDiameter();

            projectPresenter.Project.Settings.ObjectsDiameter = e;

            if (projectPresenter.Project.UseCaseMode == AppUseCaseMode.ObjectDetection)
            {
                RebuildLoadedObjectLocationAnnotations();
            }

            mainPictureBox.Invalidate();
        }

        private void SegmentationAnnotationButton_Clicked(object? sender, InteractionMode requestedMode)
        {
            var current = interactionModeController.ActiveMode;

            // Toggle behavior: clicking active tool returns to Pan
            if (current == requestedMode)
            {
                interactionModeController.SetMode(InteractionMode.Pan);
            }
            else
            {
                interactionModeController.SetMode(requestedMode);
            }
            UpdateInteractionUi();
        }

        private void ObjectDetectionAnnotationButton_Clicked(object? sender, InteractionMode requestedMode)
        {
            var current = interactionModeController.ActiveMode;

            // Toggle behavior: clicking active tool returns to Pan
            if (current == requestedMode)
            {
                interactionModeController.SetMode(InteractionMode.Pan);
            }
            else
            {
                interactionModeController.SetMode(requestedMode);
            }
            UpdateInteractionUi();
        }

        private void AnomalyImageLabel_Requested(object? sender, AnomalyLabel requestedLabel)
        {
            if (projectPresenter.Project.UseCaseMode != AppUseCaseMode.AnomalyDetection)
                return;

            var selectedImageIds = imagesControl.GetSelectedImageIds();
            if (selectedImageIds.Count == 0)
            {
                MessageBox.Show(
                    "Select one or more images before assigning an anomaly label.",
                    "Information",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Information);
                return;
            }

            foreach (var imageId in selectedImageIds)
            {
                if (!projectPresenter.TryGetImageItem(imageId, out var imageItem) || imageItem == null)
                    continue;

                imageItem.AnomalyLabel = requestedLabel;
                imagesControl.UpdateAnomalyLabel(imageId, requestedLabel);
            }
        }

        private async void ImageGridControl_ImageSelected(object? sender, Guid imageGuid)
        {
            if (imageGuid == Guid.Empty)
                return;

            if (!projectPresenter.TryGetImageItem(imageGuid, out var item) || item == null)
                return;

            currentSelectedImageGuid = imageGuid;

            // If user selects image first time it is added to fifo cache, otherwise get from cache
            var rt = await imageRuntimeLoader.EnsureFullImageLoadedAsync(item, imagesRepo, projectPresenter);

            // If project was saved we have to load annotation, masks or heatmaps one time from disk into runtime cache
            if (!string.IsNullOrEmpty(projectPresenter.ProjectPath))
            {
                switch (currentPipelineLoopState)
                {
                    case PipelineLoopState.Annotation:
                        if (projectPresenter.Project.UseCaseMode != AppUseCaseMode.AnomalyDetection &&
                            (!rt.AnnotationLoadedOnce || !rt.MaskLoadedOnce))
                        {
                            await imageRuntimeLoader.EnsureAnnotationAndMaskLoadedAsync(item, imagesRepo, projectPresenter);
                        }
                        break;

                    case PipelineLoopState.InferenceResults:
                        if (projectPresenter.Project.UseCaseMode == AppUseCaseMode.AnomalyDetection)
                        {
                            await EnsureAnomalyHeatmapLoadedAsync(item);
                            imagesControl.UpdateTrainResult(
                                item.Guid,
                                $"{(item.PredictedAnomalyLabel == AnomalyLabel.NotOk ? "NOK" : "OK")} {item.AnomalyScore:P1}");
                            inferenceResultsControlCurrentImage.ShowAnomalyImageResult(
                                item.AnomalyScore,
                                item.PredictedAnomalyLabel,
                                item.AnomalyLabel);
                        }
                        else if (currentSelectedFeature != null)
                        {
                            if (projectPresenter.Project.UseCaseMode == AppUseCaseMode.ObjectDetection)
                            {
                                var locations = await LoadObjectDetectionResultsAsync(
                                    item,
                                    projectPresenter,
                                    tBThreshold.Value);

                                item.InferenceObjectLocations = locations;

                                objectDetectionController.RebuildObjectLocationsInferenceResults(
                                    item,
                                    rt,
                                    currentSelectedFeature,
                                    currentFeatureColorMap,
                                    currentObjectDiameter,
                                    CurrentDownSample,
                                    tBThreshold.Value);
                            }
                            else
                            {
                                await imageRuntimeLoader.EnsureHeatmapLoadedAsync(item, imagesRepo, projectPresenter, currentSelectedFeature.Name, tBThreshold.Value);

                            }
                            var segRes = item.SegmentationStats.Values.ToList();
                            var (macro, micro) = AggregateResults(segRes);

                            if (!IsAllZero(macro))
                                inferenceResultsControlCurrentImage.SegmentationStats = macro;
                        }
                        break;
                    default:
                        break;
                }
            }

            // Initialize ROI controller
            roiController = new RoiController(item.Roi);

            // Reset viewport and show image at 100% zoom, centered
            viewport = new Viewport();
            viewport.FitToView(rt.FullImage.Size, mainPictureBox.ClientSize);

            if (projectPresenter.Project.UseCaseMode == AppUseCaseMode.ObjectDetection)
            {
                if (currentPipelineLoopState == PipelineLoopState.InferenceResults)
                    objectDetectionController.RebuildObjectLocationsInferenceResults(
                        item,
                        rt,
                        currentSelectedFeature,
                        currentFeatureColorMap,
                        currentObjectDiameter,
                        CurrentDownSample,
                        tBThreshold.Value);
                else
                    objectDetectionController.RebuildObjectLocations(
                        item,
                        rt,
                        currentFeatureColorMap,
                        currentObjectDiameter,
                        CurrentDownSample);
            }

            mainPictureBox.Invalidate();
        }

        private void tBThreshold_ValueChanged(object sender, EventArgs e)
        {
            lblThreshold.Text = tBThreshold.Value.ToString();

            if (projectPresenter.Project == null)
                return;

            if (currentPipelineLoopState != PipelineLoopState.InferenceResults)
                return;

            if (currentSelectedImageGuid == Guid.Empty)
                return;

            if (!projectPresenter.TryGetImageItem(currentSelectedImageGuid, out var item) || item == null)
                return;

            projectPresenter.Project.Settings.HeatmapThreshold =
                MapRangeStringToInt(lblThreshold.Text, 0, 100, 0, 255);

            var rt = imagesRepo.GetRuntime(
                item,
                ensureAnnotationData:
                    projectPresenter.Project.UseCaseMode != AppUseCaseMode.AnomalyDetection);

            if (projectPresenter.Project.UseCaseMode == AppUseCaseMode.ObjectDetection)
            {
                objectDetectionController.RebuildObjectLocationsInferenceResults(
                    item,
                    rt,
                    currentSelectedFeature,
                    currentFeatureColorMap,
                    currentObjectDiameter,
                    CurrentDownSample,
                    tBThreshold.Value);

                mainPictureBox.Invalidate();
                return;
            }

            var heatmapName = projectPresenter.Project.UseCaseMode == AppUseCaseMode.AnomalyDetection
                ? AnomalyInferenceResults.HeatmapName
                : currentSelectedFeature?.Name;

            if (string.IsNullOrWhiteSpace(heatmapName))
                return;

            if (!rt.HasHeatmapCertainty(heatmapName))
                return;

            var thrByte = (int)Math.Round(tBThreshold.Value * 255.0 / 100.0);
            rt.SetHeatmapThreshold(thrByte);
            rt.RegenerateHeatmapTurbo(heatmapName);

            mainPictureBox.Invalidate();
        }

        private Task EnsureAnomalyHeatmapLoadedAsync(ImageItem item)
        {
            var thresholdPercent = tBThreshold.Value;
            return Task.Run(() =>
            {
                var runtime = imagesRepo.GetRuntime(item, ensureAnnotationData: false);
                var heatmapPath = Path.Combine(
                    projectPresenter.Paths.MasksHeatmaps,
                    AnomalyInferenceResults.HeatmapName,
                    item.Guid + projectPresenter.Paths.ImagesExt);

                if (!File.Exists(heatmapPath))
                {
                    runtime.ClearHeatmapCertainty(AnomalyInferenceResults.HeatmapName);
                    runtime.ClearHeatmap();
                    return;
                }

                if (!runtime.HasHeatmapCertainty(AnomalyInferenceResults.HeatmapName))
                {
                    var certainty = LoadGrayscalePngToByteArray(
                        heatmapPath,
                        item.ImageSize.Width,
                        item.ImageSize.Height);
                    runtime.SetHeatmapCertainty(
                        AnomalyInferenceResults.HeatmapName,
                        certainty,
                        item.ImageSize.Width,
                        item.ImageSize.Height);
                }

                var thresholdByte = (int)Math.Round(thresholdPercent * 255.0 / 100.0);
                runtime.SetHeatmapThreshold(thresholdByte);
                runtime.RegenerateHeatmapTurbo(AnomalyInferenceResults.HeatmapName);
            });
        }

        private async Task UnloadCurrentProjectAsync()
        {
            if (IsDisposed || Disposing || !IsHandleCreated)
                return;

            await InvokeAsync(() =>
            {
                imagesControl.ClearGrid();
                inferenceResultsControlAllImages.ClearPlot();
                inferenceResultsControlCurrentImage.ClearPlot();
            });

            imagesRepo.Clear();
            currentSelectedImageGuid = Guid.Empty;

            // GC.Collect();
            // GC.WaitForPendingFinalizers();
        }

        #endregion

        #region UI state methods

        private void UpdateButtonsPipeLineLoopState()
        {
            btnAnnotate.BackColor =
                currentPipelineLoopState == PipelineLoopState.Annotation ? Color.Coral : Color.PeachPuff;

            btnTrain.BackColor =
                currentPipelineLoopState == PipelineLoopState.Training ? Color.Coral : Color.PeachPuff;

            btnTrainingResults.BackColor =
                currentPipelineLoopState == PipelineLoopState.InferenceResults ? Color.Coral : Color.PeachPuff;

            UpdateInteractionUi();
        }

        private void UpdateAnnotationToolPanelVisibility()
        {
            var hasImages = imagesControl.GetImageIds().Count > 0;
            var hasFeatures = currentFeatures.Count > 0;
            var toolsShouldBeUsable = hasImages && hasFeatures;
            var isSegmentation = projectPresenter.Project.UseCaseMode == AppUseCaseMode.Segmentation;
            var isObjectDetection = projectPresenter.Project.UseCaseMode == AppUseCaseMode.ObjectDetection;
            var isAnomalyDetection = projectPresenter.Project.UseCaseMode == AppUseCaseMode.AnomalyDetection;

            btnEditFeatures.Visible = !isAnomalyDetection;
            featuresControl.Visible = !isAnomalyDetection;

            segmentationToolsControl.Visible = isSegmentation && toolsShouldBeUsable;
            segmentationToolsControl.Enabled = isSegmentation && toolsShouldBeUsable;

            objectDetectionToolsControl.Visible = isObjectDetection && toolsShouldBeUsable;
            objectDetectionToolsControl.Enabled = isObjectDetection && toolsShouldBeUsable;

            anomalyDetectionToolsControl.Visible = isAnomalyDetection && hasImages;
            anomalyDetectionToolsControl.Enabled = isAnomalyDetection && hasImages;

            switch (projectPresenter.Project.UseCaseMode)
            {
                case AppUseCaseMode.Segmentation:
                    segmentationToolsControl.BringToFront();
                    objectDetectionToolsControl.SendToBack();
                    anomalyDetectionToolsControl.SendToBack();
                    break;
                case AppUseCaseMode.ObjectDetection:
                    objectDetectionToolsControl.BringToFront();
                    segmentationToolsControl.SendToBack();
                    anomalyDetectionToolsControl.SendToBack();
                    break;
                case AppUseCaseMode.AnomalyDetection:
                    anomalyDetectionToolsControl.BringToFront();
                    segmentationToolsControl.SendToBack();
                    objectDetectionToolsControl.SendToBack();
                    break;
                default:
                    break;
            }
            btnToggleRoi.Visible = hasImages;
            btnToggleRoi.Enabled = hasImages;
        }

        private void UpdateInteractionUi()
        {
            var mode = interactionModeController.ActiveMode;

            var annotationEnabled = mode != InteractionMode.Roi;
            var isSegmentation = projectPresenter.Project.UseCaseMode == AppUseCaseMode.Segmentation;
            var isObjectDetection = projectPresenter.Project.UseCaseMode == AppUseCaseMode.ObjectDetection;
            var isAnomalyDetection = projectPresenter.Project.UseCaseMode == AppUseCaseMode.AnomalyDetection;
            var isInferenceResults = currentPipelineLoopState == PipelineLoopState.InferenceResults;

            segmentationToolsControl.Enabled =
                isSegmentation &&
                segmentationToolsControl.Visible &&
                annotationEnabled;

            objectDetectionToolsControl.Enabled =
                isObjectDetection &&
                objectDetectionToolsControl.Visible &&
                annotationEnabled;

            anomalyDetectionToolsControl.Enabled =
                isAnomalyDetection &&
                anomalyDetectionToolsControl.Visible &&
                annotationEnabled;

            objectDetectionToolsControl.DiameterSizeEnabled =
                isObjectDetection &&
                objectDetectionToolsControl.Visible &&
                !isInferenceResults;

            segmentationToolsControl.SetActiveMode(mode);
            objectDetectionToolsControl.SetActiveMode(mode);

            btnToggleRoi.BackgroundImage =
                mode == InteractionMode.Roi
                    ? Properties.Resources.ToggleRoiClicked
                    : Properties.Resources.ToggleRoi;


            //switch (mode)
            //{
            //    case InteractionMode.Paint:
            //        mainDisplayPictureBox.Cursor = Cursors.Cross;
            //        break;

            //    case InteractionMode.Erase:
            //        mainDisplayPictureBox.Cursor = Cursors.No;
            //        break;

            //    case InteractionMode.Roi:
            //        mainDisplayPictureBox.Cursor = Cursors.SizeAll;
            //        break;

            //    case InteractionMode.Pan:
            //        mainDisplayPictureBox.Cursor = Cursors.Hand;
            //        break;

            //    default:
            //        mainDisplayPictureBox.Cursor = Cursors.Default;
            //        break;
            //}
        }

        #endregion

        #region Buttons

        private async void btnAddImages_Click(object sender, EventArgs e)
        {
            (sender as Button)!.Enabled = false;

            try
            {
                using var ofd = new OpenFileDialog();
                ofd.Title = "Add images";
                ofd.Filter = "Images|*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff";
                ofd.Multiselect = true;
                ofd.CheckFileExists = true;
                if (ofd.ShowDialog(this) != DialogResult.OK)
                    return;

                if (ofd.FileNames.Length > MaxImagesPerImport)
                {
                    MessageBox.Show(
                        this,
                        $"You can add a maximum of {MaxImagesPerImport} images at once. " +
                        "Please split the selection into smaller batches.",
                        "Too many images selected",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Information);
                    return;
                }

                int added = 0, failed = 0;
                var beforeCount = projectPresenter.Project.Images.Count;

                foreach (var file in ofd.FileNames)
                {

                    if (string.IsNullOrWhiteSpace(file) || !File.Exists(file))
                    {
                        failed++;
                        continue;
                    }

                    var imageSize = await imageRuntimeLoader.ReadImageSizeAsync(file);
                    var newItem = projectPresenter.AddImage(file, imageSize);

                    if (projectPresenter.Project.Images.Count > beforeCount)
                        added++;
                    if (newItem == null)
                        continue;

                    var thumb = await imageRuntimeLoader.CreateThumbnailAsync(file, imagesControl.Width);

                    imagesControl.AddImage(newItem.Guid, thumb);
                    imagesControl.UpdateCategory(newItem.Guid, newItem.Split);
                    if (projectPresenter.Project.UseCaseMode == AppUseCaseMode.AnomalyDetection)
                        imagesControl.UpdateAnomalyLabel(newItem.Guid, newItem.AnomalyLabel);
                }

                // Force reset so user can run same model again
                currentSelectedModelFileName = "";
                UpdateAnnotationToolPanelVisibility();
                imagesControl.Refresh();

                MessageBox.Show($"{added} images added.");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error adding images: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                (sender as Button)!.Enabled = true;
            }
        }

        private async void btnDeleteImages_Click(object sender, EventArgs e)
        {
            (sender as Button)!.Enabled = false;

            try
            {
                var ids = imagesControl.GetSelectedImageIds();

                if (ids.Count == 0)
                {
                    MessageBox.Show(this, "Select one or more images to delete.", "Nothing selected",
                        MessageBoxButtons.OK, MessageBoxIcon.Information);
                    return;
                }

                var confirm = MessageBox.Show(this,
                    $"Remove {ids.Count} image(s) from the project?\n\n" +
                    "This removes them from the project and clears cached bitmaps/masks.\n",
                    "Confirm delete",
                    MessageBoxButtons.YesNo, MessageBoxIcon.Question, MessageBoxDefaultButton.Button2);

                if (confirm != DialogResult.Yes) return;

                // Dispose thumbnails first and evict runtimes before deleting the
                // project files. This releases any image handles and ensures an
                // eviction cannot recreate an annotation after it was deleted.
                imagesControl.RemoveSelectedImages();

                foreach (var id in ids)
                {
                    if (currentSelectedImageGuid == id)
                        currentSelectedImageGuid = Guid.Empty;

                    imagesRepo.Remove(id);
                    await Task.Run(() => { projectPresenter.RemoveImage(id); });
                }

                if (!string.IsNullOrEmpty(projectPresenter.ProjectPath))
                {
                    await Task.Run(() => { projectPresenter.SaveProject(imagesRepo); });
                }
                mainPictureBox.Invalidate();
                UpdateAnnotationToolPanelVisibility();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error deleting images: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                (sender as Button)!.Enabled = true;
            }
        }

        private void btnEditFeatures_Click(object sender, EventArgs e)
        {
            (sender as Button)!.Enabled = false;

            try
            {
                var oldFeaturesSnapshot = currentFeatures
                    .Select(f => new Feature
                    {
                        Name = f.Name,
                        ClassId = f.ClassId,
                        Argb = f.Argb
                    }).ToList();

                using var editorForm = new FeaturesEditor(currentFeatures);

                if (editorForm.ShowDialog() == DialogResult.OK)
                {
                    currentFeatures = editorForm.GetUpdatedFeatures();
                    featuresControl.UpdateFeatures(currentFeatures);
                    projectPresenter.Project.Features = currentFeatures;

                    var classesRemap = BuildClassRemapByName(oldFeaturesSnapshot, currentFeatures);

                    currentFeatureColorMap = currentFeatures
                        .Where(f => f.ClassId != 0)
                        .GroupBy(f => f.ClassId)
                        .ToDictionary(g => g.Key, g => Color.FromArgb(g.First().Argb));
                }
                UpdateAnnotationToolPanelVisibility();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error editing features: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                (sender as Button)!.Enabled = true;
            }
        }

        private void btnToggleRoi_Click(object sender, EventArgs e)
        {
            (sender as Button)!.Enabled = false;

            try
            {
                if (interactionModeController.ActiveMode == InteractionMode.Roi)
                {
                    interactionModeController.SetMode(InteractionMode.Pan);
                }
                else
                {
                    interactionModeController.SetMode(InteractionMode.Roi);
                }
                UpdateInteractionUi();
                mainPictureBox.Invalidate();
            }
            finally
            {
                (sender as Button)!.Enabled = true;
            }
        }

        private async void btnNewProject_Click(object sender, EventArgs e)
        {
            (sender as Button)!.Enabled = false;

            try
            {
                using var startupForm = startupModeFormFactory();

                if (startupForm.ShowDialog(this) != DialogResult.OK || startupForm.SelectedMode is null)
                    return;

                projectPresenter.Project.UseCaseMode = startupForm.SelectedMode.Value;

                using var sfd = new SaveFileDialog();
                sfd.Filter = "Project JSON|*.json|All files|*.*";
                if (sfd.ShowDialog() != DialogResult.OK)
                    return;

                try
                {
                    if (string.IsNullOrWhiteSpace(sfd.FileName))
                        return;

                    tbProjectPath.Text = sfd.FileName;

                    await UnloadCurrentProjectAsync();

                    await Task.Run(() => { projectPresenter.NewProject(sfd.FileName); });
                    await Task.Run(() => { projectPresenter.LoadProject(sfd.FileName); });

                    currentPipelineLoopState = PipelineLoopState.Annotation;
                    UpdateButtonsPipeLineLoopState();

                    imagesControl.Refresh();
                    mainPictureBox.Invalidate();
                    UpdateAnnotationToolPanelVisibility();

                    MessageBox.Show($"Project created successfully.",
                        "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Failed to create project: {ex.Message}",
                        "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error loading project: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                (sender as Button)!.Enabled = true;
            }
        }

        private async void btnSaveProject_Click(object sender, EventArgs e)
        {
            (sender as Button)!.Enabled = false;

            try
            {
                if (string.IsNullOrWhiteSpace(projectPresenter.ProjectPath))
                {
                    using var sfd = new SaveFileDialog();
                    sfd.Filter = "Project JSON|*.json|All files|*.*";
                    if (sfd.ShowDialog() != DialogResult.OK)
                        return;

                    tbProjectPath.Text = sfd.FileName;
                    projectPresenter.ProjectPath = Path.GetDirectoryName(sfd.FileName);
                    projectPresenter.Project.Name = Path.GetFileNameWithoutExtension(sfd.FileName);
                }

                try
                {
                    await Task.Run(() => { projectPresenter.SaveProject(imagesRepo); });
                    MessageBox.Show("Project saved successfully.", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
                catch (OperationCanceledException)
                {
                    MessageBox.Show("Save canceled.", "Info", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error saving project: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                (sender as Button)!.Enabled = true;
            }
        }

        private async void btnSaveAs_Click(object sender, EventArgs e)
        {
            (sender as Button)!.Enabled = false;

            try
            {
                using var sfd = new SaveFileDialog();
                sfd.Filter = "Project JSON|*.json|All files|*.*";
                if (sfd.ShowDialog() != DialogResult.OK)
                    return;

                try
                {
                    tbProjectPath.Text = sfd.FileName;
                    projectPresenter.ProjectPath = Path.GetDirectoryName(sfd.FileName);
                    projectPresenter.Project.Name = Path.GetFileNameWithoutExtension(sfd.FileName);

                    await Task.Run(() => { projectPresenter.SaveProject(imagesRepo); });
                    MessageBox.Show("Project saved successfully.", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
                catch (OperationCanceledException)
                {
                    MessageBox.Show("Save canceled.", "Info", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error saving project: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                (sender as Button)!.Enabled = true;
            }
        }

        private async void btnLoadProject_Click(object sender, EventArgs e)
        {
            (sender as Button)!.Enabled = false;

            try
            {
                using var ofd = new OpenFileDialog();
                ofd.Title = "Load Project File";
                ofd.Filter = "Project JSON|*.json|All files|*.*";

                if (ofd.ShowDialog() != DialogResult.OK)
                    return;

                await LoadProjectAsync(ofd.FileName);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error loading project: {ex.Message}", "Error",
                    MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                (sender as Button)!.Enabled = true;
            }
        }

        private void btnExit_Click(object sender, EventArgs e)
        {
            (sender as Button)!.Enabled = false;

            try
            {
                var result = MessageBox.Show("Are you sure you want to exit?", "Exit Confirmation",
                    MessageBoxButtons.YesNo, MessageBoxIcon.Question);
                if (result != DialogResult.Yes)
                    return;

                imagesRepo?.Dispose();
                Application.Exit();

            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error exiting application: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                (sender as Button)!.Enabled = true;
            }
        }

        private void btnAnnotate_Click(object sender, EventArgs e)
        {
            (sender as Button)!.Enabled = false;

            try
            {
                currentPipelineLoopState = PipelineLoopState.Annotation;
                UpdateButtonsPipeLineLoopState();

                // Select first image
                var ids = imagesControl.GetImageIds();
                if (ids.Count > 0)
                    imagesControl.SelectImage(ids[0]);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error training images: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                currentPipelineLoopState = PipelineLoopState.Annotation;
                UpdateButtonsPipeLineLoopState();

                (sender as Button)!.Enabled = true;
            }
        }

        private void btnUpdateCategoryToTrain_Click(object sender, EventArgs e)
        {
            (sender as Button)!.Enabled = false;

            try
            {
                if (!imagesControl.HasSelectedImages)
                {
                    MessageBox.Show("No images are selected to categorize.", "Information",
                        MessageBoxButtons.OK, MessageBoxIcon.Information);
                    return;
                }
                foreach (var id in imagesControl.GetSelectedImageIds())
                {
                    if (!projectPresenter.TryGetImageItem(id, out var it))
                        return;

                    it.Split = DatasetSplit.Train;
                    imagesControl.UpdateCategory(id, it.Split);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error setting category to Train: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                (sender as Button)!.Enabled = true;
            }
        }

        private void btnUpdateCategoryToValidate_Click(object sender, EventArgs e)
        {
            (sender as Button)!.Enabled = false;

            try
            {
                if (!imagesControl.HasSelectedImages)
                {
                    MessageBox.Show("No images are selected to categorize.", "Information",
                        MessageBoxButtons.OK, MessageBoxIcon.Information);
                    return;
                }
                foreach (var id in imagesControl.GetSelectedImageIds())
                {
                    if (!projectPresenter.TryGetImageItem(id, out var it))
                        return;

                    it.Split = DatasetSplit.Validate;
                    imagesControl.UpdateCategory(id, it.Split);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error setting category to Validate: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                (sender as Button)!.Enabled = true;
            }
        }

        private void btnUpdateCategoryToTest_Click(object sender, EventArgs e)
        {
            (sender as Button)!.Enabled = false;

            try
            {
                if (!imagesControl.HasSelectedImages)
                {
                    MessageBox.Show("No images are selected to categorize.", "Information",
                        MessageBoxButtons.OK, MessageBoxIcon.Information);
                    return;
                }

                foreach (var id in imagesControl.GetSelectedImageIds())
                {
                    if (!projectPresenter.TryGetImageItem(id, out var it))
                        return;

                    it.Split = DatasetSplit.Test;
                    imagesControl.UpdateCategory(id, it.Split);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error setting category to Test: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                (sender as Button)!.Enabled = true;
            }
        }

        private async void btnTrain_Click(object sender, EventArgs e)
        {
            (sender as Button)!.Enabled = false;

            TrainingForm? trainingForm = null;
            TrainingFormLoggerProvider? loggerProvider = null;
            Task? trainingFormClosedTask = null;
            var trainingStarted = false;

            try
            {
                if (projectPresenter.ProjectPath == null)
                {
                    MessageBox.Show(this, "Please save project first.");
                    return;
                }

                await Task.Run(() => { projectPresenter.SaveProject(imagesRepo); });

                var trainCount = projectPresenter.Project.Images.Count(i => i.Split == DatasetSplit.Train);
                var validateCount = projectPresenter.Project.Images.Count(i => i.Split == DatasetSplit.Validate);

                if (trainCount == 0 || validateCount == 0)
                {
                    MessageBox.Show(
                        this,
                        "At least one train image and one validate image needed.",
                        "Warning",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Error);

                    return;
                }

                if (projectPresenter.Project.UseCaseMode == AppUseCaseMode.AnomalyDetection)
                {
                    var trainingImages = projectPresenter.Project.Images
                        .Where(image => image.Split == DatasetSplit.Train)
                        .ToList();
                    var validationImages = projectPresenter.Project.Images
                        .Where(image => image.Split == DatasetSplit.Validate)
                        .ToList();

                    if (!ContainsAnomalyLabel(trainingImages, AnomalyLabel.Ok) ||
                        !ContainsAnomalyLabel(validationImages, AnomalyLabel.Ok))
                    {
                        MessageBox.Show(
                            this,
                            "OK-only anomaly detection requires at least one OK image " +
                            "in both the Train and Validate categories. NOK validation " +
                            "images are optional and can be used for threshold calibration.",
                            "Warning",
                            MessageBoxButtons.OK,
                            MessageBoxIcon.Error);
                        return;
                    }
                }

                string? anomalyPretrainedWeightsPath = null;
                if (projectPresenter.Project.UseCaseMode == AppUseCaseMode.AnomalyDetection &&
                    UsePretrainedAnomalyWeights)
                {
                    anomalyPretrainedWeightsPath = Path.Combine(AppContext.BaseDirectory, PretrainedAnomalyWeightsRelativePath);

                    if (!File.Exists(anomalyPretrainedWeightsPath))
                    {
                        MessageBox.Show(
                            this,
                            "The pretrained anomaly model could not be found:" +
                            Environment.NewLine + anomalyPretrainedWeightsPath +
                            Environment.NewLine + Environment.NewLine +
                            "Run PyBridgeLoader to download and convert it first.",
                            "Pretrained model missing",
                            MessageBoxButtons.OK,
                            MessageBoxIcon.Error);
                        return;
                    }
                }

                ShowOverlay();

                trainingForm = trainingFormFactory();

                loggerProvider = new TrainingFormLoggerProvider(trainingForm);
                loggerFactory.AddProvider(loggerProvider);

                trainingFormClosedTask = trainingForm.ShowAsync(this);

                trainingStarted = true;
                await trainingForm.StartTrainingRun(
                    projectPresenter,
                    cpuMemoryBudgetBytes,
                    gpuMemoryBudgetBytes,
                    anomalyPretrainedWeightsPath);
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                       this,
                       $"Error training images: {ex.Message}",
                       "Error",
                       MessageBoxButtons.OK,
                       MessageBoxIcon.Error);
            }
            finally
            {
                if (trainingStarted)
                {
                    currentPipelineLoopState = PipelineLoopState.Training;
                    UpdateButtonsPipeLineLoopState();
                }

                (sender as Button)!.Enabled = true;

                if (trainingFormClosedTask != null)
                {
                    try
                    {
                        await trainingFormClosedTask;
                    }
                    catch (ObjectDisposedException)
                    {
                    }
                }

                loggerProvider?.Dispose();
                trainingForm?.Dispose();
                HideOverlay();
            }
        }

        private static bool ContainsAnomalyLabel(
            IEnumerable<ImageItem> images,
            AnomalyLabel requiredLabel)
        {
            return images.Any(image => image.AnomalyLabel == requiredLabel);
        }

        private async void btnTrainingResults_Click(object sender, EventArgs e)
        {
            (sender as Button)!.Enabled = false;

            try
            {
                if (projectPresenter.ProjectPath == null)
                {
                    MessageBox.Show("Please save project first.");
                    return;
                }
                else
                {
                    await Task.Run(() => { projectPresenter.SaveProject(imagesRepo); });
                }

                var paths = projectPresenter.Paths;
                var trainedModelsForm = trainedModelsFormFactory();
                trainedModelsForm.Paths = paths;
                trainedModelsForm.UseCaseMode = projectPresenter.Project.UseCaseMode;

                if (trainedModelsForm.ShowDialog(this) != DialogResult.OK ||
                    trainedModelsForm.SelectedModelFileName is not string modelName)
                {
                    currentPipelineLoopState = PipelineLoopState.Annotation;
                    UpdateButtonsPipeLineLoopState();
                    return;
                }

                var isAnomalyDetection =
                    projectPresenter.Project.UseCaseMode == AppUseCaseMode.AnomalyDetection;
                var modelPath = isAnomalyDetection
                    ? Path.Combine(paths.Models, modelName)
                    : Path.ChangeExtension(Path.Combine(paths.Models, modelName), paths.ModelExt);
                var modelSelectionKey = isAnomalyDetection
                    ? Path.GetFullPath(modelPath)
                    : modelName;

                currentPipelineLoopState = PipelineLoopState.InferenceResults;
                UpdateButtonsPipeLineLoopState();

                projectPresenter.Project.Settings.HeatmapThreshold = MapRangeStringToInt(lblThreshold.Text, 0, 100, 0, 255);

                // Selected the same model as last time - just show heatmaps without re-running inference
                if (modelSelectionKey != currentSelectedModelFileName || projectPresenter.Project.Settings.HeatmapThreshold != currentHeatmapThreshold)
                {
                    currentSelectedModelFileName = modelSelectionKey;
                    currentHeatmapThreshold = projectPresenter.Project.Settings.HeatmapThreshold;

                    // Hide MainForms overlay
                    ShowOverlay();

                    TryDeleteDirectoryContents(paths.MasksHeatmaps, out var errOverlays);

                    foreach (var image in projectPresenter.Project.Images)
                    {
                        if (!imagesRepo.TryGetRuntime(image.Guid, out var runtime))
                            continue;

                        runtime.ClearAllHeatmapCertainty();
                        runtime.ClearHeatmap();
                    }

                    var inferenceForm = inferenceFormFactory();

                    inferenceForm.FormClosed += (s, args) => HideOverlay();
                    inferenceForm.Show(this);

                    await inferenceForm.StartInferenceRun(projectPresenter, modelPath);
                }

                // Show statistic results //
                if (projectPresenter.Project.UseCaseMode == AppUseCaseMode.AnomalyDetection)
                {
                    foreach (var image in projectPresenter.Project.Images)
                    {
                        imagesControl.UpdateTrainResult(
                            image.Guid,
                            $"{(image.PredictedAnomalyLabel == AnomalyLabel.NotOk ? "NOK" : "OK")} {image.AnomalyScore:P1}");
                    }

                    inferenceResultsControlAllImages.ShowAnomalyAggregateResults(
                        projectPresenter.Project.Images);
                }
                else
                {
                    if (projectPresenter.Project.UseCaseMode == AppUseCaseMode.Segmentation)
                        UpdateTrainResultLabels();

                    var (macro, micro) = AggregateResults(
                        projectPresenter.Project.Images
                            .SelectMany(img => img.SegmentationStats.Values)
                            .ToList());

                    if (!IsAllZero(macro))
                        inferenceResultsControlAllImages.SegmentationStats = macro;
                }

                if (projectPresenter.Project.Images.Count > 0)
                {
                    // Skip the first image as it often contains outliers (e.g. long inference time due to one-time setup overhead)
                    // that skews the average compute time
                    var imagesForAverage = projectPresenter.Project.Images.Count > 1
                        ? projectPresenter.Project.Images.Skip(1)
                        : projectPresenter.Project.Images;

                    var averageInferenceMs = imagesForAverage.Average(img => img.InferenceMs);

                    lblInferenceMeanComputeTime.Text =
                        $"Average compute time: {Math.Round(averageInferenceMs, 0):F1} ms";
                }

                // Select first feature
                if (projectPresenter.Project.UseCaseMode != AppUseCaseMode.AnomalyDetection &&
                    currentFeatures.Count > 0)
                {
                    featuresControl.SelectFeature(currentFeatures[0]);
                }

                // Select first image
                var ids = imagesControl.GetImageIds();
                if (ids.Count > 0)
                    imagesControl.SelectImage(ids[0]);

                if (projectPresenter.Project.UseCaseMode == AppUseCaseMode.ObjectDetection)
                {
                    objectDetectionToolsControl.DiameterSize = projectPresenter.Project.Settings.ObjectsDiameter;

                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error running inference: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                (sender as Button)!.Enabled = true;
            }
        }

        #endregion

        #region Mouse events on Main PictureBox interactions

        private void mainDisplayPictureBox_MouseDown(object sender, MouseEventArgs e)
        {
            if (e.Button != MouseButtons.Left)
                return;

            if (!imagesRepo.TryGetRuntime(currentSelectedImageGuid, out var rt))
                return;

            switch (interactionModeController.ActiveMode)
            {
                case InteractionMode.Paint:
                    {
                        if (currentSelectedFeature == null)
                        {
                            MessageBox.Show("Please select a feature to paint.");
                            return;
                        }

                        brushController.BeginStroke(e.Location, viewport, segmentationToolsControl.BrushSize, (byte)currentSelectedFeature.ClassId);


                        if (brushController.UpdateStroke(e.Location, viewport, rt.Mask, out var dirtyImageRect))
                        {
                            rt.MutateAnnotation(bmp =>
                            {
                                OverlayRenderer.UpdateAnnotationOverlayRegion(
                                    bmp,
                                    rt.Mask,
                                    currentFeatureColorMap,
                                    dirtyImageRect,
                                    overlayAlpha);
                            });


                            rt.MutateMask(mask =>
                            {
                                var changed = brushController.UpdateStroke(e.Location, viewport, mask, out dirtyImageRect);
                            });



                            var dirtyScreenRect = viewport.ImageToScreenRect(dirtyImageRect);
                            mainPictureBox.Invalidate(new Region(dirtyScreenRect));
                        }
                        break;
                    }
                case InteractionMode.Erase:
                    {
                        brushController.BeginStroke(e.Location, viewport, segmentationToolsControl.BrushSize, (byte)0); // background

                        if (brushController.UpdateStroke(e.Location, viewport, rt.Mask, out var dirtyImageRect))
                        {
                            rt.MutateAnnotation(bmp =>
                            {
                                OverlayRenderer.UpdateAnnotationOverlayRegion(
                                    bmp,
                                    rt.Mask,
                                    currentFeatureColorMap,
                                    dirtyImageRect,
                                    overlayAlpha);
                            });

                            rt.MutateMask(mask =>
                            {
                                var changed = brushController.UpdateStroke(e.Location, viewport, mask, out dirtyImageRect);
                            });

                            var dirtyScreenRect = viewport.ImageToScreenRect(dirtyImageRect);
                            mainPictureBox.Invalidate(new Region(dirtyScreenRect));
                        }
                        break;
                    }
                case InteractionMode.PlaceObject:
                    {
                        var dragResult = 
                            objectDetectionController.TryBeginDrag(currentSelectedImageGuid, e.Location, viewport, currentObjectDiameter, CurrentDownSample);

                        if (dragResult.Applied)
                        {
                            mainPictureBox.Capture = true;
                            return;
                        }

                        var addResult = objectDetectionController.TryAddObjectOnLocation(
                            currentSelectedImageGuid,
                            e.Location,
                            viewport,
                            currentSelectedFeature,
                            currentFeatureColorMap,
                            currentObjectDiameter,
                            CurrentDownSample);

                        if (addResult.Status == ObjectDetectionEditStatus.MissingFeature)
                        {
                            MessageBox.Show("Please select a feature to place.");
                            return;
                        }

                        if (addResult.Applied)
                            mainPictureBox.Invalidate();
                        break;
                    }
                case InteractionMode.EraseObject:
                    {
                        var eraseResult = objectDetectionController.TryEraseObjectOnLocation(
                            currentSelectedImageGuid,
                            e.Location,
                            viewport,
                            currentObjectDiameter,
                            CurrentDownSample,
                            currentFeatureColorMap);

                        if (eraseResult.Applied)
                            mainPictureBox.Invalidate();
                        break;
                    }
                case InteractionMode.Roi:
                    {
                        if (roiController == null)
                            return;

                        roiController.MouseDown(e.Location, viewport);

                        // If ROI did not capture interaction, fall back to pan
                        if (roiController.Mode == RoiMode.None)
                        {
                            viewportController.BeginPan(e.Location);
                            interactionModeController.PushTemporaryMode(InteractionMode.Pan);
                        }

                        mainPictureBox.Invalidate();
                        break;
                    }
                case InteractionMode.Pan:
                    {
                        viewportController.BeginPan(e.Location);
                        mainPictureBox.Invalidate();
                        break;
                    }
                default:
                    break;

            }
        }

        private void mainDisplayPictureBox_MouseMove(object sender, MouseEventArgs e)
        {
            if (e.Button != MouseButtons.Left)
                return;

            if (!imagesRepo.TryGetRuntime(currentSelectedImageGuid, out var rt))
                return;

            switch (interactionModeController.ActiveMode)
            {
                case InteractionMode.Paint:
                case InteractionMode.Erase:
                    
                    if (rt.Mask == null)
                        return;

                    if (brushController.UpdateStroke(e.Location, viewport, rt.Mask, out var dirtyImageRect))
                    {
                        rt.MutateAnnotation(bmp =>
                        {
                            OverlayRenderer.UpdateAnnotationOverlayRegion(
                                bmp,
                                rt.Mask,
                                currentFeatureColorMap,
                                dirtyImageRect,
                                overlayAlpha);
                        });


                        rt.MutateMask(mask =>
                        {
                            var changed = brushController.UpdateStroke(e.Location, viewport, mask, out dirtyImageRect);
                        });

                        var dirtyScreenRect = viewport.ImageToScreenRect(dirtyImageRect);
                        mainPictureBox.Invalidate(new Region(dirtyScreenRect));
                    }
                    break;
                    
                case InteractionMode.PlaceObject:

                    var moveResult = objectDetectionController.TryMoveDraggedObject(
                        currentSelectedImageGuid,
                        e.Location,
                        viewport,
                        currentFeatureColorMap,
                        currentObjectDiameter,
                        CurrentDownSample);

                    if (moveResult.Applied)
                        mainPictureBox.Invalidate();
                    break;
                    
                case InteractionMode.Roi:
                    
                        if (roiController == null)
                            break;

                        roiController.MouseMove(e.Location, viewport, new SizeF(rt.FullImage.Width, rt.FullImage.Height));

                        mainPictureBox.Invalidate();
                        break;
                    
                case InteractionMode.Pan:
                    
                        viewportController.UpdatePan(e.Location, viewport);
                        mainPictureBox.Invalidate();
                        break;
                    
                default:
                    break;
            }
        }

        private void mainDisplayPictureBox_MouseUp(object sender, MouseEventArgs e)
        {
            if (e.Button != MouseButtons.Left)
                return;

            if (!imagesRepo.TryGetRuntime(currentSelectedImageGuid, out var rt))
                return;

            // Finalize brush stroke (paint / erase)
            if (interactionModeController.ActiveMode == InteractionMode.Paint ||
                interactionModeController.ActiveMode == InteractionMode.Erase)
            {
                brushController.EndStroke();
            }

            // Finalize ROI interaction
            if (roiController != null)
            {
                roiController.MouseUp();

                if (interactionModeController.ActiveMode == InteractionMode.Roi)
                {
                    if (!projectPresenter.TryGetImageItem(currentSelectedImageGuid, out var it))
                        return;
                    it.Roi = roiController.GetRoundedRoi();
                }
            }

            // Finalize viewport panning
            viewportController.EndPan();

            // Restore base interaction mode
            // (e.g. after temporary Pan override)
            interactionModeController.PopTemporaryMode();

            // Clear dragg state
            objectDetectionController.EndDrag();
            mainPictureBox.Capture = false;

            mainPictureBox.Invalidate();
        }

        private void DisplayPictureBox_MouseWheel(object? sender, MouseEventArgs e)
        {
            viewportController.Zoom(e.Delta, e.Location, viewport);
            mainPictureBox.Invalidate();
        }

        #endregion

        #region Forms Overlay to lock out user interaction during long operations

        private Panel? overlayPanel;

        private void ShowOverlay()
        {
            if (overlayPanel != null)
                return;

            overlayPanel = new Panel
            {
                Dock = DockStyle.Fill,
                BackColor = Color.FromArgb(120, 0, 0, 0),
                Enabled = false
            };

            var spinner = new PulsingSpinner
            {
                AccentColor = Color.DeepSkyBlue, // Change this to match your theme
                Radius = 20,
                DotSize = 8
            };

            var label = new Label
            {
                Text = "In progress...",
                ForeColor = Color.White,
                BackColor = Color.Transparent,
                Font = new Font("Segoe UI", 14, FontStyle.Bold),
                AutoSize = true
            };

            overlayPanel.Controls.Add(spinner);
            overlayPanel.Controls.Add(label);

            overlayPanel.Resize += (s, e) =>
            {
                spinner.Left = (overlayPanel.Width - spinner.Width) / 2;
                spinner.Top = (overlayPanel.Height / 2) - 40;

                label.Left = (overlayPanel.Width - label.Width) / 2;
                label.Top = spinner.Bottom + 10;
            };

            Controls.Add(overlayPanel);
            overlayPanel.BringToFront();
        }

        private void HideOverlay()
        {
            overlayPanel?.Dispose();
            overlayPanel = null;
        }

        #endregion
    }
}
