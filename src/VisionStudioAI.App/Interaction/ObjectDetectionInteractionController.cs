using VisionStudioAI.App.Rendering;
using VisionStudioAI.Core.Interaction;
using VisionStudioAI.Core.Models;
using VisionStudioAI.Core.Services;

namespace VisionStudioAI.App.Interaction
{
    public readonly record struct ObjectDetectionEditResult(ObjectDetectionEditStatus Status, ImageRuntime? Runtime = null)
    {
        public bool Applied => Status == ObjectDetectionEditStatus.Applied;
    }

    public sealed class ObjectDetectionInteractionController
    {
        private readonly IProjectPresenter projectPresenter;
        private readonly ImageRepository imagesRepo;
        private readonly ObjectDetectionAnnotationService annotationService;

        public ObjectDetectionInteractionController(IProjectPresenter projectPresenter, ImageRepository imagesRepo, ObjectDetectionAnnotationService annotationService)
        {
            this.projectPresenter = projectPresenter;
            this.imagesRepo = imagesRepo;
            this.annotationService = annotationService;
        }

        public ObjectDetectionInteractionController(IProjectPresenter projectPresenter, ImageRepository imagesRepo)
            : this(projectPresenter, imagesRepo, new ObjectDetectionAnnotationService())
        {
        }

        public ObjectLocation? DraggedObjectLocation => annotationService.DraggedObjectLocation;

        public Point DraggedObjectOffset => annotationService.DraggedObjectOffset;

        public int GetEffectiveObjectDiameter(int baseDiameter, int downSample)
        {
            return annotationService.GetEffectiveObjectDiameter(baseDiameter, downSample);
        }

        public bool TryFindObjectOnLocation(
            Point screenPoint,
            Viewport viewport,
            ImageItem item,
            int baseDiameter,
            int downSample,
            out ObjectLocation? nearest)
        {
            nearest = null;

            if (item == null)
                return false;

            return annotationService.TryFindObjectOnLocation(
                screenPoint,
                viewport,
                item.AnnotationObjectLocations,
                baseDiameter,
                downSample,
                out nearest);
        }

        public ObjectDetectionEditResult TryBeginDrag(Guid imageId, Point screenPoint, Viewport viewport, int baseDiameter, int downSample)
        {
            if (!projectPresenter.TryGetImageItem(imageId, out var item) || item == null)
                return new ObjectDetectionEditResult(ObjectDetectionEditStatus.MissingImage);

            if (!imagesRepo.TryGetRuntime(imageId, out var rt))
                return new ObjectDetectionEditResult(ObjectDetectionEditStatus.MissingRuntime);

            var result = annotationService.TryBeginDrag(item, screenPoint, viewport, baseDiameter, downSample);

            return new ObjectDetectionEditResult(result.Status, rt);
        }

        public ObjectDetectionEditResult TryAddObjectOnLocation(
            Guid imageId,
            Point screenPoint,
            Viewport viewport,
            Feature? selectedFeature,
            IReadOnlyDictionary<int, Color> featureColorMap,
            int baseDiameter,
            int downSample)
        {
            if (selectedFeature == null)
                return new ObjectDetectionEditResult(ObjectDetectionEditStatus.MissingFeature);

            if (!projectPresenter.TryGetImageItem(imageId, out var item) || item == null)
                return new ObjectDetectionEditResult(ObjectDetectionEditStatus.MissingImage);

            if (!imagesRepo.TryGetRuntime(imageId, out var rt))
                return new ObjectDetectionEditResult(ObjectDetectionEditStatus.MissingRuntime);

            var result = annotationService.TryAddObjectOnLocation(
                item,
                screenPoint,
                viewport,
                selectedFeature,
                rt.FullImage.Size);

            if (!result.WasApplied)
                return new ObjectDetectionEditResult(result.Status, rt);

            RebuildObjectLocations(item, rt, featureColorMap, baseDiameter, downSample);

            return new ObjectDetectionEditResult(ObjectDetectionEditStatus.Applied, rt);
        }

        public ObjectDetectionEditResult TryEraseObjectOnLocation(
            Guid imageId,
            Point screenPoint,
            Viewport viewport,
            int baseDiameter,
            int downSample,
            IReadOnlyDictionary<int, Color> featureColorMap)
        {
            if (!projectPresenter.TryGetImageItem(imageId, out var item) || item == null)
                return new ObjectDetectionEditResult(ObjectDetectionEditStatus.MissingImage);

            if (!imagesRepo.TryGetRuntime(imageId, out var rt))
                return new ObjectDetectionEditResult(ObjectDetectionEditStatus.MissingRuntime);

            var result = annotationService.TryEraseObjectOnLocation(
                item,
                screenPoint,
                viewport,
                baseDiameter,
                downSample);

            if (!result.WasApplied)
                return new ObjectDetectionEditResult(result.Status, rt);

            RebuildObjectLocations(item, rt, featureColorMap, baseDiameter, downSample);

            return new ObjectDetectionEditResult(ObjectDetectionEditStatus.Applied, rt);
        }

        public ObjectDetectionEditResult TryMoveDraggedObject(
            Guid imageId,
            Point screenPoint,
            Viewport viewport,
            IReadOnlyDictionary<int, Color> featureColorMap,
            int baseDiameter,
            int downSample)
        {
            if (!projectPresenter.TryGetImageItem(imageId, out var item) || item == null)
                return new ObjectDetectionEditResult(ObjectDetectionEditStatus.MissingImage);

            if (!imagesRepo.TryGetRuntime(imageId, out var rt))
                return new ObjectDetectionEditResult(ObjectDetectionEditStatus.MissingRuntime);

            var result = annotationService.TryMoveDraggedObject(
                item,
                screenPoint,
                viewport,
                rt.FullImage.Size);

            if (!result.WasApplied)
                return new ObjectDetectionEditResult(result.Status, rt);

            RebuildObjectLocations(item, rt, featureColorMap, baseDiameter, downSample);

            return new ObjectDetectionEditResult(ObjectDetectionEditStatus.Applied, rt);
        }

        public void EndDrag()
        {
            annotationService.EndDrag();
        }

        public void RebuildObjectLocationsInferenceResults(
            ImageItem item,
            ImageRuntime rt,
            Feature? selectedFeature,
            IReadOnlyDictionary<int, Color> featureColorMap,
            int baseDiameter,
            int downSample,
            int thresholdPercent)
        {
            var visibleLocations = annotationService.GetVisibleInferenceLocations(item.InferenceObjectLocations, selectedFeature, thresholdPercent);

            rt.MutateAnnotation(bmp =>
            {
                OverlayRenderer.RebuildObjectLocationsInferenceOverlay(
                    bmp,
                    visibleLocations,
                    featureColorMap,
                    baseDiameter,
                    downSample);
            });
        }

        public void RebuildObjectLocations(
            ImageItem item,
            ImageRuntime rt,
            IReadOnlyDictionary<int, Color> featureColorMap,
            int baseDiameter,
            int downSample)
        {
            rt.MutateAnnotation(bmp =>
            {
                OverlayRenderer.RebuildObjectLocationsAnnotationOverlay(
                    bmp,
                    item.AnnotationObjectLocations,
                    featureColorMap,
                    baseDiameter,
                    downSample);
            });

            rt.MutateMask(mask =>
            {
                annotationService.RebuildObjectLocationsMask(
                    mask,
                    item.AnnotationObjectLocations,
                    baseDiameter,
                    downSample);
            });
        }

        public void RebuildLoadedObjectLocationAnnotations(IReadOnlyDictionary<int, Color> featureColorMap, int baseDiameter, int downSample)
        {
            if (projectPresenter.Project.UseCaseMode != AppUseCaseMode.ObjectDetection)
                return;

            foreach (var item in projectPresenter.Project.Images)
            {
                if (!imagesRepo.TryGetRuntime(item.Guid, out var rt))
                    continue;

                RebuildObjectLocations(item, rt, featureColorMap, baseDiameter, downSample);
            }
        }
    }
}
