using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using VisionStudioAI.Core.Models;

namespace VisionStudioAI.Core.Interaction
{
    public sealed class ObjectDetectionAnnotationService
    {
        public ObjectLocation DraggedObjectLocation { get; private set; }

        public Point DraggedObjectOffset { get; private set; }

        public int GetEffectiveObjectDiameter(int baseDiameter, int downSample)
        {
            if (baseDiameter <= 0)
                return 0;

            if (downSample < 0)
                downSample = 0;

            return baseDiameter * (1 << downSample);
        }

        public bool TryFindObjectOnLocation(
            Point screenPoint,
            Viewport viewport,
            IEnumerable<ObjectLocation> locations,
            int baseDiameter,
            int downSample,
            out ObjectLocation nearest)
        {
            if (viewport == null)
                throw new ArgumentNullException(nameof(viewport));

            if (locations == null)
                throw new ArgumentNullException(nameof(locations));

            nearest = null;

            var imagePointF = viewport.ScreenToImage(screenPoint);
            var clickPoint = new Point(
                (int)Math.Round(imagePointF.X),
                (int)Math.Round(imagePointF.Y));

            var effectiveDiameter = GetEffectiveObjectDiameter(baseDiameter, downSample);
            var hitRadius = Math.Max(4, effectiveDiameter / 2);
            var hitRadiusSq = hitRadius * hitRadius;

            var bestDistanceSq = int.MaxValue;

            foreach (var loc in locations)
            {
                var dx = loc.Center.X - clickPoint.X;
                var dy = loc.Center.Y - clickPoint.Y;
                var distSq = dx * dx + dy * dy;

                if (distSq <= hitRadiusSq && distSq < bestDistanceSq)
                {
                    bestDistanceSq = distSq;
                    nearest = loc;
                }
            }

            return nearest != null;
        }

        public ObjectDetectionAnnotationEditResult TryBeginDrag(
            ImageItem item,
            Point screenPoint,
            Viewport viewport,
            int baseDiameter,
            int downSample)
        {
            if (item == null)
                return ObjectDetectionAnnotationEditResult.MissingImage();

            if (!TryFindObjectOnLocation(
                    screenPoint,
                    viewport,
                    item.AnnotationObjectLocations,
                    baseDiameter,
                    downSample,
                    out var nearest) ||
                nearest == null)
            {
                return ObjectDetectionAnnotationEditResult.NoObjectHit();
            }

            var imagePointF = viewport.ScreenToImage(screenPoint);
            var clickPoint = new Point(
                (int)Math.Round(imagePointF.X),
                (int)Math.Round(imagePointF.Y));

            DraggedObjectLocation = nearest;
            DraggedObjectOffset = new Point(
                nearest.Center.X - clickPoint.X,
                nearest.Center.Y - clickPoint.Y);

            return ObjectDetectionAnnotationEditResult.Applied(nearest);
        }

        public ObjectDetectionAnnotationEditResult TryAddObjectOnLocation(
            ImageItem item,
            Point screenPoint,
            Viewport viewport,
            Feature selectedFeature,
            Size imageSize)
        {
            if (selectedFeature == null)
                return ObjectDetectionAnnotationEditResult.MissingFeature();

            if (item == null)
                return ObjectDetectionAnnotationEditResult.MissingImage();

            if (viewport == null)
                throw new ArgumentNullException(nameof(viewport));

            var imagePointF = viewport.ScreenToImage(screenPoint);
            var center = new Point(
                (int)Math.Round(imagePointF.X),
                (int)Math.Round(imagePointF.Y));

            if (!IsInsideImage(center, imageSize))
                return ObjectDetectionAnnotationEditResult.OutsideImage();

            var location = new ObjectLocation
            {
                ClassId = selectedFeature.ClassId,
                Center = center
            };

            item.AnnotationObjectLocations.Add(location);

            return ObjectDetectionAnnotationEditResult.Applied(location);
        }

        public ObjectDetectionAnnotationEditResult TryEraseObjectOnLocation(
            ImageItem item,
            Point screenPoint,
            Viewport viewport,
            int baseDiameter,
            int downSample)
        {
            if (item == null)
                return ObjectDetectionAnnotationEditResult.MissingImage();

            if (!TryFindObjectOnLocation(
                    screenPoint,
                    viewport,
                    item.AnnotationObjectLocations,
                    baseDiameter,
                    downSample,
                    out var nearest) ||
                nearest == null)
            {
                return ObjectDetectionAnnotationEditResult.NoObjectHit();
            }

            item.AnnotationObjectLocations.Remove(nearest);

            return ObjectDetectionAnnotationEditResult.Applied(nearest);
        }

        public ObjectDetectionAnnotationEditResult TryMoveDraggedObject(ImageItem item, Point screenPoint, Viewport viewport, Size imageSize)
        {
            if (DraggedObjectLocation == null)
                return ObjectDetectionAnnotationEditResult.NoDragInProgress();

            if (item == null)
                return ObjectDetectionAnnotationEditResult.MissingImage();

            if (viewport == null)
                throw new ArgumentNullException(nameof(viewport));

            var imagePointF = viewport.ScreenToImage(screenPoint);

            var x = (int)Math.Round(imagePointF.X) + DraggedObjectOffset.X;
            var y = (int)Math.Round(imagePointF.Y) + DraggedObjectOffset.Y;

            x = Clamp(x, 0, imageSize.Width - 1);
            y = Clamp(y, 0, imageSize.Height - 1);

            DraggedObjectLocation.Center = new Point(x, y);

            return ObjectDetectionAnnotationEditResult.Applied(DraggedObjectLocation);
        }

        public void EndDrag()
        {
            DraggedObjectLocation = null;
            DraggedObjectOffset = Point.Empty;
        }

        public IReadOnlyList<ObjectLocation> GetVisibleInferenceLocations(IEnumerable<ObjectLocation> locations, Feature selectedFeature, int thresholdPercent)
        {
            if (locations == null)
                throw new ArgumentNullException(nameof(locations));

            return locations
                .Where(loc => selectedFeature == null || loc.ClassId == selectedFeature.ClassId)
                .Where(loc => loc.ProbabilityPercent >= thresholdPercent)
                .ToList();
        }

        public void RebuildObjectLocationsMask(LabelMask mask, IEnumerable<ObjectLocation> locations, int baseDiameter, int downSample)
        {
            if (mask == null)
                throw new ArgumentNullException(nameof(mask));

            if (locations == null)
                throw new ArgumentNullException(nameof(locations));

            Array.Clear(mask.Data, 0, mask.Data.Length);

            var effectiveDiameter = GetEffectiveObjectDiameter(baseDiameter, downSample);

            foreach (var loc in locations)
            {
                var left = loc.Center.X - effectiveDiameter / 2;
                var top = loc.Center.Y - effectiveDiameter / 2;

                mask.FillRectangle(
                    left,
                    top,
                    effectiveDiameter,
                    effectiveDiameter,
                    (byte)loc.ClassId);
            }
        }

        private static bool IsInsideImage(Point point, Size imageSize)
        {
            return point.X >= 0 &&
                   point.Y >= 0 &&
                   point.X < imageSize.Width &&
                   point.Y < imageSize.Height;
        }

        private static int Clamp(int value, int min, int max)
        {
            if (value < min)
                return min;

            if (value > max)
                return max;

            return value;
        }
    }

    public sealed class ObjectDetectionAnnotationEditResult
    {
        private ObjectDetectionAnnotationEditResult(ObjectDetectionEditStatus status, ObjectLocation location)
        {
            this.Status = status;
            this.Location = location;
        }

        public ObjectDetectionEditStatus Status { get; }

        public ObjectLocation Location { get; }

        public bool WasApplied => Status == ObjectDetectionEditStatus.Applied;

        public static ObjectDetectionAnnotationEditResult Applied(ObjectLocation location)
        {
            return new ObjectDetectionAnnotationEditResult(ObjectDetectionEditStatus.Applied, location);
        }

        public static ObjectDetectionAnnotationEditResult MissingFeature()
        {
            return new ObjectDetectionAnnotationEditResult(ObjectDetectionEditStatus.MissingFeature, null);
        }

        public static ObjectDetectionAnnotationEditResult MissingImage()
        {
            return new ObjectDetectionAnnotationEditResult(ObjectDetectionEditStatus.MissingImage, null);
        }

        public static ObjectDetectionAnnotationEditResult OutsideImage()
        {
            return new ObjectDetectionAnnotationEditResult(ObjectDetectionEditStatus.OutsideImage, null);
        }

        public static ObjectDetectionAnnotationEditResult NoObjectHit()
        {
            return new ObjectDetectionAnnotationEditResult(ObjectDetectionEditStatus.NoObjectHit, null);
        }

        public static ObjectDetectionAnnotationEditResult NoDragInProgress()
        {
            return new ObjectDetectionAnnotationEditResult(ObjectDetectionEditStatus.NoDragInProgress, null);
        }
    }
}
