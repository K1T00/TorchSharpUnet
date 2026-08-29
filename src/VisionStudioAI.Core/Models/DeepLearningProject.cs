using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace VisionStudioAI.Core.Models
{
    /// <summary>
    /// persistent domain model (what exists). DTO
    /// Serializable data: image entries(Guid, relative path), ROI, split, notes, any per-item metadata.
    /// No disposable resources, no GDI objects, no caches, no UI concerns.
    /// </summary>
    public sealed class DeepLearningProject
    {
        public string Name { get; set; } = string.Empty;

        public AppUseCaseMode UseCaseMode { get; set; }

        public List<ImageItem> Images { get; set; } = new List<ImageItem>();

        public List<Feature> Features { get; set; } = new List<Feature>();

        public DeepLearningSettings Settings { get; set; } = new DeepLearningSettings();

        public DeepLearningProject()
        {
        }

        public void CopyFrom(DeepLearningProject other)
        {
            if (other == null) return;

            this.Name = other.Name;
            this.UseCaseMode = other.UseCaseMode;
            this.Features = other.Features.ToList() ?? new List<Feature>();
            this.Images = other.Images.ToList() ?? new List<ImageItem>();
            this.Settings.CopyFrom(other.Settings);
        }
    }

    public sealed class ImageItem
    {
        /// <summary>
        /// Stable id for cross-referencing
        /// </summary>
        public Guid Guid { get; set; } = Guid.NewGuid();

        /// <summary>
        /// Path to image file. Should only be used for transition original source files to project.
        /// Use relative paths + GUID + extension
        /// </summary>
        public string Path { get; set; } = string.Empty;

        public Size ImageSize { get; set; }

        /// <summary>
        /// Region of Interest in image pixel coordinates.
        /// </summary>
        public Rectangle Roi { get; set; }

        /// <summary>
        /// Dataset split/category.
        /// </summary>
        public DatasetSplit Split { get; set; } = DatasetSplit.Train;

        /// <summary>
        /// Image-level label used by the weakly supervised anomaly detector.
        /// Existing projects deserialize to Ok because it is the zero value.
        /// </summary>
        public AnomalyLabel AnomalyLabel { get; set; } = AnomalyLabel.Ok;

        public double AnomalyScore { get; set; }

        public AnomalyLabel PredictedAnomalyLabel { get; set; } = AnomalyLabel.Ok;

        /// <summary>
        /// Per-feature segmentation statistics.
        /// Key = featureId (classId). Background (0) is omitted.
        /// </summary>
        public Dictionary<int, SegmentationStats> SegmentationStats { get; set; } = new Dictionary<int, SegmentationStats>();

        /// <summary>
        /// Per-image object annotations for object detection workflows.
        /// One image can contain multiple annotated objects.
        /// </summary>
        public List<ObjectLocation> AnnotationObjectLocations { get; set; } = new List<ObjectLocation>();

        /// <summary>
        /// Per-image object inference results for object detection workflows.
        /// One image can contain multiple objects.
        /// </summary>
        public List<ObjectLocation> InferenceObjectLocations { get; set; } = new List<ObjectLocation>();

        /// <summary>
        /// Computation time in milliseconds per image during inference.
        /// </summary>
        public double InferenceMs { get; set; }

        /// <summary>
        /// When was this image added to the project.
        /// </summary>
        public DateTimeOffset AddedAt { get; set; } = DateTimeOffset.Now;
    }

    /// <summary>
    /// One image-level anomaly sample. Tile paths remain grouped so a not-OK
    /// label is applied to the complete image rather than incorrectly to every tile.
    /// </summary>
    public sealed class AnomalyImageSample
    {
        public Guid ImageId { get; set; }
        public AnomalyLabel Label { get; set; }
        public List<string> TilePaths { get; set; } = new List<string>();
    }

    public sealed class Feature
    {
        /// <summary>Zero-based class id. Convention: 0 = background.</summary>
        public int ClassId { get; set; }

        /// <summary>Display name of the class.</summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>ARGB color stored as 32-bit integer for stable JSON (e.g., 0xFFRRGGBB).</summary>
        public int Argb { get; set; }
    }

    /// <summary>
    /// One object instance in an image for object detection workflows.
    /// Diameter is intentionally not stored here because it is a global setting.
    /// </summary>
    public sealed class ObjectLocation
    {
        /// <summary>
        /// References the project feature/class.
        /// </summary>
        public int ClassId { get; set; }

        /// <summary>
        /// Center position of the annotated object in image pixel coordinates.
        /// </summary>
        public Point Center { get; set; }

        /// <summary>
        /// Probability/confidence score of the detection, as a percentage from 0 to 100.
        /// </summary>
        public double ProbabilityPercent { get; set; }
    }

    public class SegmentationStats
    {
        /// <summary>
        /// True Positives: predicted = 1 & ground truth = 1
        /// </summary>
        public int TP { get; set; }

        /// <summary>
        /// False Positives: predicted = 1 & ground truth = 0
        /// </summary>
        public int FP { get; set; }

        /// <summary>
        /// False Negatives: predicted = 0 & ground truth = 1
        /// </summary>
        public int FN { get; set; }

        /// <summary>
        /// True Negatives: predicted = 0 & ground truth = 0
        /// </summary>
        public int TN { get; set; }

        /// <summary>
        /// (TP + TN) / (TP + TN + FP + FN) := Fraction of correctly classified pixels
        /// In multi-class, average accuracy over all classes.
        /// </summary>
        public double Accuracy { get; set; }

        /// <summary>
        /// TP / (TP + FP) := How reliable positive predictions are
        /// </summary>
        public double Precision { get; set; }

        /// <summary>
        /// TP / (TP + FN) := How much of the true object was found
        /// </summary>
        public double Recall { get; set; } /// Sensitivity

        /// <summary>
        /// TN / (TN + FP) := How well false positives are avoided
        /// </summary>
        public double Specificity { get; set; }

        /// <summary>
        /// 2 * TP / (2 * TP + FP + FN) := Overlap between predicted and ground truth masks
        /// In multi-class, average Dice over all classes.
        /// </summary>
        public double Dice { get; set; } /// F1 Score

        /// <summary>
        /// TP / (TP + FP + FN) := Intersection over Union between predicted and ground truth masks
        /// In multi-class, average IoU over all classes.
        /// </summary>
        public double IoU { get; set; }

        /// <summary>
        /// FP / (FP + TN) := Fraction of negative pixels incorrectly classified as positive
        /// </summary>
        public double FPR { get; set; }


    }
}
