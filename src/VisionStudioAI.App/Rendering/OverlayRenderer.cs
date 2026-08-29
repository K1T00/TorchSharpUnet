using VisionStudioAI.Core.Interaction;
using VisionStudioAI.Core.Models;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;

namespace VisionStudioAI.App.Rendering
{
    /// <summary>
    /// Stateless renderer + fast overlay updater.
    ///
    /// Responsibilities:
    /// - Draw bitmaps in viewport space (no bitmap ownership)
    /// - Draw ROI and UI overlays
    /// - Incrementally update annotation overlay bitmap from LabelMask (dirty rect)
    ///
    /// IMPORTANT:
    /// - Callers must synchronize bitmap access (AnnotationLock / HeatmapLock) if the bitmap
    ///   can be LockBits()'d and DrawImage()'d concurrently.
    /// - This class does NOT lock; it stays UI/framework agnostic.
    /// </summary>
    public static class OverlayRenderer
    {
        /// <summary>
        /// Draw a bitmap in image space using the viewport transform.
        /// This is for "image-space" bitmaps: FullImage, AnnotationOverlay, Heatmaps that match image size.
        /// </summary>
        public static void DrawImage(Graphics g, Bitmap image, Viewport viewport)
        {
            ArgumentNullException.ThrowIfNull(g);
            ArgumentNullException.ThrowIfNull(image);
            ArgumentNullException.ThrowIfNull(viewport);

            var imgRect = viewport.ImageToScreenRect(new RectangleF(0, 0, image.Width, image.Height));

            g.InterpolationMode = InterpolationMode.NearestNeighbor;
            g.PixelOffsetMode = PixelOffsetMode.Half;

            g.DrawImage(
                image,
                imgRect.X,
                imgRect.Y,
                imgRect.Width,
                imgRect.Height);
        }

        /// <summary>
        /// Draw ROI rectangle (in image coordinates) onto the screen.
        /// </summary>
        public static void DrawRoi(Graphics g, RoiController roiController, Viewport viewport)
        {
            ArgumentNullException.ThrowIfNull(g);
            ArgumentNullException.ThrowIfNull(roiController);
            ArgumentNullException.ThrowIfNull(viewport);

            var roi = roiController.Roi;
            var roiScreen = viewport.ImageToScreenRect(roi);

            using var pen = new Pen(Color.Red, 2f);
            pen.Alignment = PenAlignment.Inset;

            g.DrawRectangle(
                pen,
                roiScreen.X,
                roiScreen.Y,
                roiScreen.Width,
                roiScreen.Height);
        }

        public static void DrawBrushIndicator(Graphics g, RoiController roiController, Viewport viewport, PictureBox mainPictureBox, int currentBrushSize)
        {
            ArgumentNullException.ThrowIfNull(g);
            ArgumentNullException.ThrowIfNull(roiController);
            ArgumentNullException.ThrowIfNull(viewport);
            ArgumentNullException.ThrowIfNull(mainPictureBox);

            //var roi = roiController.Roi;
            //var roiScreen = viewport.ImageToScreenRect(roi);

            using var pen = new Pen(Color.Blue, 4) { DashStyle = DashStyle.Solid };
            g.DrawEllipse(
                pen,
                mainPictureBox.Width / 2 - currentBrushSize / 2,
                mainPictureBox.Height / 2 - currentBrushSize / 2,
                (int)(currentBrushSize * viewport.Zoom),
                (int)(currentBrushSize * viewport.Zoom));
        }

        /// <summary>
        /// Optional: draw debug text in screen space.
        /// </summary>
        public static void DrawText(Graphics g, string text, Point screenPos)
        {
            ArgumentNullException.ThrowIfNull(g);

            using var font = new Font("Segoe UI", 9);
            using var brush = new SolidBrush(Color.White);
            g.DrawString(text, font, brush, screenPos);
        }


        /// <summary>
        /// Incrementally updates a rectangular region of an ARGB annotation overlay bitmap from the LabelMask.
        /// Used for segmentation.
        /// </summary>
        public static void UpdateAnnotationOverlayRegion(Bitmap overlay, LabelMask mask, IReadOnlyDictionary<int, Color> featureColorMap, Rectangle imageRect, byte overlayAlpha)
        {
            ArgumentNullException.ThrowIfNull(overlay);
            ArgumentNullException.ThrowIfNull(mask);
            ArgumentNullException.ThrowIfNull(featureColorMap);

            if (imageRect.Width <= 0 || imageRect.Height <= 0)
                return;

            var data = overlay.LockBits(imageRect, ImageLockMode.WriteOnly, PixelFormat.Format32bppArgb);

            try
            {
                unsafe
                {
                    var dstBase = (byte*)data.Scan0;
                    var dstStride = data.Stride;

                    for (var y = 0; y < imageRect.Height; y++)
                    {
                        var imgY = imageRect.Y + y;
                        var dstRow = dstBase + y * dstStride;
                        var maskRow = imgY * mask.Width;

                        for (var x = 0; x < imageRect.Width; x++)
                        {
                            var imgX = imageRect.X + x;
                            var classId = mask.Data[maskRow + imgX];

                            var i = x * 4;

                            if (classId == 0 || !featureColorMap.TryGetValue(classId, out var c))
                            {
                                // Background = fully transparent in visualization overlay
                                dstRow[i + 0] = 0;
                                dstRow[i + 1] = 0;
                                dstRow[i + 2] = 0;
                                dstRow[i + 3] = 0;
                            }
                            else
                            {
                                dstRow[i + 0] = c.B;
                                dstRow[i + 1] = c.G;
                                dstRow[i + 2] = c.R;
                                dstRow[i + 3] = overlayAlpha;
                            }
                        }
                    }
                }
            }
            finally
            {
                overlay.UnlockBits(data);
            }
        }

        /// <summary>
        /// Fully rebuilds the object-detection annotation overlay from persisted object centers.
        /// The stored centers stay fixed; only the rendered marker size changes with diameter/down sample.
        /// </summary>
        public static void RebuildObjectLocationsAnnotationOverlay(Bitmap overlay, IEnumerable<ObjectLocation> locations, IReadOnlyDictionary<int, Color> featureColorMap, int diameter, int downSample)
        {
            ArgumentNullException.ThrowIfNull(overlay);
            ArgumentNullException.ThrowIfNull(locations);
            ArgumentNullException.ThrowIfNull(featureColorMap);

            var effectiveDiameter = GetEffectiveObjectDiameter(diameter, downSample);

            using var g = Graphics.FromImage(overlay);
            g.Clear(Color.Transparent);
            g.SmoothingMode = SmoothingMode.AntiAlias;
            g.InterpolationMode = InterpolationMode.HighQualityBilinear;
            g.PixelOffsetMode = PixelOffsetMode.HighQuality;

            foreach (var loc in locations)
            {
                if (!featureColorMap.TryGetValue(loc.ClassId, out var color))
                    continue;

                DrawObjectLocationMarker(
                    g,
                    loc.Center,
                    effectiveDiameter,
                    color,
                    overlay.Width,
                    overlay.Height);
            }
        }

        /// <summary>
        /// Draws a semi-transparent grayscale mask (e.g. prediction or label).
        /// </summary>
        public static void DrawAnnotation(Graphics g, Bitmap mask, Viewport viewport, float opacity = 0.7f)
        {
            if (mask == null)
                return;

            ArgumentNullException.ThrowIfNull(g);
            ArgumentNullException.ThrowIfNull(viewport);

            var rect = viewport.ImageToScreenRect(
                new RectangleF(0, 0, mask.Width, mask.Height));

            using var attr = new ImageAttributes();
            var matrix = new ColorMatrix
            {
                Matrix33 = opacity
            };

            attr.SetColorMatrix(matrix);

            g.DrawImage(
                mask,
                Rectangle.Round(rect),
                0,
                0,
                mask.Width,
                mask.Height,
                GraphicsUnit.Pixel,
                attr);
        }

     
        // Draw feature size rectangle at top-left of the PictureBox
        public static void DrawSliceSizeRectangle(Graphics g, RoiController roiController, Viewport viewport, int sliceSize, int downSample)
        {
            ArgumentNullException.ThrowIfNull(g);
            ArgumentNullException.ThrowIfNull(roiController);
            ArgumentNullException.ThrowIfNull(viewport);

            var effectiveSize = sliceSize * (1 << downSample);
            var screenRect = new Rectangle(10, 10, (int)(effectiveSize * viewport.Zoom), (int)(effectiveSize * viewport.Zoom));

            using var p = new Pen(Color.Green, 2);
            g.DrawRectangle(p, screenRect);
        }

        /// <summary>
        /// Draws a heatmap bitmap aligned with the image.
        /// </summary>
        public static void DrawHeatmap(Graphics g, Bitmap heatmap, Viewport viewport, float opacity = 0.7f)
        {
            if (heatmap == null)
                return;

            ArgumentNullException.ThrowIfNull(g);
            ArgumentNullException.ThrowIfNull(viewport);

            var rect = viewport.ImageToScreenRect(
                new RectangleF(0, 0, heatmap.Width, heatmap.Height));

            using var attr = new ImageAttributes();

            // Treat pixels that are exactly black as transparent (your below-threshold pixels)
            attr.SetColorKey(Color.Black, Color.Black);
            
            var matrix = new ColorMatrix
            {
                Matrix33 = opacity
            };
            attr.SetColorMatrix(matrix);

            g.DrawImage(
                heatmap,
                Rectangle.Round(rect),
                0,
                0,
                heatmap.Width,
                heatmap.Height,
                GraphicsUnit.Pixel,
                attr);
        }

        private static void DrawObjectLocationMarker(Graphics g, Point center, int diameter, Color color, int imageWidth, int imageHeight)
        {
            if (diameter <= 0)
                return;

            var radius = diameter / 2;
            var left = center.X - radius;
            var top = center.Y - radius;
            var size = diameter;

            var bounds = new Rectangle(left, top, size, size);

            if (!bounds.IntersectsWith(new Rectangle(0, 0, imageWidth, imageHeight)))
                return;

            using var pen = new Pen(color, Math.Max(1f, diameter / 16f));
            g.DrawEllipse(pen, bounds);

            var crossHalf = Math.Max(4, diameter / 4);

            g.DrawLine(
                pen,
                Math.Max(0, center.X - crossHalf),
                center.Y,
                Math.Min(imageWidth - 1, center.X + crossHalf),
                center.Y);

            g.DrawLine(
                pen,
                center.X,
                Math.Max(0, center.Y - crossHalf),
                center.X,
                Math.Min(imageHeight - 1, center.Y + crossHalf));
        }

        public static void RebuildObjectLocationsInferenceOverlay(Bitmap overlay, IEnumerable<ObjectLocation> locations, IReadOnlyDictionary<int, Color> featureColorMap, int diameter, int downSample)
        {
            if (overlay == null)
                throw new ArgumentNullException(nameof(overlay));
            if (locations == null)
                throw new ArgumentNullException(nameof(locations));
            if (featureColorMap == null)
                throw new ArgumentNullException(nameof(featureColorMap));

            var effectiveDiameter = GetEffectiveObjectDiameter(diameter, downSample);

            using var g = Graphics.FromImage(overlay);
            g.Clear(Color.Transparent);
            g.SmoothingMode = SmoothingMode.AntiAlias;
            g.InterpolationMode = InterpolationMode.HighQualityBilinear;
            g.PixelOffsetMode = PixelOffsetMode.HighQuality;

            foreach (var loc in locations)
            {
                Color color;
                if (!featureColorMap.TryGetValue(loc.ClassId, out color))
                    continue;

                DrawObjectLocationResultMarker(
                    g,
                    loc.Center,
                    effectiveDiameter,
                    color,
                    overlay.Width,
                    overlay.Height,
                    loc.ProbabilityPercent);
            }
        }

        private static void DrawObjectLocationResultMarker(
            Graphics g,
            Point center,
            int diameter,
            Color color,
            int imageWidth,
            int imageHeight,
            double probabilityPercent)
        {
            if (diameter <= 0)
                return;

            var radius = diameter / 2;
            var left = center.X - radius;
            var top = center.Y - radius;
            var size = diameter;

            var bounds = new Rectangle(left, top, size, size);

            if (!bounds.IntersectsWith(new Rectangle(0, 0, imageWidth, imageHeight)))
                return;

            using (var pen = new Pen(color, Math.Max(1f, diameter / 16f)))
            {
                g.DrawEllipse(pen, bounds);

                var crossHalf = Math.Max(4, diameter / 4);

                g.DrawLine(
                    pen,
                    Math.Max(0, center.X - crossHalf),
                    center.Y,
                    Math.Min(imageWidth - 1, center.X + crossHalf),
                    center.Y);

                g.DrawLine(
                    pen,
                    center.X,
                    Math.Max(0, center.Y - crossHalf),
                    center.X,
                    Math.Min(imageHeight - 1, center.Y + crossHalf));
            }

            DrawObjectProbabilityLabel(g, center, diameter, imageWidth, imageHeight, probabilityPercent);
        }

        private static void DrawObjectProbabilityLabel(
            Graphics g,
            Point center,
            int diameter,
            int imageWidth,
            int imageHeight,
            double probabilityPercent)
        {
            var label = string.Format("{0:F1}%", probabilityPercent);

            using var font = new Font("Segoe UI", 12f, FontStyle.Bold);
            //using var font = new Font("Segoe UI", Math.Max(8f, diameter / 5f), FontStyle.Bold);
            using var textBrush = new SolidBrush(Color.White);
            using var backgroundBrush = new SolidBrush(Color.FromArgb(170, 0, 0, 0));
            var textSize = g.MeasureString(label, font);

            var textX = center.X - textSize.Width / 2f;
            var textY = center.Y - diameter / 2f - textSize.Height - 4f;

            if (textX < 0)
                textX = 0;

            if (textY < 0)
                textY = 0;

            if (textX + textSize.Width > imageWidth)
                textX = imageWidth - textSize.Width;

            g.FillRectangle(
                backgroundBrush,
                textX - 2f,
                textY - 1f,
                textSize.Width + 4f,
                textSize.Height + 2f);

            g.DrawString(label, font, textBrush, textX, textY);
        }


        public static int GetEffectiveObjectDiameter(int diameter, int downSample)
        {
            if (diameter <= 0)
                return 0;

            if (downSample < 0)
                downSample = 0;

            return diameter * (1 << downSample);
        }

    }
}
