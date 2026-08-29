using VisionStudioAI.Core.Models;
using System;
using System.Collections.Generic;
using System.Threading;
using static VisionStudioAI.ML.Utils.TensorProcessing.TensorComputations;
using static TorchSharp.torch;
using static TorchSharp.torchvision.transforms.functional;

namespace VisionStudioAI.ML.Utils
{
    /// <summary>
    /// Interface for paired image and mask transformations.
    /// Wrappers are needed because we may need to set up different transforms for images and masks.
    /// Implementations must keep the spatial relationship between image and mask intact.
    /// </summary>
    internal interface IPairedTransform
    {
        /// <summary>
        /// Applies an augmentation to one image tensor and its matching mask tensor.
        /// Image tensors are expected to contain normalized values in the 0..1 range.
        /// Geometric transforms must use the same sampled parameters for both tensors.
        /// </summary>
        (Tensor image, Tensor mask) Apply(Tensor image, Tensor mask);
    }

    /// <summary>
    /// Runs multiple paired transforms in sequence and returns the final image/mask tensors.
    /// </summary>
    internal class ComposePairedTransforms : IPairedTransform
    {
        private readonly List<IPairedTransform> transforms;

        public ComposePairedTransforms(List<IPairedTransform> transforms)
        {
            this.transforms = transforms;
        }

        public (Tensor image, Tensor mask) Apply(Tensor image, Tensor mask)
        {
            // Intermediate tensors created by TorchSharp transforms can hold native memory.
            // The dispose scope releases those intermediates while preserving the final result.
            using (var scope = NewDisposeScope())
            {
                foreach (var t in transforms)
                    (image, mask) = t.Apply(image, mask);

                return (image.MoveToOuterDisposeScope(), mask.MoveToOuterDisposeScope());
            }
        }
    }

    internal static class ImageAugmentations
    {
        /// <summary>
        /// Returns a composition of paired augmentations for training.
        /// Geometric transforms are paired because image and mask must stay aligned.
        /// Photometric transforms are image-only because masks contain labels, not colors.
        /// </summary>
        public static IPairedTransform BuildAugmentations(AugmentationSettings s)
        {
            var list = new List<IPairedTransform>();

            // --- Geometric (paired) ---
            if (s.FlipHorizontal) list.Add(new PairedHorizontalFlip(0.5));
            if (s.FlipVertical) list.Add(new PairedVerticalFlip(0.5));

            if (s.Rotation != 0 || s.RelativeTranslation != 0 ||
                s.MinScale > 0 || s.MaxScale > 0 ||
                s.HorizontalShear != 0 || s.VerticalShear != 0)
            {
                list.Add(new PairedRandomAffine(s));
            }

            // --- Photometric (image only) ---
            if (s.Brightness > 0) list.Add(new AdjustBrightness(s.Brightness));
            if (s.Contrast > 0) list.Add(new AdjustContrast(s.Contrast));
            if (s.Luminance > 0) list.Add(new AdjustLuminance(s.Luminance));
            if (s.Noise > 0) list.Add(new AddNoise(s.Noise));
            if (s.GaussianBlur > 0) list.Add(new GaussianBlurImageOnly(s.GaussianBlur));
            //if (s.Gamma > 0) list.Add(new AdjustGamma(s.Gamma)); // ToDo
            //if (s.Hue > 0) list.Add(new AdjustHue(s.Hue)); // ToDo

            return new ComposePairedTransforms(list);
        }
    }

    internal static class AugmentationRandom
    {
        // Each worker/thread gets its own Random instance, avoiding contention and repeated
        // sequences when data loaders access augmentations concurrently.
        private static int seed = Environment.TickCount;

        [ThreadStatic]
        private static Random current;

        public static Random Current
        {
            get
            {
                if (current == null)
                    current = new Random(Interlocked.Increment(ref seed));

                return current;
            }
        }
    }

    #region Geometric transforms (paired image and mask)

    /// <summary>
    /// Randomly mirrors both image and mask along the width axis.
    /// </summary>
    internal class PairedHorizontalFlip : IPairedTransform
    {
        private readonly double p;

        public PairedHorizontalFlip(double probability)
        {
            this.p = probability;
        }

        public (Tensor image, Tensor mask) Apply(Tensor image, Tensor mask)
        {
            var rng = AugmentationRandom.Current;
            if (rng.NextDouble() < p)
            {
                // The same random decision is applied to both tensors so labels remain aligned.
                image = hflip(image);
                mask = hflip(mask);
            }

            return (image, mask);
        }
    }

    /// <summary>
    /// Randomly mirrors both image and mask along the height axis.
    /// </summary>
    internal class PairedVerticalFlip : IPairedTransform
    {
        private readonly double p;

        public PairedVerticalFlip(double probability)
        {
            this.p = probability;
        }

        public (Tensor image, Tensor mask) Apply(Tensor image, Tensor mask)
        {
            var rng = AugmentationRandom.Current;
            if (rng.NextDouble() < p)
            {
                // The same random decision is applied to both tensors so labels remain aligned.
                image = vflip(image);
                mask = vflip(mask);
            }

            return (image, mask);
        }
    }

    /// <summary>
    /// Applies one sampled affine transform to both image and mask:
    /// rotation, translation, scale, and shear.
    /// </summary>
    internal class PairedRandomAffine : IPairedTransform
    {
        private readonly int maxDeg, maxTrans, minScale, maxScale, shearX, shearY;

        public PairedRandomAffine(AugmentationSettings s)
        {
            this.maxDeg = s.Rotation;
            this.maxTrans = s.RelativeTranslation;
            ResolveScaleRange(s, out this.minScale, out this.maxScale);
            this.shearX = s.HorizontalShear;
            this.shearY = s.VerticalShear;
        }

        public (Tensor image, Tensor mask) Apply(Tensor image, Tensor mask)
        {
            var rng = AugmentationRandom.Current;

            // Sample one set of geometric parameters and reuse it for the image and mask.
            var angle = (float)((rng.NextDouble() * 2 - 1) * maxDeg);
            var scale = (float)(rng.Next(minScale, maxScale + 1) / 100.0);

            int tx = 0, ty = 0;
            if (maxTrans > 0)
            {
                // image.shape is channel-height-width, so width is index 2 and height is index 1.
                tx = (int)((rng.NextDouble() * 2 - 1) * maxTrans / 100.0 * image.shape[2]);
                ty = (int)((rng.NextDouble() * 2 - 1) * maxTrans / 100.0 * image.shape[1]);
            }

            var sx = (float)((rng.NextDouble() * 2 - 1) * shearX);
            var sy = (float)((rng.NextDouble() * 2 - 1) * shearY);

            // Images use bilinear interpolation for smoother pixels; masks use nearest-neighbor
            // interpolation so class ids are not blended into invalid label values.
            image = SafeAffine(image, angle, new int[] { tx, ty }, scale, new float[] { sx, sy }, InterpolationMode.Bilinear);
            mask = SafeAffine(mask, angle, new int[] { tx, ty }, scale, new float[] { sx, sy }, InterpolationMode.Nearest);

            return (image, mask);
        }

        private static void ResolveScaleRange(AugmentationSettings s, out int minScale, out int maxScale)
        {
            // A scale value of 0 means "not configured"; default to 100% so affine can still run
            // when rotation, translation, or shear is enabled without scaling.
            minScale = s.MinScale > 0 ? s.MinScale : 100;
            maxScale = s.MaxScale > 0 ? s.MaxScale : 100;

            if (minScale > maxScale)
            {
                throw new ArgumentException(
                    "MinScale must be less than or equal to MaxScale after defaulting unset values to 100%.");
            }
        }
    }

    #endregion

    #region Photometric transforms (image only)

    /// <summary>
    /// Multiplies image intensities by a random brightness factor and leaves the mask unchanged.
    /// </summary>
    internal class AdjustBrightness : IPairedTransform
    {
        private readonly int value;

        public AdjustBrightness(int value)
        {
            this.value = value;
        }

        public (Tensor image, Tensor mask) Apply(Tensor image, Tensor mask)
        {
            var rng = AugmentationRandom.Current;
            // value is a percentage, so 20 means factor is sampled from 0.8 to 1.2.
            var factor = 1.0 + (rng.NextDouble() * 2 - 1) * (value / 100.0);
            image = adjust_brightness(image, factor);
            return (image, mask);
        }
    }

    /// <summary>
    /// Changes contrast around the image mean and leaves the mask unchanged.
    /// </summary>
    internal class AdjustContrast : IPairedTransform
    {
        private readonly int value;

        public AdjustContrast(int value)
        {
            this.value = value;
        }

        public (Tensor image, Tensor mask) Apply(Tensor image, Tensor mask)
        {
            var rng = AugmentationRandom.Current;
            // value is a percentage, so 20 means factor is sampled from 0.8 to 1.2.
            var factor = 1.0 + (rng.NextDouble() * 2 - 1) * (value / 100.0);
            image = adjust_contrast(image, factor);
            return (image, mask);
        }
    }

    /// <summary>
    /// Adds a random intensity offset to every image channel and leaves the mask unchanged.
    /// This works for grayscale and RGB tensors because it does not depend on color channels.
    /// </summary>
    internal class AdjustLuminance : IPairedTransform
    {
        private readonly int value;

        public AdjustLuminance(int value)
        {
            this.value = value;
        }

        public (Tensor image, Tensor mask) Apply(Tensor image, Tensor mask)
        {
            var rng = AugmentationRandom.Current;
            // Keep normalized image values valid after shifting the luminance.
            var offset = (float)((rng.NextDouble() * 2 - 1) * (value / 100.0));
            image = (image + offset).clamp(0, 1);
            return (image, mask);
        }
    }

    /// <summary>
    /// Applies gamma correction to image intensities and leaves the mask unchanged.
    /// Currently available for future settings but not wired into BuildAugmentations.
    /// </summary>
    internal class AdjustGamma : IPairedTransform
    {
        private readonly int value;

        public AdjustGamma(int value)
        {
            this.value = value;
        }

        public (Tensor image, Tensor mask) Apply(Tensor image, Tensor mask)
        {
            var rng = AugmentationRandom.Current;
            // Gamma values below 1 brighten midtones; values above 1 darken them.
            var gamma = 1.0 + (rng.NextDouble() * 2 - 1) * (value / 100.0);
            image = adjust_gamma(image, gamma);
            return (image, mask);
        }
    }

    /// <summary>
    /// Rotates image hue in color space and leaves the mask unchanged.
    /// Currently available for future settings but not wired into BuildAugmentations.
    /// </summary>
    internal class AdjustHue : IPairedTransform
    {
        private readonly int value;

        public AdjustHue(int value)
        {
            this.value = value;
        }

        public (Tensor image, Tensor mask) Apply(Tensor image, Tensor mask)
        {
            var rng = AugmentationRandom.Current;
            // TorchVision expects hue shift as a small normalized value, not degrees.
            var hueFactor = (rng.NextDouble() * 2 - 1) * (value / 100.0);
            image = adjust_hue(image, hueFactor);
            return (image, mask);
        }
    }

    /// <summary>
    /// Adds Gaussian noise to the image tensor and leaves the mask unchanged.
    /// </summary>
    internal class AddNoise : IPairedTransform
    {
        private readonly int std;

        public AddNoise(int std)
        {
            this.std = std;
        }

        public (Tensor image, Tensor mask) Apply(Tensor image, Tensor mask)
        {
            if (std <= 0)
                return (image, mask);

            // Noise standard deviation is configured as a percentage of the normalized 0..1 range.
            var noise = randn_like(image) * (std / 100.0f);
            image = (image + noise).clamp(0, 1);
            return (image, mask);
        }
    }

    /// <summary>
    /// Blurs the image tensor with a random sigma and leaves the mask unchanged.
    /// </summary>
    internal class GaussianBlurImageOnly : IPairedTransform
    {
        private readonly int sigma;

        public GaussianBlurImageOnly(int sigma)
        {
            this.sigma = sigma;
        }

        public (Tensor image, Tensor mask) Apply(Tensor image, Tensor mask)
        {
            var rng = AugmentationRandom.Current;
            const long kernel = 5;
            // Keep sigma above zero because Gaussian blur requires a positive standard deviation.
            var sig = (float)Math.Max(0.1, rng.NextDouble() * sigma);
            image = gaussian_blur(image, kernel, sig);
            return (image, mask);
        }
    }

    #endregion
}
