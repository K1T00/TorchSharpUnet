using VisionStudioAI.ML.Models;
using VisionStudioAI.ML.Models.AnomalyDetection;
using VisionStudioAI.Core.Models;
using System;
using static TorchSharp.torch;

namespace VisionStudioAI.ML.Utils
{
    internal class BatchSizeEstimator
    {
        // The analytical ResNet18 activation graph deliberately counts every
        // retained intermediate. TorchSharp reuses/releases part of that storage,
        // so measurements otherwise undershoot the requested VRAM target by about
        // 20-25%. Keep CPU estimation conservative and calibrate CUDA only.
        private const double AnomalyGpuActivationCalibration = 0.80;

        internal enum OptimizerType
        {
            SGD,
            Adam,
            AdamW
        }

        public static int EstimateBatchSize(
            SegmentationModelConfig cfg,
            int inputWidth,
            int inputHeight,
            int numChannels,
            long availableMemoryBytes,
            ComputeDevice device,
            OptimizerType optimizer,
            double targetUtilization = 0.8)
        {
            if (availableMemoryBytes <= 0)
                throw new ArgumentException("Available memory must be > 0");

            if (numChannels <= 0)
                throw new ArgumentException("Number of channels must be > 0", nameof(numChannels));

            var bytesPerElement = GetBytesPerElement(cfg.TrainPrecision);

            // Estimate activation memory per sample
            var activationElements = EstimateActivationElements(cfg, inputWidth, inputHeight, numChannels);

            var activationCalibration = GetActivationCalibration(cfg);
            var backpropMultiplier = GetBackpropMultiplier(device, cfg, availableMemoryBytes);

            var activationBytesPerSample = activationElements * activationCalibration * bytesPerElement * backpropMultiplier;

            // Estimate parameter + optimizer overhead
            var parameterBytes = EstimateParameterBytes(cfg, numChannels, bytesPerElement);
            var optimizerBytes = EstimateOptimizerBytes(parameterBytes, optimizer);

            // Compute usable memory
            var utilization = ClampUtilization(targetUtilization, device, cfg.TrainPrecision, availableMemoryBytes);
            var usableBytes = availableMemoryBytes * utilization - optimizerBytes;

            double workspaceReserveBytes;

            if (device == ComputeDevice.Gpu)
            {
                var vramGB = availableMemoryBytes / (1024.0 * 1024 * 1024);

                // allocator + fragmentation tax
                usableBytes -= 100e6;

                if (cfg.TrainPrecision == ScalarType.Float32)
                    workspaceReserveBytes = vramGB <= 8 ? 600e6 : 400e6;
                else
                    workspaceReserveBytes = vramGB <= 8 ? 400e6 : 250e6;

                if (cfg.Architecture == SegmentationArchitecture.UNetPlusPlus)
                {
                    workspaceReserveBytes *= 0.75;
                }
            }
            else
            {
                workspaceReserveBytes = 0;
            }

            usableBytes -= workspaceReserveBytes;

            if (usableBytes <= 0)
                return 1;

            // Final batch size
            var batch = (int)Math.Floor(usableBytes / activationBytesPerSample);

            return Math.Max(batch, 1);
        }

        /// <summary>
        /// Estimates how many original OK tiles can be processed by similarity
        /// training. Each original is expanded to three pretext samples.
        /// </summary>
        public static int EstimateAnomalyTileBatchSize(
            BinaryAnomalyModelConfig cfg,
            int inputWidth,
            int inputHeight,
            int numChannels,
            long availableMemoryBytes,
            ComputeDevice device,
            OptimizerType optimizer,
            double targetUtilization = 0.85)
        {
            if (cfg == null) throw new ArgumentNullException(nameof(cfg));
            if (inputWidth <= 0) throw new ArgumentOutOfRangeException(nameof(inputWidth));
            if (inputHeight <= 0) throw new ArgumentOutOfRangeException(nameof(inputHeight));
            if (numChannels <= 0) throw new ArgumentOutOfRangeException(nameof(numChannels));
            if (availableMemoryBytes <= 0)
                throw new ArgumentException("Available memory must be > 0", nameof(availableMemoryBytes));

            cfg.Validate();
            var bytesPerElement = GetBytesPerElement(cfg.TrainPrecision);
            var activationElements = EstimateAnomalyActivationElements(
                cfg,
                inputWidth,
                inputHeight,
                numChannels);
            var backpropMultiplier = GetAnomalyBackpropMultiplier(
                device,
                cfg.TrainPrecision,
                availableMemoryBytes);
            var activationBytesPerTile =
                activationElements * bytesPerElement * backpropMultiplier;
            if (device == ComputeDevice.Gpu)
                activationBytesPerTile *= AnomalyGpuActivationCalibration;

            var parameterBytes = EstimateAnomalyParameterBytes(
                cfg,
                bytesPerElement);
            var optimizerBytes = EstimateOptimizerBytes(parameterBytes, optimizer);
            var utilization = ClampAnomalyUtilization(
                targetUtilization,
                device,
                cfg.TrainPrecision,
                availableMemoryBytes);
            var usableBytes = availableMemoryBytes * utilization;

            // Parameters, gradients, and optimizer state remain resident for the
            // complete step. Parameter storage is separate from AdamW's m/v state.
            usableBytes -= parameterBytes * 2 + optimizerBytes;

            if (device == ComputeDevice.Gpu)
            {
                var vramGb = availableMemoryBytes / (1024.0 * 1024 * 1024);
                usableBytes -= 100e6; // allocator/fragmentation reserve
                usableBytes -= cfg.TrainPrecision == ScalarType.Float32
                    ? (vramGb <= 8 ? 600e6 : 400e6)
                    : (vramGb <= 8 ? 400e6 : 250e6);
            }

            if (usableBytes <= 0 || activationBytesPerTile <= 0)
                return 1;

            return Math.Max(1, (int)Math.Floor(usableBytes / activationBytesPerTile));
        }

        /// <summary>
        /// Optional refinement step: run one dry forward/backward pass
        /// </summary>
        public static int RefineByDryRun(Func<int, bool> tryRunBatch, int initialBatch)
        {
            var batch = initialBatch;

            while (batch > 1)
            {
                if (tryRunBatch(batch))
                    return batch;

                batch /= 2;
            }
            return 1;
        }

        private static double EstimateActivationElements(SegmentationModelConfig cfg, int width, int height, int inChannels)
        {
            return cfg.Architecture == SegmentationArchitecture.UNetPlusPlus
                ? EstimateUnetPlusPlusActivationElements(cfg, width, height, inChannels)
                : EstimateUnetActivationElements(cfg, width, height, inChannels);
        }

        private static double EstimateUnetActivationElements(SegmentationModelConfig cfg, int width, int height, int inChannels)
        {
            var depth = cfg.Depth;
            var filters = cfg.FirstFilter;

            double sum = 0;
            var curW = width;
            var curH = height;
            var curC = inChannels;

            // Encoder
            for (var d = 0; d < depth; d++)
            {
                var outC = filters << d;

                // double conv output/intermediate
                sum += curW * curH * outC * 2;

                // stored skip
                sum += curW * curH * outC;

                if (cfg.UseChannelAttention || cfg.UseSelfAttention)
                    sum += curW * curH * outC * 0.25;

                curW /= 2;
                curH /= 2;
                curC = outC;
            }

            // Bottleneck
            sum += curW * curH * curC * 2;

            if (cfg.UseSelfAttention)
                sum += curW * curH * curC * 0.5;

            // Decoder
            for (var d = depth - 1; d >= 0; d--)
            {
                var outC = filters << d;

                // upsampled tensor
                sum += curW * curH * curC;

                // concatenated tensor
                sum += curW * curH * (curC + outC) * 0.65;

                // decoder double conv
                sum += curW * curH * outC * 2;

                if (cfg.UseAttentionGates)
                    sum += curW * curH * outC * 0.35;

                curW *= 2;
                curH *= 2;
                curC = outC;
            }

            return sum;
        }

        private static double EstimateUnetPlusPlusActivationElements(SegmentationModelConfig cfg, int width, int height, int inChannels)
        {
            var depth = cfg.Depth;
            var filters = cfg.FirstFilter;

            double sum = 0;

            var widths = new int[depth + 1];
            var heights = new int[depth + 1];
            var channels = new int[depth + 1];

            var curW = width;
            var curH = height;
            var curC = inChannels;

            for (var i = 0; i <= depth; i++)
            {
                var outC = filters << i;

                widths[i] = curW;
                heights[i] = curH;
                channels[i] = outC;

                sum += curW * curH * outC * 2;
                sum += curW * curH * outC * 0.6;

                if (cfg.UseChannelAttention)
                    sum += curW * curH * outC * 0.2;

                if (i < depth)
                {
                    curW /= 2;
                    curH /= 2;
                    curC = outC;
                }
            }

            if (cfg.UseSelfAttention)
            {
                sum += widths[depth] * heights[depth] * channels[depth] * 0.35;
            }

            for (var j = 1; j <= depth; j++)
            {
                for (var i = 0; i <= depth - j; i++)
                {
                    var w = widths[i];
                    var h = heights[i];
                    var outC = channels[i];
                    var lowerC = channels[i + 1];

                    sum += w * h * lowerC * 0.55;

                    var concatChannels = (j + 1) * outC;
                    sum += w * h * concatChannels * 0.45;

                    sum += w * h * outC * 1.6;

                    if (cfg.UseAttentionGates)
                        sum += w * h * outC * 0.2;
                }
            }

            return sum;
        }

        private static double EstimateParameterBytes(SegmentationModelConfig cfg, int inChannels, double bytesPerElement)
        {
            return cfg.Architecture == SegmentationArchitecture.UNetPlusPlus
                ? EstimateUnetPlusPlusParameterBytes(cfg, inChannels, bytesPerElement)
                : EstimateUnetParameterBytes(cfg, inChannels, bytesPerElement);
        }

        private static double EstimateAnomalyActivationElements(
            BinaryAnomalyModelConfig cfg,
            int width,
            int height,
            int inChannels)
        {
            // ResNet18 stem and four residual stages. The factors include block
            // intermediates/residuals as well as the layer2/layer3 hook clones.
            double sum = width * height * inChannels;
            sum += Area(width, height, 2) * 64 * 1.5;
            sum += Area(width, height, 4) * 64 * 6.0;
            sum += Area(width, height, 8) * 128 * 7.0;
            sum += Area(width, height, 16) * 256 * 7.0;
            sum += Area(width, height, 32) * 512 * 5.0;

            // Multi-scale projections, concatenation, pooling and pretext head.
            var embedding = cfg.EmbeddingChannels;
            sum += Area(width, height, 8) * embedding * 2.0;
            sum += Area(width, height, 8) * (2 * embedding) * 1.25;

            // Original, rectangular CutPaste, and scar CutPaste are forwarded
            // together. The returned capacity is expressed in original tiles.
            return sum * SimilarityAnomalyModel.PretextClassCount;
        }

        private static double EstimateAnomalyParameterBytes(
            BinaryAnomalyModelConfig cfg,
            double bytesPerElement)
        {
            // ResNet18 without its classification layer. Greyscale inputs are
            // repeated to RGB by the model, so the backbone always has 3 inputs.
            double parameters = 7 * 7 * 3 * 64;
            parameters += 2 * 2 * 9 * 64 * 64;
            parameters += ResidualStageParameters(64, 128);
            parameters += ResidualStageParameters(128, 256);
            parameters += ResidualStageParameters(256, 512);

            var embedding = cfg.EmbeddingChannels;
            parameters += 128 * embedding;
            parameters += 256 * embedding;
            parameters += (2 * embedding) * SimilarityAnomalyModel.PretextClassCount;
            return parameters * bytesPerElement;
        }

        private static double ResidualStageParameters(int inputChannels, int outputChannels)
        {
            // First block changes resolution/channels and has a 1x1 skip; the
            // second block keeps the stage width.
            return 9.0 * inputChannels * outputChannels +
                   9.0 * outputChannels * outputChannels +
                   inputChannels * outputChannels +
                   2.0 * 9.0 * outputChannels * outputChannels;
        }

        private static double Area(int width, int height, int divisor)
        {
            return Math.Max(1, width / divisor) * (double)Math.Max(1, height / divisor);
        }

        private static double EstimateUnetParameterBytes(SegmentationModelConfig cfg, int inChannels, double bytesPerElement)
        {
            var depth = cfg.Depth;
            var filters = cfg.FirstFilter;

            double paramCount = 0;
            var inC = inChannels;

            for (var d = 0; d < depth; d++)
            {
                var outC = filters << d;
                paramCount += 9 * inC * outC * 2;
                inC = outC;
            }

            paramCount += 9 * inC * inC * 2;

            for (var d = depth - 1; d >= 0; d--)
            {
                var outC = filters << d;
                paramCount += 9 * (inC + outC) * outC * 2;
                inC = outC;
            }

            return paramCount * bytesPerElement;
        }

        private static double EstimateUnetPlusPlusParameterBytes(SegmentationModelConfig cfg, int inChannels, double bytesPerElement)
        {
            var depth = cfg.Depth;
            var filters = cfg.FirstFilter;

            double paramCount = 0;
            var encoderChannels = new int[depth + 1];
            var inC = inChannels;

            // Encoder X(i,0)
            for (var i = 0; i <= depth; i++)
            {
                var outC = filters << i;
                encoderChannels[i] = outC;

                paramCount += 9 * inC * outC * 2;
                inC = outC;
            }

            // Nested decoder X(i,j)
            for (var i = 0; i < depth; i++)
            {
                var outC = encoderChannels[i];

                for (var j = 1; j <= depth - i; j++)
                {
                    var decoderInChannels = (j + 1) * outC;
                    paramCount += 9 * decoderInChannels * outC * 2;
                }
            }

            return paramCount * bytesPerElement;
        }

        private static double EstimateOptimizerBytes(double parameterBytes, OptimizerType optimizer)
        {
            switch (optimizer)
            {
                case OptimizerType.Adam:
                case OptimizerType.AdamW:
                    return parameterBytes * 2; // m + v
                default:
                    return 0;
            }
        }

        private static double GetBackpropMultiplier(ComputeDevice device, SegmentationModelConfig cfg, long availableBytes)
        {
            if (device != ComputeDevice.Gpu)
                return 2.2;

            var vramGB = availableBytes / (1024.0 * 1024 * 1024);

            if (cfg.TrainPrecision == ScalarType.Float16 || cfg.TrainPrecision == ScalarType.BFloat16)
            {
                return vramGB <= 8 ? 3.0 : 2.4;
            }

            return vramGB <= 8 ? 3.8 : 3.0;
        }

        private static double GetAnomalyBackpropMultiplier(
            ComputeDevice device,
            ScalarType precision,
            long availableBytes)
        {
            if (device != ComputeDevice.Gpu)
                return 2.4;

            var vramGb = availableBytes / (1024.0 * 1024 * 1024);
            if (precision == ScalarType.Float16 || precision == ScalarType.BFloat16)
                return vramGb <= 8 ? 2.9 : 2.5;
            return vramGb <= 8 ? 3.6 : 3.0;
        }

        private static double GetActivationCalibration(SegmentationModelConfig cfg)
        {
            if (cfg.Architecture != SegmentationArchitecture.UNetPlusPlus)
                return 1.0;

            double factor = 0.68;

            if (cfg.UseAttentionGates)
                factor *= 1.06;

            if (cfg.UseSelfAttention)
                factor *= 1.05;

            if (cfg.UseChannelAttention)
                factor *= 1.03;

            return factor;
        }

        private static double GetBytesPerElement(ScalarType precision)
        {
            switch (precision)
            {
                case ScalarType.Float16:
                case ScalarType.BFloat16:
                    return 2;
                case ScalarType.Float64:
                    return 8;
                default:
                    return 4;
            }
        }

        private static double ClampUtilization(double target, ComputeDevice device, ScalarType precision, long availableBytes)
        {
            if (device != ComputeDevice.Gpu)
                return Math.Min(target, 0.80);

            var vramGB = availableBytes / (1024.0 * 1024 * 1024);

            double max;

            if (vramGB <= 8)
                max = 0.80;
            else if (vramGB <= 12)
                max = 0.85;
            else
                max = 0.90;

            if (precision != ScalarType.Float32)
                max += 0.03;

            return Math.Min(target, max);
        }

        private static double ClampAnomalyUtilization(
            double target,
            ComputeDevice device,
            ScalarType precision,
            long availableBytes)
        {
            if (device != ComputeDevice.Gpu)
                return Math.Min(target, 0.80);

            var vramGb = availableBytes / (1024.0 * 1024 * 1024);
            var maximum = vramGb <= 8 ? 0.90 : vramGb <= 12 ? 0.91 : 0.93;
            if (precision != ScalarType.Float32)
                maximum += 0.02;
            return Math.Min(target, Math.Min(maximum, 0.95));
        }

    }
}
