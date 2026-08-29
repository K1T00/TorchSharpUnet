using System;
using System.Collections.Generic;
using System.Threading;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace VisionStudioAI.ML.Models.AnomalyDetection
{
    /// <summary>
    /// From-scratch feature extractor trained to distinguish original OK tiles
    /// from two generated CutPaste variants. Forward returns three pretext-class
    /// logits; ExtractPatchEmbeddings returns normalized local descriptors.
    /// </summary>
    public sealed class SimilarityAnomalyModel : Module<Tensor, Tensor>
    {
        public const int PretextClassCount = 3;

        private readonly Module<Tensor, Tensor> backbone;
        private readonly Module<Tensor, Tensor> layer2Projection;
        private readonly Module<Tensor, Tensor> layer3Projection;
        private readonly Module<Tensor, Tensor> pretextHead;
        private readonly List<Action> removeHooks = new List<Action>();
        private readonly BinaryAnomalyModelConfig config;
        private Tensor layer2Features;
        private Tensor layer3Features;
        private int forwardActive;

        public SimilarityAnomalyModel(
            int inChannels,
            BinaryAnomalyModelConfig config,
            Device device)
            : base(nameof(SimilarityAnomalyModel))
        {
            if (inChannels != 1 && inChannels != 3)
                throw new ArgumentOutOfRangeException(nameof(inChannels));
            this.config = config ?? throw new ArgumentNullException(nameof(config));
            config.Validate();

            if (config.Approach != AnomalyDetectionApproach.SimilarityBased)
            {
                throw new NotSupportedException(
                    "The Golden Sample anomaly model has not been implemented yet.");
            }
            if (config.Backbone != AnomalyBackbone.ResNet18)
                throw new ArgumentOutOfRangeException(nameof(config.Backbone));

            backbone = torchvision.models.resnet18(
                weights_file: config.Initialization == WeightInitialization.Pretrained
                    ? config.PretrainedWeightsPath
                    : null,
                skipfc: true,
                device: device);

            layer2Projection = Conv2d(
                128,
                config.EmbeddingChannels,
                1,
                device: device,
                dtype: config.TrainPrecision);
            layer3Projection = Conv2d(
                256,
                config.EmbeddingChannels,
                1,
                device: device,
                dtype: config.TrainPrecision);
            pretextHead = Sequential(
                ("dropout", Dropout(config.Dropout)),
                ("classifier", Linear(
                    2 * config.EmbeddingChannels,
                    PretextClassCount,
                    device: device,
                    dtype: config.TrainPrecision)));

            RegisterFeatureHooks();
            RegisterComponents();
            this.to(device, true);
            this.to(config.TrainPrecision, true);
            ConfigureForEpoch(1);
        }

        public override Tensor forward(Tensor input)
        {
            EnterForward();
            try
            {
                using (var scope = NewDisposeScope())
                {
                    CaptureBackboneFeatures(input);
                    var embeddings = BuildEmbeddingMap();
                    var pooled = adaptive_avg_pool2d(embeddings, new long[] { 1, 1 })
                        .flatten(start_dim: 1);
                    return pretextHead.call(pooled).MoveToOuterDisposeScope();
                }
            }
            finally
            {
                ClearCapturedFeatures();
                Volatile.Write(ref forwardActive, 0);
            }
        }

        /// <summary>Returns L2-normalized descriptors [tiles,D,h,w].</summary>
        public Tensor ExtractPatchEmbeddings(Tensor input)
        {
            EnterForward();
            try
            {
                using (var scope = NewDisposeScope())
                {
                    CaptureBackboneFeatures(input);
                    return BuildEmbeddingMap().MoveToOuterDisposeScope();
                }
            }
            finally
            {
                ClearCapturedFeatures();
                Volatile.Write(ref forwardActive, 0);
            }
        }

        public void ConfigureForEpoch(int epoch)
        {
            if (epoch <= 0) throw new ArgumentOutOfRangeException(nameof(epoch));

            var trainBackbone = config.Initialization == WeightInitialization.Random ||
                                epoch > config.FrozenBackboneEpochs;
            foreach (var parameter in backbone.parameters())
                parameter.requires_grad_(trainBackbone);

            if (trainBackbone)
                backbone.train(true);
            else
                backbone.eval();

            if (config.Initialization == WeightInitialization.Pretrained &&
                config.FreezePretrainedBatchNormalization)
            {
                foreach (var namedModule in backbone.named_modules())
                {
                    var module = namedModule.Item2;
                    if (module.GetType().Name.IndexOf(
                        "BatchNorm",
                        StringComparison.OrdinalIgnoreCase) >= 0)
                    {
                        module.eval();
                    }
                }
            }
        }

        private Tensor BuildEmbeddingMap()
        {
            var feature2 = layer2Projection.call(layer2Features);
            var feature3 = layer3Projection.call(layer3Features);
            feature3 = interpolate(
                feature3,
                new long[] { feature2.shape[2], feature2.shape[3] },
                mode: InterpolationMode.Bilinear,
                align_corners: false);

            var kernel = config.PatchAggregationKernelSize;
            var padding = kernel / 2;
            var aggregated2 = avg_pool2d(feature2, kernel, stride: 1, padding: padding);
            var aggregated3 = avg_pool2d(feature3, kernel, stride: 1, padding: padding);
            return normalize(cat(new[] { aggregated2, aggregated3 }, 1), p: 2.0, dim: 1);
        }

        private void CaptureBackboneFeatures(Tensor input)
        {
            if (ReferenceEquals(input, null)) throw new ArgumentNullException(nameof(input));
            if (input.Dimensions != 4)
                throw new ArgumentException("Expected input shape [N,C,H,W].", nameof(input));
            if (input.shape[1] != 1 && input.shape[1] != 3)
                throw new ArgumentException("Expected one or three input channels.", nameof(input));

            var backboneInput = input.shape[1] == 1
                ? input.repeat(new long[] { 1, 3, 1, 1 })
                : input;
            backbone.call(backboneInput);

            if (ReferenceEquals(layer2Features, null) || ReferenceEquals(layer3Features, null))
                throw new InvalidOperationException("ResNet layer2/layer3 features were not captured.");
        }

        private void RegisterFeatureHooks()
        {
            foreach (var namedModule in backbone.named_modules())
            {
                var name = namedModule.Item1;
                var module = namedModule.Item2 as Module<Tensor, Tensor>;
                if (module == null) continue;

                if (name == "layer2")
                {
                    var hook = module.register_forward_hook((m, input, output) =>
                    {
                        layer2Features = output.clone().MoveToOuterDisposeScope();
                        return output;
                    });
                    removeHooks.Add(hook.remove);
                }
                else if (name == "layer3")
                {
                    var hook = module.register_forward_hook((m, input, output) =>
                    {
                        layer3Features = output.clone().MoveToOuterDisposeScope();
                        return output;
                    });
                    removeHooks.Add(hook.remove);
                }
            }

            if (removeHooks.Count != 2)
                throw new InvalidOperationException("Could not find ResNet layer2 and layer3 modules.");
        }

        private void EnterForward()
        {
            if (Interlocked.Exchange(ref forwardActive, 1) != 0)
            {
                throw new InvalidOperationException(
                    "Concurrent forwards on one anomaly model instance are not supported.");
            }
        }

        private void ClearCapturedFeatures()
        {
            layer2Features = null;
            layer3Features = null;
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                foreach (var removeHook in removeHooks) removeHook();
                removeHooks.Clear();
                backbone.Dispose();
                layer2Projection.Dispose();
                layer3Projection.Dispose();
                pretextHead.Dispose();
                ClearModules();
            }

            base.Dispose(disposing);
        }
    }
}
