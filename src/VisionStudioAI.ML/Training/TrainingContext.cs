using VisionStudioAI.Core.Models;
using System;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;
using static TorchSharp.torch.optim.lr_scheduler;

namespace VisionStudioAI.ML.Training
{
    /// <summary>
    /// Holds optimizer / scheduler configuration and runtime objects
    /// used during training.
    /// </summary>
    internal class TrainingOptimizationContext : IDisposable
    {
        private bool disposed;

        /// <summary>
        /// Optimizer used for training.
        /// </summary>
        public Optimizer Optimizer { get; set; }

        /// <summary>
        /// Optional learning-rate scheduler.
        /// Can be null when no scheduler is used.
        /// </summary>
        public LRScheduler Scheduler { get; set; }

        /// <summary>
        /// True when the scheduler expects a monitored metric
        /// such as validation loss, e.g. ReduceLROnPlateau.
        /// </summary>
        public bool SchedulerRequiresMetric { get; set; }

        /// <summary>
        /// Initial learning rate chosen by the optimization policy.
        /// </summary>
        public double InitialLearningRate { get; set; }

        /// <summary>
        /// Weight decay chosen by the optimization policy.
        /// </summary>
        public double WeightDecay { get; set; }

        /// <summary>
        /// Optional minimum learning rate used by the scheduler.
        /// </summary>
        public double? MinLearningRate { get; set; }

        /// <summary>
        /// Human-readable optimizer name, e.g. AdamW.
        /// </summary>
        public string OptimizerName { get; set; }

        /// <summary>
        /// Human-readable scheduler name, e.g. ReduceLROnPlateau.
        /// Can be null when no scheduler is used.
        /// </summary>
        public string SchedulerName { get; set; }

        /// <summary>
        /// Optional summary string for logging/debugging so it is visible
        /// which automatic optimization policy was chosen.
        /// </summary>
        public string PolicySummary { get; set; }

        public bool LegacyMode { get; set; } = false;

        public void Dispose()
        {
            if (disposed)
                return;

            DisposeIfSupported(Scheduler);
            DisposeIfSupported(Optimizer);

            disposed = true;
        }

        private static void DisposeIfSupported(object value)
        {
            var disposable = value as IDisposable;
            if (disposable != null)
            {
                disposable.Dispose();
            }
        }
    }

    /// <summary>
    /// Training context shared between pipeline and trainer.
    /// Contains all runtime objects needed during one training run.
    /// </summary>
    internal class TrainingContext
    {
        /// <summary>
        /// Device used for training.
        /// </summary>
        public Device Device { get; set; }

        /// <summary>
        /// TorchSharp segmentation module instance.
        /// </summary>
        public Module<Tensor, Tensor> Model { get; set; }

        /// <summary>
        /// Optimizer / scheduler context.
        /// </summary>
        public TrainingOptimizationContext Optimization { get; set; }

        /// <summary>
        /// Training data loader.
        /// </summary>
        public DataLoader TrainLoader { get; set; }

        /// <summary>
        /// Validation data loader.
        /// </summary>
        public DataLoader ValLoader { get; set; }

        /// <summary>
        /// Deep-learning settings used for this run.
        /// </summary>
        public DeepLearningSettings Settings { get; set; }

        /// <summary>
        /// Monitors stopping conditions such as patience, max time,
        /// max iterations, and target validation error.
        /// </summary>
        public TrainingStopMonitor StoppingMonitor { get; set; }

        /// <summary>
        /// Binary or multiclass segmentation mode.
        /// </summary>
        public SegmentationMode SegmentationMode { get; set; }
    }
}
