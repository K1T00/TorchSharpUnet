using System;
using System.IO;
using static TorchSharp.torch;

namespace VisionStudioAI.ML.Training.AnomalyDetection
{
    /// <summary>Owns a CPU float32 matrix [normal patches, descriptor dimensions].</summary>
    public sealed class SimilarityMemoryBank : IDisposable
    {
        private bool disposed;

        public SimilarityMemoryBank(Tensor embeddings)
        {
            if (ReferenceEquals(embeddings, null)) throw new ArgumentNullException(nameof(embeddings));
            if (embeddings.Dimensions != 2 || embeddings.shape[0] == 0)
            {
                throw new ArgumentException(
                    "Memory-bank embeddings must have shape [M,D] with M > 0.",
                    nameof(embeddings));
            }

            Embeddings = embeddings.detach()
                .clone()
                .to_type(ScalarType.Float32)
                .cpu()
                .contiguous();
        }

        public Tensor Embeddings { get; private set; }
        public long Count => Embeddings.shape[0];
        public long Dimensions => Embeddings.shape[1];

        public void Save(string path)
        {
            if (string.IsNullOrWhiteSpace(path))
                throw new ArgumentException("Path is required.", nameof(path));
            var directory = Path.GetDirectoryName(path);
            if (!string.IsNullOrWhiteSpace(directory)) Directory.CreateDirectory(directory);
            save(Embeddings, path);
        }

        public static SimilarityMemoryBank Load(string path)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException("Similarity memory bank not found.", path);
            using (var loaded = load(path))
                return new SimilarityMemoryBank(loaded);
        }

        public void Dispose()
        {
            if (disposed) return;
            disposed = true;
            Embeddings?.Dispose();
            Embeddings = null;
        }
    }
}
