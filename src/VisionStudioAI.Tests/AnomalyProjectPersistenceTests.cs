using System;
using System.Drawing;
using System.IO;
using Microsoft.Extensions.DependencyInjection;
using VisionStudioAI.Core;
using VisionStudioAI.Core.Models;
using VisionStudioAI.Core.Services;
using Xunit;

namespace VisionStudioAI.Tests
{
    public sealed class AnomalyProjectPersistenceTests
    {
        [Fact]
        public void SaveProject_AnomalyDetection_DoesNotCreateAnnotationOrMaskFiles()
        {
            var root = CreateTemporaryDirectory();
            var sourceDirectory = CreateTemporaryDirectory();
            var sourcePath = Path.Combine(sourceDirectory, "source.png");
            File.WriteAllBytes(sourcePath, new byte[] { 1, 2, 3, 4 });

            try
            {
                using (var services = BuildServices())
                using (var repository = new ImageRepository())
                {
                    var presenter = ConfigureProject(services, root, AppUseCaseMode.AnomalyDetection);
                    var item = presenter.AddImage(sourcePath, new Size(16, 16));

                    presenter.SaveProject(repository);

                    Assert.True(File.Exists(Path.Combine(
                        presenter.Paths.Images,
                        item.Guid + presenter.Paths.ImagesExt)));
                    Assert.False(File.Exists(Path.Combine(
                        presenter.Paths.Annotations,
                        item.Guid + presenter.Paths.ImagesExt)));
                    Assert.False(File.Exists(Path.Combine(
                        presenter.Paths.Masks,
                        item.Guid + presenter.Paths.ImagesExt)));
                }
            }
            finally
            {
                Directory.Delete(root, recursive: true);
                Directory.Delete(sourceDirectory, recursive: true);
            }
        }

        [Fact]
        public void SaveProject_Segmentation_StillCreatesAnnotationAndMaskFiles()
        {
            var root = CreateTemporaryDirectory();
            var sourceDirectory = CreateTemporaryDirectory();
            var sourcePath = Path.Combine(sourceDirectory, "source.png");
            File.WriteAllBytes(sourcePath, new byte[] { 1, 2, 3, 4 });

            try
            {
                using (var services = BuildServices())
                using (var repository = new ImageRepository())
                {
                    var presenter = ConfigureProject(services, root, AppUseCaseMode.Segmentation);
                    var item = presenter.AddImage(sourcePath, new Size(16, 16));

                    presenter.SaveProject(repository);

                    Assert.True(File.Exists(Path.Combine(
                        presenter.Paths.Annotations,
                        item.Guid + presenter.Paths.ImagesExt)));
                    Assert.True(File.Exists(Path.Combine(
                        presenter.Paths.Masks,
                        item.Guid + presenter.Paths.ImagesExt)));
                }
            }
            finally
            {
                Directory.Delete(root, recursive: true);
                Directory.Delete(sourceDirectory, recursive: true);
            }
        }

        [Fact]
        public void ImageRepository_CanOmitAnomalyAnnotationBuffers()
        {
            using (var repository = new ImageRepository())
            {
                var item = new ImageItem { ImageSize = new Size(16, 16) };

                var runtime = repository.GetRuntime(item, ensureAnnotationData: false);

                Assert.False(runtime.HasAnnotation);
                Assert.False(runtime.HasMask);

                repository.GetRuntime(item, ensureAnnotationData: true);

                Assert.True(runtime.HasAnnotation);
                Assert.True(runtime.HasMask);
            }
        }

        private static ServiceProvider BuildServices()
        {
            var services = new ServiceCollection();
            services.AddCoreServices();
            return services.BuildServiceProvider();
        }

        private static IProjectPresenter ConfigureProject(
            ServiceProvider services,
            string root,
            AppUseCaseMode useCaseMode)
        {
            var presenter = services.GetRequiredService<IProjectPresenter>();
            presenter.ProjectPath = root;
            presenter.Project.Name = "persistence-test";
            presenter.Project.UseCaseMode = useCaseMode;
            services.GetRequiredService<IProjectOptionsService>().EnsureAll(root);
            return presenter;
        }

        private static string CreateTemporaryDirectory()
        {
            var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(path);
            return path;
        }
    }
}
