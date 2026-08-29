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
    public sealed class ProjectPresenterImageDeletionTests
    {
        [Fact]
        public void RemoveImage_DeletesProjectCopyButPreservesOriginalSource()
        {
            var root = CreateTemporaryDirectory();
            var sourceDirectory = CreateTemporaryDirectory();
            var sourcePath = Path.Combine(sourceDirectory, "original.jpg");
            File.WriteAllBytes(sourcePath, new byte[] { 1, 2, 3, 4 });

            try
            {
                using (var services = BuildServices())
                using (var repository = new ImageRepository())
                {
                    var presenter = ConfigureProject(services, root);
                    var item = presenter.AddImage(sourcePath, new Size(16, 16));
                    presenter.SaveProject(repository);
                    var projectCopy = item.Path;

                    Assert.True(File.Exists(projectCopy));
                    presenter.RemoveImage(item.Guid);

                    Assert.False(File.Exists(projectCopy));
                    Assert.True(File.Exists(sourcePath));
                }
            }
            finally
            {
                Directory.Delete(root, recursive: true);
                Directory.Delete(sourceDirectory, recursive: true);
            }
        }

        [Fact]
        public void RemoveImage_DoesNotDeleteUnsavedExternalSource()
        {
            var root = CreateTemporaryDirectory();
            var sourceDirectory = CreateTemporaryDirectory();
            var sourcePath = Path.Combine(sourceDirectory, "original.png");
            File.WriteAllBytes(sourcePath, new byte[] { 1, 2, 3, 4 });

            try
            {
                using (var services = BuildServices())
                {
                    var presenter = ConfigureProject(services, root);
                    var item = presenter.AddImage(sourcePath, new Size(16, 16));

                    presenter.RemoveImage(item.Guid);

                    Assert.True(File.Exists(sourcePath));
                }
            }
            finally
            {
                Directory.Delete(root, recursive: true);
                Directory.Delete(sourceDirectory, recursive: true);
            }
        }

        [Fact]
        public void RemoveImage_DeletesGuidImageWithNonDefaultExtensionFromProjectImagesFolder()
        {
            var root = CreateTemporaryDirectory();

            try
            {
                using (var services = BuildServices())
                {
                    var presenter = ConfigureProject(services, root);
                    var item = presenter.AddImage("unused", new Size(16, 16));
                    var projectImage = Path.Combine(
                        presenter.Paths.Images,
                        item.Guid + ".jpg");
                    File.WriteAllBytes(projectImage, new byte[] { 1, 2, 3, 4 });
                    item.Path = projectImage;

                    presenter.RemoveImage(item.Guid);

                    Assert.False(File.Exists(projectImage));
                }
            }
            finally
            {
                Directory.Delete(root, recursive: true);
            }
        }

        [Fact]
        public void RemoveImage_DeletesReadOnlyProjectImage()
        {
            var root = CreateTemporaryDirectory();

            try
            {
                using (var services = BuildServices())
                {
                    var presenter = ConfigureProject(services, root);
                    var item = presenter.AddImage("unused", new Size(16, 16));
                    var projectImage = Path.Combine(
                        presenter.Paths.Images,
                        item.Guid + presenter.Paths.ImagesExt);
                    File.WriteAllBytes(projectImage, new byte[] { 1, 2, 3, 4 });
                    File.SetAttributes(
                        projectImage,
                        File.GetAttributes(projectImage) | FileAttributes.ReadOnly);
                    item.Path = projectImage;

                    presenter.RemoveImage(item.Guid);

                    Assert.False(File.Exists(projectImage));
                }
            }
            finally
            {
                Directory.Delete(root, recursive: true);
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
            string root)
        {
            var presenter = services.GetRequiredService<IProjectPresenter>();
            presenter.ProjectPath = root;
            presenter.Project.Name = "image-deletion-test";
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
