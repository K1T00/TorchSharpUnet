using VisionStudioAI.Core.Models;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text.Json;
using static VisionStudioAI.Core.Utils.CoreUtils;


namespace VisionStudioAI.Core.Services
{
    /// <summary>
    /// Runtime project presenter/manager.
    /// </summary>
    public class ProjectPresenter : IProjectPresenter
    {
        /// <summary>
        /// Gets the deep learning project associated with this instance.
        /// </summary>
        public DeepLearningProject Project { get; private set; }


        public event EventHandler ProjectLoaded;
        public event EventHandler<string> ErrorOccured;

        private readonly JsonSerializerOptions jsonOptions;
        private readonly IProjectOptionsService projectOptionsService;
        private ProjectPaths paths;
        private string projectPath;

        // Bound repository to MainForm's ImageRepository (Workaround for when ImageRepo fifo buffer runs out of space ...)
        private ImageRepository boundRepo;

        public ProjectPresenter(JsonSerializerOptions jsonOptions, IProjectOptionsService projectOptionsService)
        {
            this.jsonOptions = jsonOptions;
            this.projectOptionsService = projectOptionsService;
            this.Project = new DeepLearningProject();
        }

        /// <summary>
        /// After new/load/save project this is the current project path
        /// </summary>
        public string ProjectPath
        {
            get => this.projectPath;
            set
            {
                if (this.projectPath == value)
                    return;

                this.projectPath = value;

                // Invalidate derived paths
                this.paths = null;
            }
        }

        /// <summary>
        /// File paths depending on current ProjectPath and ProjectOptions
        /// </summary>
        public ProjectPaths Paths
        {
            get
            {
                if (this.paths == null)
                {
                    if (string.IsNullOrWhiteSpace(this.ProjectPath))
                        throw new InvalidOperationException("ProjectPath not set.");

                    this.paths = new ProjectPaths(this.ProjectPath, this.Project.Name, this.projectOptionsService);
                }
                return this.paths;
            }
        }

        /// <summary>
        /// Call this once from App after you create your ImageRepository, so the presenter can
        /// save dirty runtimes when the repo evicts them.
        /// </summary>
        public void BindRepository(ImageRepository repo)
        {
            if (ReferenceEquals(boundRepo, repo))
                return;

            if (boundRepo != null)
                boundRepo.RuntimeEvicting -= OnRuntimeEvicting;

            boundRepo = repo;

            if (boundRepo != null)
                boundRepo.RuntimeEvicting += OnRuntimeEvicting;
        }

        public void LoadProject(string jsonPath)
        {
            try
            {
                ProjectPath = Path.GetDirectoryName(jsonPath);
                Project.Name = Path.GetFileNameWithoutExtension(jsonPath);

                // Create folder structure if missing
                projectOptionsService.EnsureAll(ProjectPath);

                // Resolve all paths
                var paths = Paths;

                // Load project json
                var json = File.ReadAllText(paths.JsonPath);
                var loadedTemp = JsonSerializer.Deserialize<DeepLearningProject>(json, jsonOptions);

                Project.CopyFrom(loadedTemp);

                if (Project.Images == null)
                    return;
           
                ProjectLoaded?.Invoke(this, EventArgs.Empty);
            }
            catch (OperationCanceledException) { /* canceled, ignore */ }
            catch (Exception ex)
            {
                ErrorOccured?.Invoke(this, ex.Message);
            }
        }

        public void UpdateTrainingSettings(string jsonPath)
        {
            try
            {
                var json = File.ReadAllText(jsonPath);
                var settingsTemp = JsonSerializer.Deserialize<SavedModelPackage>(json, jsonOptions);

                Project.Settings.CopyFrom(settingsTemp.Settings);
            }
            catch (Exception ex)
            {
                ErrorOccured?.Invoke(this, ex.Message);
            }
        }

        public void SaveJsonFile()
        {
            var paths = Paths;

            // Save json project file
            var json = JsonSerializer.Serialize(Project, jsonOptions);
            File.WriteAllText(paths.JsonPath, json);
        }

        public void SaveProject(ImageRepository imagesRepo)
        {
            try
            {
                // Resolve all paths and create folder structure if missing
                var paths = Paths;
                projectOptionsService.EnsureAll(paths.Root);

                // Copy/save into folder/{guid}.{ext}
                foreach (var item in Project.Images)
                {
                    var id = item.Guid;

                    // All source images are expected to be in this folder, if not -> copy from original path to project folder
                    var imgProjectPath = Path.Combine(paths.Images, id + paths.ImagesExt);

                    if (!File.Exists(imgProjectPath))
                    {
                        if (File.Exists(item.Path))
                        {
                            File.Copy(item.Path, imgProjectPath, overwrite: true);
                            ClearReadOnlyAttribute(imgProjectPath);
                        }
                        else
                        {
                            ErrorOccured?.Invoke(this, $"Image file not found for image {id}: {item.Path}");
                            continue;

                        }
                    }
                    item.Path = imgProjectPath;

                    // Similarity-based anomaly detection uses image-level labels and
                    // generated heatmaps. It has no pixel annotations or ground-truth masks.
                    if (Project.UseCaseMode == AppUseCaseMode.AnomalyDetection)
                        continue;

                    // Save annotations/masks
                    var annPath = Path.Combine(paths.Annotations, id + paths.ImagesExt);
                    var maskPath = Path.Combine(paths.Masks, id + paths.ImagesExt);

                    // If runtime exists, persist from it (prefer dirty-only)
                    if (imagesRepo != null && imagesRepo.TryGetRuntime(id, out var rt))
                    {
                        SaveRuntimeIfDirtyOrMissingFiles(rt, annPath, maskPath);
                        continue;
                    }
                    // Ensure annotation/mask files exist. If missing, create blank ones.
                    EnsureBlankAnnotationExists(item, annPath);
                    EnsureBlankMaskExists(item, maskPath);
                }
                // Save json project file
                var json = JsonSerializer.Serialize(Project, jsonOptions);
                File.WriteAllText(paths.JsonPath, json);
            }
            catch (Exception ex)
            {
                ErrorOccured?.Invoke(this, ex.Message);
            }
        }

        /// <summary>
        /// Add new ImageItem DTO to project (with path only) and get back the created item with assigned GUID
        /// </summary>
        /// <param name="imageSourcePath"></param>
        /// <param name="imageSize"></param>
        /// <returns></returns>
        public ImageItem AddImage(string imageSourcePath, Size imageSize)
        {
            try
            {
                var item = new ImageItem() { 
                    Path = imageSourcePath,
                    ImageSize = imageSize,
                    Roi = new Rectangle(0, 0, imageSize.Width, imageSize.Height), // Always init new image with full Roi
                    Split = DatasetSplit.Train, // Default to train split, user can change it later
                };
                Project.Images.Add(item);

                return item;
            }
            catch (Exception ex)
            {
                ErrorOccured?.Invoke(this, ex.Message);
                return null;
            }
        }

        public void RemoveImage(Guid imageId)
        {
            if (Project == null) 
                return;

            var toRemove = FindById(Project, imageId);
            var storedImagePath = toRemove?.Path;

            // Remove from project
            if (toRemove != null)
            {
                Project.Images.Remove(toRemove);
            }

            // If project not saved yet, just return
            if (string.IsNullOrWhiteSpace(ProjectPath)) 
                return;

            var paths = Paths;

            // Remove associated files
            DeleteProjectOwnedImageFiles(paths, imageId, storedImagePath);
            SafeDeleteFile(Path.Combine(paths.Annotations, imageId + paths.ImagesExt));
            SafeDeleteFile(Path.Combine(paths.Masks, imageId + paths.ImagesExt));
        }

        private void DeleteProjectOwnedImageFiles(
            ProjectPaths paths,
            Guid imageId,
            string storedImagePath)
        {
            var imagesDirectory = Path.GetFullPath(paths.Images)
                .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            var candidates = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            if (Directory.Exists(imagesDirectory))
            {
                foreach (var path in Directory.EnumerateFiles(
                    imagesDirectory,
                    imageId.ToString("D") + ".*", // "D" specifies the standard GUID format with hyphens and no braces
                    SearchOption.TopDirectoryOnly))
                {
                    candidates.Add(Path.GetFullPath(path));
                }
            }

            if (!string.IsNullOrWhiteSpace(storedImagePath))
            {
                var resolvedStoredPath = Path.IsPathRooted(storedImagePath)
                    ? Path.GetFullPath(storedImagePath)
                    : Path.GetFullPath(Path.Combine(paths.Root, storedImagePath));

                if (IsDirectChildOfDirectory(resolvedStoredPath, imagesDirectory))
                    candidates.Add(resolvedStoredPath);
            }

            foreach (var candidate in candidates)
            {
                // Recheck every candidate immediately before deletion. This is
                // deliberately stricter than a prefix check so a similarly named
                // directory or an original import path can never be affected.
                if (!IsDirectChildOfDirectory(candidate, imagesDirectory))
                    continue;

                try
                {
                    if (File.Exists(candidate))
                    {
                        ClearReadOnlyAttribute(candidate);
                        File.Delete(candidate);
                    }
                }
                catch (Exception ex)
                {
                    ErrorOccured?.Invoke(
                        this,
                        $"Could not delete project image '{candidate}': {ex.Message}");
                }
            }
        }

        private static void ClearReadOnlyAttribute(string filePath)
        {
            var attributes = File.GetAttributes(filePath);
            if ((attributes & FileAttributes.ReadOnly) != 0)
                File.SetAttributes(filePath, attributes & ~FileAttributes.ReadOnly);
        }

        private static bool IsDirectChildOfDirectory(string filePath, string directoryPath)
        {
            var parent = Path.GetDirectoryName(Path.GetFullPath(filePath));
            if (string.IsNullOrWhiteSpace(parent))
                return false;

            return string.Equals(
                parent.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar),
                directoryPath.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar),
                StringComparison.OrdinalIgnoreCase);
        }

        public void NewProject(string jsonPath)
        {
            try
            {
                // Set up paths and create folder structure
                ProjectPath = Path.GetDirectoryName(jsonPath);
                Project.Name = Path.GetFileNameWithoutExtension(jsonPath);
                projectOptionsService.EnsureAll(ProjectPath);

                Project.CopyFrom(new DeepLearningProject());

                // Serialize json project file
                var json = JsonSerializer.Serialize(Project, jsonOptions);
                File.WriteAllText(jsonPath, json);
            }
            catch (Exception ex)
            {
                ErrorOccured?.Invoke(this, $"Error creating new project: {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Get SLICED image/mask pairs for training based on the current project and specified dataset split.
        /// </summary>
        /// <param name="split"></param>
        /// <returns></returns>
        /// <exception cref="InvalidOperationException"></exception>
        public IReadOnlyList<(string imagePath, string maskPath)> GetSlicedTrainingPairs(DatasetSplit split)
        {
            if (Project == null)
                throw new InvalidOperationException("Project not loaded.");

            var result = new List<(string imagePath, string maskPath)>();
            var paths = Paths;

            // Gather all GUIDs that are marked for training
            var trainGuids = Project.Images.Where(img => img.Split == split).Select(img => img.Guid);

            var allSliceImages = Directory
                .EnumerateFiles(paths.SlicedImages, "*" + paths.ImagesExt)
                .Select(path => new
                {
                    Path = path,
                    Name = Path.GetFileNameWithoutExtension(path)
                })
                .ToList();

            var slicedMasksByName = Directory
                .EnumerateFiles(paths.SlicedMasks, "*" + paths.ImagesExt)
                .GroupBy(Path.GetFileNameWithoutExtension, StringComparer.OrdinalIgnoreCase)
                .ToDictionary(g => g.Key, g => g.First(), StringComparer.OrdinalIgnoreCase);

            foreach (var id in trainGuids)
            {
                var tileNamePrefix = id + "_";

                // Pair image and mask tiles by exact basename, e.g. {guid}_0001.
                var sliceImages = allSliceImages
                    .Where(i => i.Name.StartsWith(tileNamePrefix, StringComparison.OrdinalIgnoreCase))
                    .OrderBy(i => i.Name, StringComparer.OrdinalIgnoreCase);

                foreach (var sliceImage in sliceImages)
                {
                    if (slicedMasksByName.TryGetValue(sliceImage.Name, out var maskPath))
                    {
                        result.Add((sliceImage.Path, maskPath));
                    }
                }
            }
            return result;
        }

        public IReadOnlyList<AnomalyImageSample> GetAnomalyTrainingSamples(DatasetSplit split)
        {
            if (Project == null)
                throw new InvalidOperationException("Project not loaded.");

            var result = new List<AnomalyImageSample>();
            var allSliceImages = Directory
                .EnumerateFiles(Paths.SlicedImages, "*" + Paths.ImagesExt)
                .Select(path => new
                {
                    Path = path,
                    Name = Path.GetFileNameWithoutExtension(path)
                })
                .ToList();

            foreach (var image in Project.Images.Where(i => i.Split == split))
            {
                var prefix = image.Guid + "_";
                var tilePaths = allSliceImages
                    .Where(i => i.Name.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                    .OrderBy(i => i.Name, StringComparer.OrdinalIgnoreCase)
                    .Select(i => i.Path)
                    .ToList();

                if (tilePaths.Count == 0)
                    continue;

                result.Add(new AnomalyImageSample
                {
                    ImageId = image.Guid,
                    Label = image.AnomalyLabel,
                    TilePaths = tilePaths
                });
            }

            return result;
        }

        public string ResolveImagePath(Guid imageGuid)
        {
            var img = Project.Images.FirstOrDefault(i => i.Guid == imageGuid);
            if (img == null)
                return null;

            var imgPath = string.Empty;


            // Project not saved yet -> no Projectpath -> use original image path
            if (ProjectPath == null)
            {
                imgPath = img.Path;
            }
            else
            {
                var paths = Paths;
                var itemPath = Path.Combine(Paths.Images, imageGuid + paths.ImagesExt);

                // Use relative path if image exists in project folder
                if (File.Exists(itemPath))
                {
                    imgPath = itemPath;
                }
                else
                {
                    // Project saved -> added new images -> new images not saved yet in project -> use original path
                    imgPath = img.Path;
                }
            }
            return imgPath;
        }

        public bool TryGetImageItem(Guid id, out ImageItem imageItem)
        {
            if (Project?.Images == null)
            {
                imageItem = null;
                return false;
            }

            imageItem = Project.Images.FirstOrDefault(i => i.Guid == id);
            return imageItem != null;
        }

        private void OnRuntimeEvicting(Guid id, ImageRuntime rt)
        {
            // If we don't have a saved project yet, we have nowhere to persist.
            // ToDo: We could consider forcing a save at this point, but for now just skip persistence on eviction if project not saved yet.
            if (string.IsNullOrWhiteSpace(ProjectPath))
                return;

            // Anomaly projects do not own annotation or ground-truth mask files.
            if (Project.UseCaseMode == AppUseCaseMode.AnomalyDetection)
                return;

            if (rt == null)
                return;

            if (!rt.IsAnnotationDirty && !rt.IsMaskDirty)
                return;

            if (!TryGetImageItem(id, out var item) || item == null)
                return;

            var paths = Paths;

            var annPath = Path.Combine(paths.Annotations, id + paths.ImagesExt);
            var maskPath = Path.Combine(paths.Masks, id + paths.ImagesExt);

            try
            {
                SaveRuntimeIfDirtyOrMissingFiles(rt, annPath, maskPath);
            }
            catch (Exception ex)
            {
                // Don't throw: eviction must proceed. But we should notify.
                ErrorOccured?.Invoke(this, $"Failed to persist evicted runtime {id}: {ex.Message}");
            }
        }

        private void SaveRuntimeIfDirtyOrMissingFiles(ImageRuntime rt, string annPath, string maskPath)
        {
            if (rt == null)
                throw new ArgumentNullException(nameof(rt));

            // Annotation: only persist after its initial load state is resolved.
            if ((rt.IsAnnotationDirty || !File.Exists(annPath)) && rt.AnnotationLoadedOnce)
            {
                rt.ReadAnnotation(bmp =>
                {
                    Directory.CreateDirectory(Path.GetDirectoryName(annPath));
                    bmp.Save(annPath, ImageFormat.Png);
                });

                rt.MarkAnnotationClean();
            }

            // Mask: only persist after its initial load state is resolved.
            // This prevents saving the placeholder zero-filled runtime mask
            // before the async disk load has completed.
            if ((rt.IsMaskDirty || !File.Exists(maskPath)) && rt.MaskLoadedOnce)
            {
                rt.ReadMask(m =>
                {
                    Directory.CreateDirectory(Path.GetDirectoryName(maskPath));
                    LabelMask.SavePng8(maskPath, m);
                });

                rt.MarkMaskClean();
            }
        }

        private void EnsureBlankAnnotationExists(ImageItem item, string annPath)
        {
            if (File.Exists(annPath))
                return;

            Directory.CreateDirectory(Path.GetDirectoryName(annPath));

            using (var bmp = new Bitmap(item.ImageSize.Width, item.ImageSize.Height, PixelFormat.Format32bppArgb))
            {
                // Transparent by default; no need to clear
                bmp.Save(annPath, ImageFormat.Png);
            }
        }

        private void EnsureBlankMaskExists(ImageItem item, string maskPath)
        {
            if (File.Exists(maskPath))
                return;

            Directory.CreateDirectory(Path.GetDirectoryName(maskPath));

            // Create an empty mask (all zeros)
            var blank = new LabelMask(item.ImageSize.Width, item.ImageSize.Height);
            LabelMask.SavePng8(maskPath, blank);
        }
    }
}
