using System.Net;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using Razorvine.Pickle;
using TorchSharp.PyBridge;
using static TorchSharp.torch;

const string pytorchModelFileName = "resnet18-f37072fd.pth";
const string torchSharpModelFileName = "resnet18-imagenet1k-v1.dat";
const string expectedPytorchSha256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec";

var modelUri = new Uri("https://download.pytorch.org/models/resnet18-f37072fd.pth");
var projectDirectory = FindProjectDirectory();
var modelDirectory = Path.Combine(projectDirectory, "models", "resnet18");
var pytorchModelPath = Path.Combine(modelDirectory, pytorchModelFileName);
var torchSharpModelPath = Path.Combine(modelDirectory, torchSharpModelFileName);

Directory.CreateDirectory(modelDirectory);

if (File.Exists(pytorchModelPath) && await HasExpectedPytorchHashAsync(pytorchModelPath))
{
    Console.WriteLine(
        $"PyTorch ResNet-18 weights are already available at:" +
        $"{Environment.NewLine}{pytorchModelPath}");
}
else
{
    await DownloadPytorchWeightsAsync(modelUri, pytorchModelPath);
}

if (IsValidTorchSharpModel(torchSharpModelPath))
{
    Console.WriteLine(
        $"TorchSharp ResNet-18 weights are already available at:" +
        $"{Environment.NewLine}{torchSharpModelPath}");
    return;
}

ConvertToTorchSharp(pytorchModelPath, torchSharpModelPath);

static async Task DownloadPytorchWeightsAsync(Uri modelUri, string destinationPath)
{
    var temporaryPath = destinationPath + ".download";

    try
    {
        File.Delete(temporaryPath);

        using var httpClient = new HttpClient
        {
            Timeout = TimeSpan.FromMinutes(10)
        };
        using var response = await httpClient.GetAsync(
            modelUri,
            HttpCompletionOption.ResponseHeadersRead);

        if (response.StatusCode != HttpStatusCode.OK)
        {
            throw new HttpRequestException(
                $"Downloading ResNet-18 failed with HTTP status {(int)response.StatusCode} " +
                $"({response.ReasonPhrase}).");
        }

        var totalBytes = response.Content.Headers.ContentLength;
        Console.WriteLine($"Downloading {modelUri}");

        await using (var source = await response.Content.ReadAsStreamAsync())
        await using (var destination = new FileStream(
            temporaryPath,
            FileMode.CreateNew,
            FileAccess.Write,
            FileShare.None,
            bufferSize: 81920,
            useAsync: true))
        {
            var buffer = new byte[81920];
            long downloadedBytes = 0;
            int lastReportedPercent = -1;

            while (true)
            {
                var bytesRead = await source.ReadAsync(buffer);
                if (bytesRead == 0)
                    break;

                await destination.WriteAsync(buffer.AsMemory(0, bytesRead));
                downloadedBytes += bytesRead;

                if (totalBytes is > 0)
                {
                    var percent = (int)(downloadedBytes * 100 / totalBytes.Value);
                    if (percent != lastReportedPercent && percent % 5 == 0)
                    {
                        Console.WriteLine($"{percent}%");
                        lastReportedPercent = percent;
                    }
                }
            }
        }

        if (!await HasExpectedPytorchHashAsync(temporaryPath))
        {
            throw new InvalidDataException(
                $"The downloaded file does not have the expected SHA-256 hash " +
                $"'{expectedPytorchSha256}'.");
        }

        File.Move(temporaryPath, destinationPath, overwrite: true);
        Console.WriteLine(
            $"PyTorch ResNet-18 ImageNet-1K V1 weights saved to:" +
            $"{Environment.NewLine}{destinationPath}");
    }
    finally
    {
        File.Delete(temporaryPath);
    }
}

static void ConvertToTorchSharp(string pytorchModelPath, string torchSharpModelPath)
{
    var temporaryPath = torchSharpModelPath + ".conversion";

    try
    {
        File.Delete(temporaryPath);
        Console.WriteLine("Loading the PyTorch state dictionary through PyBridge...");
        ConfigurePyBridgeForTorchVisionStateDictionary();

        using (var model = TorchSharp.torchvision.models.resnet18(
            num_classes: 1000,
            weights_file: null,
            skipfc: false,
            device: CPU))
        {
            var loadedParameters = new Dictionary<string, bool>();
            model.load_py(
                pytorchModelPath,
                strict: false,
                loadedParameters: loadedParameters);

            var unmatchedSourceKeys = loadedParameters
                .Where(parameter => !parameter.Value)
                .Select(parameter => parameter.Key)
                .OrderBy(key => key)
                .ToArray();
            var loadedKeys = loadedParameters
                .Where(parameter => parameter.Value)
                .Select(parameter => parameter.Key)
                .ToHashSet(StringComparer.Ordinal);
            var missingTargetKeys = model.state_dict().Keys
                .Where(key => !loadedKeys.Contains(key))
                .OrderBy(key => key)
                .ToArray();
            var missingRequiredTargetKeys = missingTargetKeys
                .Where(key => !key.EndsWith(
                    ".num_batches_tracked",
                    StringComparison.Ordinal))
                .ToArray();

            if (unmatchedSourceKeys.Length > 0 || missingRequiredTargetKeys.Length > 0)
            {
                throw new InvalidDataException(
                    "The PyTorch and TorchSharp ResNet-18 state dictionaries do not match." +
                    $"{Environment.NewLine}Unmatched source keys: " +
                    string.Join(", ", unmatchedSourceKeys) +
                    $"{Environment.NewLine}Missing required target keys: " +
                    string.Join(", ", missingRequiredTargetKeys));
            }

            if (missingTargetKeys.Length > 0)
            {
                Console.WriteLine(
                    "The legacy checkpoint omits BatchNorm num_batches_tracked counters; " +
                    "TorchSharp's zero-initialized values will be retained.");
            }

            model.save(temporaryPath);
        }

        if (!IsValidTorchSharpModel(temporaryPath))
        {
            throw new InvalidDataException(
                "The converted file could not be loaded by TorchSharp.");
        }

        File.Move(temporaryPath, torchSharpModelPath, overwrite: true);
        Console.WriteLine(
            $"TorchSharp ResNet-18 ImageNet-1K V1 weights saved to:" +
            $"{Environment.NewLine}{torchSharpModelPath}");
    }
    finally
    {
        File.Delete(temporaryPath);
    }
}

static void ConfigurePyBridgeForTorchVisionStateDictionary()
{
    // The official TorchVision checkpoint is a state dictionary, but its values
    // were serialized as torch.nn.Parameter objects. PyBridge normally treats
    // every Parameter as evidence of a complete model and rejects the file.
    // For a state dictionary, the first reconstruction argument is the tensor
    // that load_state_dict needs, so unwrap that Parameter to its tensor.
    RuntimeHelpers.RunClassConstructor(typeof(PyTorchUnpickler).TypeHandle);
    Unpickler.registerConstructor(
        "torch._utils",
        "_rebuild_parameter",
        new StateDictionaryParameterConstructor());
}

static bool IsValidTorchSharpModel(string modelPath)
{
    if (!File.Exists(modelPath))
        return false;

    try
    {
        using var model = TorchSharp.torchvision.models.resnet18(
            num_classes: 1000,
            weights_file: modelPath,
            skipfc: false,
            device: CPU);
        return true;
    }
    catch (Exception exception)
    {
        Console.WriteLine(
            $"Existing TorchSharp weights are invalid and will be replaced: " +
            exception.Message);
        return false;
    }
}

static string FindProjectDirectory()
{
    for (var directory = new DirectoryInfo(AppContext.BaseDirectory);
         directory != null;
         directory = directory.Parent)
    {
        if (File.Exists(Path.Combine(directory.FullName, "PyBridgeLoader.csproj")))
            return directory.FullName;
    }

    throw new DirectoryNotFoundException(
        "Could not locate the PyBridgeLoader project directory.");
}

static async Task<bool> HasExpectedPytorchHashAsync(string filePath)
{
    await using var stream = File.OpenRead(filePath);
    var hash = await SHA256.HashDataAsync(stream);
    var hashText = Convert.ToHexStringLower(hash);
    return string.Equals(hashText, expectedPytorchSha256, StringComparison.Ordinal);
}

sealed class StateDictionaryParameterConstructor : IObjectConstructor
{
    public object construct(object[] args)
    {
        if (args.Length == 0 || args[0] == null)
        {
            throw new InvalidDataException(
                "A PyTorch Parameter did not contain the expected tensor value.");
        }

        return args[0];
    }
}
