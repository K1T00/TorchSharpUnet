using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.IO.Compression;
using System.Text.Json;

namespace NativeOps
{

	// https://download.pytorch.org/whl/ -> torch ->
	// https://download.pytorch.org/whl/torch/ -> torch version + cuda version + python version (doesn't matter)
	// torch-2.4.0+cu121-cp312-cp312-linux_x86_64.whl
	// add .zip and unzip 
	// get \torch\lib\torch.dll

	public static class LibTorchLoader
	{
		private static object loadLock = new object();
		private static volatile bool loaded = false;
		private static volatile bool nativeOpsLoaded = false;
		private static string loadedPath = "";

		public static string LoadedPath => loadedPath;
		public static bool Loaded => loaded;
		public static bool NativeOpsLoaded => nativeOpsLoaded;

		public static void EnsureLoaded()
		{
			lock (loadLock)
			{
				//const string libTorch = @"C:\Users\flavi\.cache\torch\torch_2.4.0_cu121\torch\lib\torch.dll";

				var libTorch = 
					Directory.GetParent(Directory.GetCurrentDirectory())?.Parent?.Parent?.Parent?.Parent?.FullName + "\\NativeOps\\torch.dll";

				NativeLibrary.Load(libTorch);
				loaded = true;
				loadedPath = libTorch;
			
				NativeOps.Ops.cuda_empty_cache();
				nativeOpsLoaded = true;
			}
		}

	}
}
