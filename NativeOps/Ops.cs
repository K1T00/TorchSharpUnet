using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;


namespace NativeOps
{
	public class Ops
	{

		[DllImport("TorchOps.dll")]
		private static extern IntPtr torch_check_last_err();

		[DllImport("TorchOps.dll")]
		private static extern void torch_empty_cache();


		public static void CheckForErrors()
		{
			var error = torch_check_last_err();
			if (error != IntPtr.Zero)
				throw new ExternalException(Marshal.PtrToStringAnsi(error));
		}

		public static void cuda_empty_cache()
		{
			torch_empty_cache();
			CheckForErrors();
		}




	}

}
