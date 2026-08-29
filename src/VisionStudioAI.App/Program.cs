using VisionStudioAI.App.Forms;
using Microsoft.Extensions.DependencyInjection;

namespace VisionStudioAI.App
{
    internal static class Program
    {
        /// <summary>
        ///  The main entry point for the application.
        /// </summary>
        [STAThread]
        static void Main()
        {
            Application.SetHighDpiMode(HighDpiMode.PerMonitorV2);
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.SetColorMode(SystemColorMode.System);

            // Bootstrap DI container
            var serviceProvider = ServiceRegistration.ConfigureServices();

            // User has to choose the use case mode before the main form is shown, so we show the StartupModeForm first.
            using var startupForm = serviceProvider.GetRequiredService<StartupModeForm>();

            var result = startupForm.ShowDialog();
            if (result != DialogResult.OK || startupForm.SelectedMode is null)
                return;

            var mainForm = ActivatorUtilities.CreateInstance<MainForm>(serviceProvider, startupForm.SelectedMode.Value, startupForm.StartupProjectPath ?? string.Empty);

            Application.Run(mainForm);
        }
    }
}