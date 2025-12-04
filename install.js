module.exports = {
  run: [
    
    // Edit this step with your custom install commands
    {
      method: "shell.run",
      params: {
        venv: "venv",                // Edit this to customize the venv folder path
        message: [
          ".\\setup.bat"
        ],
      }
    },
  ]
}
