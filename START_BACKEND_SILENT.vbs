' START_BACKEND_SILENT.vbs
' Runs the Python backend server invisibly in the background
' No command window will appear

Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' Get the directory where this script is located
scriptPath = fso.GetParentFolderName(WScript.ScriptFullName)

' Run the batch file in hidden mode
' 0 = hidden window, False = don't wait for completion
WshShell.Run """" & scriptPath & "\run_backend.bat""", 0, False
